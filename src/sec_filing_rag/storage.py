"""
Persistence layer. Two stores in one module because they're always paired:
- Chroma for the chunk vectors (RAG path)
- Neo4j for the extracted entity graph (KG path)

Both wrappers are idempotent: re-running `build` from the CLI replaces
existing data for a (ticker, filing_type, fiscal_year) tuple rather than
duplicating it.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_neo4j import Neo4jGraph

from sec_filing_rag.config import settings
from sec_filing_rag.extraction import get_embeddings
from sec_filing_rag.schema import FilingExtraction

log = logging.getLogger(__name__)


# ─── Vector store ──────────────────────────────────────────────────────────


def get_vector_store() -> Chroma:
    """Persistent Chroma collection. Cheap to construct repeatedly."""
    return Chroma(
        collection_name=settings.chroma_collection,
        embedding_function=get_embeddings(),
        persist_directory=str(settings.chroma_dir),
    )


def upsert_chunks(chunks: Iterable[Document]) -> int:
    """Add chunks to Chroma. Uses a deterministic id so reruns replace, not duplicate."""
    store = get_vector_store()
    docs = list(chunks)
    ids = [_chunk_id(d) for d in docs]
    # Chroma upserts on matching ids.
    store.add_documents(docs, ids=ids)
    log.info("Upserted %d chunks into Chroma", len(docs))
    return len(docs)


def _chunk_id(d: Document) -> str:
    m = d.metadata
    return f"{m.get('ticker')}-{m.get('fiscal_year')}-{m.get('item_no')}-{m.get('chunk_idx')}"


# ─── Graph store ───────────────────────────────────────────────────────────


def get_graph() -> Neo4jGraph:
    return Neo4jGraph(
        url=settings.neo4j_uri,
        username=settings.neo4j_user,
        password=settings.neo4j_password,
        refresh_schema=False,
    )


# Cypher fragments — kept as constants so the graph schema is in one place.

_INIT_CONSTRAINTS = [
    "CREATE CONSTRAINT company_name IF NOT EXISTS FOR (c:Company) REQUIRE c.name IS UNIQUE",
    "CREATE CONSTRAINT person_name  IF NOT EXISTS FOR (p:Person)  REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT product_name IF NOT EXISTS FOR (p:Product) REQUIRE p.name IS UNIQUE",
    "CREATE CONSTRAINT risk_summary IF NOT EXISTS FOR (r:Risk)    REQUIRE r.summary IS UNIQUE",
    "CREATE CONSTRAINT location_name IF NOT EXISTS FOR (l:Location) REQUIRE l.name IS UNIQUE",
    "CREATE INDEX company_ticker IF NOT EXISTS FOR (c:Company) ON (c.ticker)",
]

_UPSERT_COMPANY = """
MERGE (c:Company {name: $name})
ON CREATE SET c.ticker = $ticker, c.cik = $cik
ON MATCH  SET c.ticker = coalesce(c.ticker, $ticker), c.cik = coalesce(c.cik, $cik)
"""

_UPSERT_RELATION = """
MATCH (s {name: $source}), (t {name: $target})
CALL apoc.merge.relationship(
  s, $rel_type, {},
  {evidence: $evidence, confidence: $confidence,
   filing_id: $filing_id, fiscal_year: $fiscal_year},
  t
) YIELD rel
RETURN count(rel)
"""


def init_graph_schema() -> None:
    g = get_graph()
    for stmt in _INIT_CONSTRAINTS:
        g.query(stmt)
    log.info("Graph constraints applied")


def upsert_extraction(
    extraction: FilingExtraction,
    *,
    filing_id: str,
    fiscal_year: int,
) -> None:
    """Merge a single chunk's extraction into the graph."""
    g = get_graph()

    for c in extraction.companies:
        g.query(_UPSERT_COMPANY, {"name": c.name, "ticker": c.ticker, "cik": c.cik})
    for p in extraction.people:
        g.query("MERGE (p:Person {name: $name}) SET p.title = coalesce(p.title, $title)",
                {"name": p.name, "title": p.title})
    for pr in extraction.products:
        g.query("MERGE (p:Product {name: $name}) SET p.category = coalesce(p.category, $category)",
                {"name": pr.name, "category": pr.category})
    for r in extraction.risks:
        g.query("MERGE (r:Risk {summary: $summary}) SET r.category = $category",
                {"summary": r.summary, "category": r.category})
    for loc in extraction.locations:
        g.query("MERGE (l:Location {name: $name}) SET l.kind = $kind",
                {"name": loc.name, "kind": loc.kind})

    for rel in extraction.relations:
        g.query(_UPSERT_RELATION, {
            "source": rel.source, "target": rel.target,
            "rel_type": rel.type,
            "evidence": rel.evidence, "confidence": rel.confidence,
            "filing_id": filing_id, "fiscal_year": fiscal_year,
        })
