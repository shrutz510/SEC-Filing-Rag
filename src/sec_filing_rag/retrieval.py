"""
Three retrievers — vector, graph, hybrid. Each takes a question and
returns a uniform `RetrievalResult` so the synthesizer doesn't care which
path produced the context.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from langchain_core.documents import Document
from langchain_neo4j import GraphCypherQAChain

from sec_filing_rag.extraction import get_chat_model
from sec_filing_rag.schema import CYPHER_SCHEMA_HINT
from sec_filing_rag.storage import get_graph, get_vector_store

log = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    """Uniform return type from any retriever."""

    chunks: list[Document] = field(default_factory=list)        # text passages
    graph_facts: list[dict[str, Any]] = field(default_factory=list)  # rows from Cypher
    cypher: str | None = None                                    # query used (if any)
    mode: str = "vector"                                         # 'vector' | 'graph' | 'hybrid'


# ─── Vector ────────────────────────────────────────────────────────────────


def vector_retrieve(question: str, *, k: int = 6, filter: dict | None = None) -> RetrievalResult:
    store = get_vector_store()
    docs = store.max_marginal_relevance_search(question, k=k, fetch_k=k * 4, filter=filter)
    return RetrievalResult(chunks=docs, mode="vector")


# ─── Graph (Cypher) ────────────────────────────────────────────────────────


def graph_retrieve(question: str) -> RetrievalResult:
    """Use the LLM to write Cypher against our schema, run it, return rows."""
    graph = get_graph()
    chain = GraphCypherQAChain.from_llm(
        llm=get_chat_model(),
        graph=graph,
        verbose=False,
        return_intermediate_steps=True,
        allow_dangerous_requests=True,        # acknowledged: read-only Cypher only
        cypher_prompt_extras=CYPHER_SCHEMA_HINT,
    )
    out = chain.invoke({"query": question})

    cypher = ""
    rows: list[dict[str, Any]] = []
    for step in out.get("intermediate_steps", []):
        if "query" in step:
            cypher = step["query"]
        if "context" in step and isinstance(step["context"], list):
            rows = step["context"]

    return RetrievalResult(graph_facts=rows, cypher=cypher, mode="graph")


# ─── Hybrid ────────────────────────────────────────────────────────────────


def hybrid_retrieve(question: str, *, k: int = 6) -> RetrievalResult:
    """Run Cypher first; use the resulting company set to scope vector retrieval.

    Pattern: questions like 'how do companies audited by Deloitte describe
    climate risk?' decompose into (1) get the set of companies, (2) read
    what they actually said. Step 2 is plain RAG, but filtered.
    """
    g_result = graph_retrieve(question)

    # Pull any 'name' / 'ticker' / 'cik' fields out of the graph rows to use
    # as a metadata filter for vector retrieval.
    tickers: set[str] = set()
    for row in g_result.graph_facts:
        for v in row.values():
            if isinstance(v, dict):
                if t := v.get("ticker"):
                    tickers.add(t)

    filter_ = {"ticker": {"$in": list(tickers)}} if tickers else None
    v_result = vector_retrieve(question, k=k, filter=filter_)

    return RetrievalResult(
        chunks=v_result.chunks,
        graph_facts=g_result.graph_facts,
        cypher=g_result.cypher,
        mode="hybrid",
    )
