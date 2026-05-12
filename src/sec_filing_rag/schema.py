"""
Extraction schema for the SEC Filing RAG.

Used with `chat_model.with_structured_output(FilingExtraction)` to pull
typed entities and relations from SEC filing sections. Field descriptions
become the LLM's instructions when LangChain compiles this into a tool schema,
so keep them precise and self-explanatory.
"""

from __future__ import annotations

from typing import Literal, Optional

from pydantic import BaseModel, Field


# ─── Entity types (graph nodes) ────────────────────────────────────────────


class Company(BaseModel):
    """A corporation mentioned in the filing — filer, subsidiary, customer,
    supplier, competitor, or acquisition target. Role is captured by relations,
    not subtypes."""

    name: str = Field(description="Canonical company name as written in the filing.")
    ticker: Optional[str] = Field(default=None, description="Stock ticker if stated. Do not infer.")
    cik: Optional[str] = Field(default=None, description="SEC CIK if known (usually only the filer).")
    aliases: list[str] = Field(default_factory=list, description="Other names used in the same filing.")


class Person(BaseModel):
    """A named individual — executive, director, auditor, plaintiff."""

    name: str
    title: Optional[str] = Field(default=None, description="Role at time of filing.")


class Product(BaseModel):
    name: str
    category: Optional[str] = Field(default=None, description="High-level category, e.g. 'cloud services'.")


class Risk(BaseModel):
    summary: str = Field(description="One-sentence noun-phrase summary of the risk.")
    category: Literal[
        "market", "operational", "financial", "regulatory", "cybersecurity",
        "supply_chain", "geopolitical", "climate", "litigation", "human_capital",
        "ai_technology", "other",
    ]


class Location(BaseModel):
    name: str
    kind: Literal["country", "region", "city", "facility"]


class LegalMatter(BaseModel):
    description: str
    status: Optional[Literal["pending", "settled", "dismissed", "ongoing", "unknown"]] = None


# ─── Relation types (graph edges) ──────────────────────────────────────────

RelationType = Literal[
    "SUBSIDIARY_OF", "EXECUTIVE_OF", "DIRECTOR_OF", "AUDITED_BY",
    "SUPPLIES", "CUSTOMER_OF", "COMPETES_WITH", "ACQUIRED", "DIVESTED",
    "OPERATES_IN", "MANUFACTURES", "DISCLOSES_RISK", "PARTY_TO_LITIGATION",
]


class Relation(BaseModel):
    """A typed edge between two entities found in the same filing chunk.

    `source` and `target` must reference the `name` of an entity that also
    appears in this extraction. Do not invent entities here.
    """

    type: RelationType
    source: str = Field(description="`name` of the source entity.")
    target: str = Field(description="`name` of the target entity.")
    evidence: str = Field(description="Verbatim quote (≤200 chars) supporting this relation.")
    confidence: Literal["high", "medium", "low"]


# ─── Top-level container ───────────────────────────────────────────────────


class FilingExtraction(BaseModel):
    """All entities and relations extracted from a single filing chunk.

    Extraction guidance for the LLM:
    - Only extract what the text explicitly supports.
    - Skip boilerplate disclaimers and forward-looking-statement notices.
    - Resolve self-references ('the Company', 'we', 'our') to the filer's
      company name provided in the prompt context.
    - Prefer fewer high-confidence extractions over exhaustive coverage.
    """

    companies: list[Company] = Field(default_factory=list)
    people: list[Person] = Field(default_factory=list)
    products: list[Product] = Field(default_factory=list)
    risks: list[Risk] = Field(default_factory=list)
    locations: list[Location] = Field(default_factory=list)
    legal_matters: list[LegalMatter] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)


# ─── Cypher schema hint (handed to GraphCypherQAChain) ─────────────────────

CYPHER_SCHEMA_HINT = """
Node labels:
  Company   {name, ticker, cik}
  Person    {name, title}
  Product   {name, category}
  Risk      {summary, category}
  Location  {name, kind}
  LegalMatter {description, status}

Relationships (all directed; all carry {evidence, confidence, filing_id, fiscal_year}):
  (Company)-[:SUBSIDIARY_OF]->(Company)
  (Person)-[:EXECUTIVE_OF|DIRECTOR_OF]->(Company)
  (Company)-[:AUDITED_BY]->(Company)
  (Company)-[:SUPPLIES|CUSTOMER_OF|COMPETES_WITH]->(Company)
  (Company)-[:ACQUIRED|DIVESTED]->(Company)
  (Company)-[:OPERATES_IN]->(Location)
  (Company)-[:MANUFACTURES]->(Product)
  (Company)-[:DISCLOSES_RISK]->(Risk)
  (Company)-[:PARTY_TO_LITIGATION]->(LegalMatter)
"""
