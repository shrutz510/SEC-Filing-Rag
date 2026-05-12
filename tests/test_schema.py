"""Sanity tests for the extraction schema. Run with: pytest"""

from sec_filing_rag.schema import (
    Company,
    FilingExtraction,
    Relation,
    Risk,
)


def test_company_minimal():
    c = Company(name="Apple Inc.")
    assert c.name == "Apple Inc."
    assert c.ticker is None
    assert c.aliases == []


def test_relation_round_trip():
    rel = Relation(
        type="AUDITED_BY",
        source="Apple Inc.",
        target="PricewaterhouseCoopers LLP",
        evidence="Our independent registered public accounting firm is PricewaterhouseCoopers LLP.",
        confidence="high",
    )
    assert rel.type == "AUDITED_BY"


def test_extraction_json_schema_is_emittable():
    """LangChain uses this JSON Schema when binding `with_structured_output`."""
    schema = FilingExtraction.model_json_schema()
    assert "properties" in schema
    assert "relations" in schema["properties"]


def test_risk_category_closed_vocab():
    r = Risk(summary="dependence on a single foundry", category="supply_chain")
    assert r.category == "supply_chain"
