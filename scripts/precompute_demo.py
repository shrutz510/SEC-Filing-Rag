"""
Run a curated set of demo questions through the workflow and dump the
results to docs/data/precomputed.json so the GitHub Pages demo can show
them without a live backend.

Usage:
    python scripts/precompute_demo.py
"""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from sec_filing_rag.agent import answer

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "precomputed.json"
EVAL_FILE = ROOT / "eval" / "questions.yaml"


def serialize(state: dict) -> dict:
    r = state.get("retrieval")
    return {
        "question": state["question"],
        "query_type": state.get("query_type"),
        "answer": state.get("answer"),
        "cypher": getattr(r, "cypher", None),
        "graph_facts": getattr(r, "graph_facts", []) or [],
        "chunks": [
            {
                "ticker": d.metadata.get("ticker"),
                "fiscal_year": d.metadata.get("fiscal_year"),
                "item_no": d.metadata.get("item_no"),
                "text": d.page_content,
            }
            for d in (getattr(r, "chunks", []) or [])
        ],
    }


def main() -> None:
    questions = yaml.safe_load(EVAL_FILE.read_text())["questions"]
    out = []
    for q in questions:
        if not q.get("demo"):
            continue
        print(f"→ {q['question']}")
        state = answer(q["question"])
        out.append(serialize({**state, "question": q["question"]}))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2))
    print(f"Wrote {len(out)} demo questions to {OUT}")


if __name__ == "__main__":
    main()
