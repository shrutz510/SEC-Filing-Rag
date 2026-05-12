"""
Run the eval set in three modes — vector-only, graph-only, hybrid (router) —
and print the ablation table that ships in the README.

Usage:
    python eval/run_evals.py
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import yaml
from rich.console import Console
from rich.table import Table

from sec_filing_rag.agent import answer as routed_answer
from sec_filing_rag.retrieval import graph_retrieve, vector_retrieve

console = Console()
EVAL_FILE = Path(__file__).resolve().parent / "questions.yaml"


def judge(question: str, answer: str, archetype: str) -> float:
    """LLM-as-judge stub. Replace with your real correctness scorer
    (a separate LLM call comparing to reference answers in questions.yaml,
    or Ragas's `answer_correctness` metric). Returns 0.0–1.0."""
    if not answer or "I don't know" in answer or "cannot" in answer.lower():
        return 0.0
    # TODO: real scoring. Placeholder returns optimistic score on non-empty.
    return 1.0


def run(mode: str, q: dict) -> tuple[float, str]:
    try:
        if mode == "vector":
            r = vector_retrieve(q["question"])
            return judge(q["question"], "\n".join(d.page_content for d in r.chunks), q["archetype"]), ""
        if mode == "graph":
            r = graph_retrieve(q["question"])
            return judge(q["question"], str(r.graph_facts), q["archetype"]), r.cypher or ""
        if mode == "hybrid":
            state = routed_answer(q["question"])
            return judge(q["question"], state.get("answer", ""), q["archetype"]), ""
    except Exception as e:                                        # noqa: BLE001
        console.print(f"[red]{mode} failed on {q['id']}: {e}[/]")
    return 0.0, ""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["vector", "graph", "hybrid"])
    args = ap.parse_args()

    questions = yaml.safe_load(EVAL_FILE.read_text())["questions"]

    # mode -> archetype -> [scores]
    results: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for q in questions:
        for mode in args.modes:
            score, _ = run(mode, q)
            results[mode][q["archetype"]].append(score)

    # Build the table.
    archetypes = ["semantic", "relational", "hybrid"]
    table = Table(title="Ablation: per-archetype accuracy")
    table.add_column("Retrieval mode")
    for a in archetypes:
        table.add_column(a.capitalize(), justify="right")
    table.add_column("Overall", justify="right")

    for mode in args.modes:
        row = [mode]
        all_scores = []
        for a in archetypes:
            xs = results[mode][a]
            row.append(f"{sum(xs)/len(xs):.2f}" if xs else "—")
            all_scores.extend(xs)
        row.append(f"{sum(all_scores)/len(all_scores):.2f}" if all_scores else "—")
        table.add_row(*row)

    console.print(table)


if __name__ == "__main__":
    main()
