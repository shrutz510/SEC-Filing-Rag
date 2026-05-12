"""
Export a Neo4j subgraph as a standalone interactive HTML file (pyvis).
The output is self-contained and works on GitHub Pages with no backend.

Usage:
    python scripts/export_graph_viz.py
"""

from __future__ import annotations

from pathlib import Path

from pyvis.network import Network

from sec_filing_rag.storage import get_graph

OUT = Path(__file__).resolve().parents[1] / "docs" / "graph.html"

NODE_COLORS = {
    "Company": "#4f46e5",
    "Person": "#0891b2",
    "Product": "#16a34a",
    "Risk": "#dc2626",
    "Location": "#ca8a04",
    "LegalMatter": "#9333ea",
}

# Pull a focused subgraph — companies + their immediate neighbours.
# Adjust the LIMIT once your graph grows.
QUERY = """
MATCH (c:Company)-[r]-(n)
RETURN c, r, n
LIMIT 250
"""


def main() -> None:
    graph = get_graph()
    rows = graph.query(QUERY)

    net = Network(height="700px", width="100%", bgcolor="#ffffff", directed=True)
    net.force_atlas_2based(gravity=-50)

    seen: set[str] = set()

    def add(node) -> str:
        label = next(iter(node.labels)) if hasattr(node, "labels") else "Node"
        key = f"{label}:{node.get('name') or node.get('summary') or node.get('description') or id(node)}"
        if key not in seen:
            net.add_node(
                key,
                label=node.get("name") or node.get("summary") or label,
                color=NODE_COLORS.get(label, "#64748b"),
                title=label,
            )
            seen.add(key)
        return key

    for row in rows:
        c_key = add(row["c"])
        n_key = add(row["n"])
        r = row["r"]
        rel_type = type(r).__name__ if hasattr(r, "__name__") else "REL"
        net.add_edge(c_key, n_key, title=rel_type, label=rel_type)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    net.write_html(str(OUT), notebook=False)
    print(f"Wrote {OUT}")


if __name__ == "__main__":
    main()
