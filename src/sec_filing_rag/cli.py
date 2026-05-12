"""
Typer CLI: `sec-filing-rag ingest | build | query | clear`.

This is the surface a developer touches day-to-day; keep it dead simple.
"""

from __future__ import annotations

import logging

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from sec_filing_rag.agent import answer
from sec_filing_rag.extraction import extract_all
from sec_filing_rag.ingestion import chunk_filing, download_filings, parse_filing
from sec_filing_rag.storage import (
    get_graph,
    init_graph_schema,
    upsert_chunks,
    upsert_extraction,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)s  %(message)s")
console = Console()
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def ingest(
    tickers: list[str] = typer.Option(..., "--tickers", "-t", help="Tickers to pull."),
    years: int = typer.Option(3, "--years", "-y", help="Most recent N years of filings."),
    filing_type: str = typer.Option("10-K", "--filing-type"),
) -> None:
    """Download filings from EDGAR into the local cache."""
    paths = download_filings(tickers, filing_type=filing_type, years=years)
    console.print(f"[green]Downloaded {len(paths)} filings.[/]")


@app.command()
def build(
    tickers: list[str] = typer.Option(..., "--tickers", "-t"),
    filing_type: str = typer.Option("10-K", "--filing-type"),
) -> None:
    """Parse cached filings, extract entities, populate Chroma + Neo4j."""
    init_graph_schema()

    # 1. Parse + chunk
    from sec_filing_rag.config import settings
    all_chunks = []
    for t in tickers:
        root = settings.edgar_cache_dir / "sec-edgar-filings" / t / filing_type
        for sub in sorted(root.rglob("full-submission.txt")):
            doc = parse_filing(sub)
            all_chunks.extend(chunk_filing(doc))

    console.print(f"Parsed {len(all_chunks)} chunks across {len(tickers)} tickers.")
    if not all_chunks:
        raise typer.Exit("No chunks parsed — did you run `ingest` first?")

    # 2. Vector store
    upsert_chunks(all_chunks)

    # 3. Extract + graph
    pairs = extract_all(all_chunks)
    for chunk, ex in pairs:
        upsert_extraction(
            ex,
            filing_id=chunk.metadata.get("accession_no", ""),
            fiscal_year=int(chunk.metadata.get("fiscal_year", 0)),
        )
    console.print(f"[green]Built graph + vector store from {len(pairs)} extractions.[/]")


@app.command()
def query(question: str = typer.Argument(..., help="Your natural-language question.")) -> None:
    """Run a question end-to-end through the LangGraph workflow."""
    result = answer(question)

    table = Table(show_header=False, box=None)
    table.add_row("[bold]Route[/]", result.get("query_type", "?"))
    if result.get("retrieval") and result["retrieval"].cypher:
        table.add_row("[bold]Cypher[/]", result["retrieval"].cypher)
    console.print(table)
    console.print(Panel(result["answer"], title="Answer", border_style="green"))


@app.command()
def clear() -> None:
    """Wipe Chroma and Neo4j. Useful between experiments."""
    from sec_filing_rag.config import settings
    import shutil

    if settings.chroma_dir.exists():
        shutil.rmtree(settings.chroma_dir)
        console.print(f"Removed {settings.chroma_dir}")
    get_graph().query("MATCH (n) DETACH DELETE n")
    console.print("[green]Cleared graph.[/]")


if __name__ == "__main__":
    app()
