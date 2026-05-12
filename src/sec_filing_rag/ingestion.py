"""
EDGAR ingestion: download filings, split into Items, chunk for embedding.

Keep this module narrow. Each function takes plain types in and returns
plain types out, so it's trivial to unit-test without hitting the network.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sec_edgar_downloader import Downloader

from sec_filing_rag.config import settings

log = logging.getLogger(__name__)


# Sections we actually want from a 10-K. Skip the cover page, signatures,
# and the exhibits index. Item 1A and Item 7 are where the gold is.
TEN_K_SECTIONS = {
    "1": "Business",
    "1A": "Risk Factors",
    "1B": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "MD&A",
    "7A": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements",
    "9A": "Controls and Procedures",
}

# Per-section chunk sizing. Risk factors and MD&A are dense; cut smaller.
SECTION_CHUNK_SIZE = {
    "1A": 1200,
    "7": 1500,
    "_default": 2000,
}


@dataclass
class FilingDoc:
    """A parsed filing, ready to chunk."""

    cik: str
    ticker: str
    filing_type: str       # "10-K", "10-Q", "8-K"
    fiscal_year: int
    accession_no: str
    sections: dict[str, str] = field(default_factory=dict)   # item_no -> text


# ─── Download ──────────────────────────────────────────────────────────────


def download_filings(tickers: list[str], filing_type: str = "10-K", years: int = 3) -> list[Path]:
    """Pull filings via sec-edgar-downloader. Returns the paths of full
    submission .txt files for downstream parsing.

    SEC requires a User-Agent identifying the requester; configure via env.
    """
    name, email = _parse_user_agent(settings.edgar_user_agent)
    dl = Downloader(name, email, str(settings.edgar_cache_dir))

    paths: list[Path] = []
    for t in tickers:
        log.info("Downloading %s filings for %s", filing_type, t)
        dl.get(filing_type, t, limit=years)
        # sec_edgar_downloader stores under {edgar_cache_dir}/sec-edgar-filings/{TICKER}/{TYPE}/...
        root = settings.edgar_cache_dir / "sec-edgar-filings" / t / filing_type
        if root.exists():
            paths.extend(sorted(root.rglob("full-submission.txt")))
    return paths


def _parse_user_agent(ua: str) -> tuple[str, str]:
    """SEC wants 'Name email@host'; split for the downloader's signature."""
    parts = ua.rsplit(" ", 1)
    if len(parts) != 2 or "@" not in parts[1]:
        raise ValueError(
            f"EDGAR_USER_AGENT must look like 'Your Name your@email.com'; got: {ua!r}"
        )
    return parts[0], parts[1]


# ─── Parse ─────────────────────────────────────────────────────────────────

_ITEM_HEADING = re.compile(r"\bITEM\s+(\d+[A-Z]?)\.?\b", re.IGNORECASE)


def parse_filing(submission_path: Path) -> FilingDoc:
    """Extract sections from a 10-K full-submission.txt file.

    The SEC bundles many sub-files. We grab the primary HTML doc, strip
    tags, then split on 'Item N.' headings. This is heuristic but works
    for ~95% of large filers — good enough for a portfolio project.
    """
    raw = submission_path.read_text(encoding="utf-8", errors="ignore")

    # Pull the primary 10-K HTML doc out of the multi-doc submission envelope.
    html_match = re.search(r"<TYPE>10-K.*?<TEXT>(.*?)</TEXT>", raw, re.DOTALL | re.IGNORECASE)
    body_html = html_match.group(1) if html_match else raw

    text = BeautifulSoup(body_html, "lxml").get_text("\n")
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Split on Item N. headings.
    sections: dict[str, str] = {}
    positions = [(m.group(1).upper(), m.start()) for m in _ITEM_HEADING.finditer(text)]
    for (item, start), (_, end) in zip(positions, positions[1:] + [(None, len(text))]):
        if item in TEN_K_SECTIONS and item not in sections:
            sections[item] = text[start:end].strip()

    # Metadata — best effort from path / filename.
    parts = submission_path.parts
    ticker = parts[-4] if len(parts) >= 4 else "UNKNOWN"
    accession = submission_path.parent.name
    fiscal_year = _guess_fiscal_year(text)

    return FilingDoc(
        cik="",                  # populated later from EDGAR metadata if needed
        ticker=ticker,
        filing_type="10-K",
        fiscal_year=fiscal_year,
        accession_no=accession,
        sections=sections,
    )


def _guess_fiscal_year(text: str) -> int:
    """Pull the fiscal year from a 'For the fiscal year ended ...' line."""
    m = re.search(r"fiscal year ended\s+\w+\s+\d{1,2},?\s+(\d{4})", text, re.IGNORECASE)
    return int(m.group(1)) if m else 0


# ─── Chunk ─────────────────────────────────────────────────────────────────


def chunk_filing(doc: FilingDoc) -> list[Document]:
    """Convert a parsed filing into LangChain Documents with rich metadata.

    Metadata is used downstream for filtered retrieval (e.g. 'risk factors
    only, AAPL, FY2024').
    """
    out: list[Document] = []
    for item_no, body in doc.sections.items():
        chunk_size = SECTION_CHUNK_SIZE.get(item_no, SECTION_CHUNK_SIZE["_default"])
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.1),
            separators=["\n\n", "\n", ". ", " "],
        )
        for i, piece in enumerate(splitter.split_text(body)):
            out.append(
                Document(
                    page_content=piece,
                    metadata={
                        "ticker": doc.ticker,
                        "cik": doc.cik,
                        "filing_type": doc.filing_type,
                        "fiscal_year": doc.fiscal_year,
                        "accession_no": doc.accession_no,
                        "item_no": item_no,
                        "section_name": TEN_K_SECTIONS.get(item_no, ""),
                        "chunk_idx": i,
                    },
                )
            )
    return out
