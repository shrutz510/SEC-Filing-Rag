"""
LLM factory + structured extraction of entities/relations from filing chunks.

The LLM factory is the single switch between free (Ollama) and paid
(OpenAI/Anthropic) backends. Everything downstream — extractor, retriever,
synthesizer — uses `get_chat_model()` and is provider-agnostic.
"""

from __future__ import annotations

import logging
from typing import Iterable

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from tenacity import retry, stop_after_attempt, wait_exponential

from sec_filing_rag.config import settings
from sec_filing_rag.schema import FilingExtraction

log = logging.getLogger(__name__)


# ─── LLM factory ───────────────────────────────────────────────────────────


def get_chat_model(*, structured_output: type | None = None) -> BaseChatModel:
    """Return a chat model wired for the configured provider.

    If `structured_output` is passed, the returned runnable will parse output
    into that Pydantic model (LangChain calls this `with_structured_output`).
    """
    match settings.llm_provider:
        case "ollama":
            from langchain_ollama import ChatOllama

            model = ChatOllama(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                # Ollama needs `format="json"` for reliable structured output
                # on smaller models; with_structured_output handles this when
                # used with method="json_schema".
            )
        case "openai":
            from langchain_openai import ChatOpenAI

            model = ChatOpenAI(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=settings.openai_api_key,
            )
        case "anthropic":
            from langchain_anthropic import ChatAnthropic

            model = ChatAnthropic(
                model=settings.llm_model,
                temperature=settings.llm_temperature,
                api_key=settings.anthropic_api_key,
            )
        case other:                                              # pragma: no cover
            raise ValueError(f"Unknown LLM_PROVIDER: {other!r}")

    if structured_output is not None:
        # `method="json_schema"` works for both Ollama (recent versions) and
        # the hosted providers; gives stricter conformance than function-calling
        # on smaller local models.
        return model.with_structured_output(structured_output, method="json_schema")
    return model


def get_embeddings() -> Embeddings:
    """Embeddings model. Free default: Ollama nomic-embed-text."""
    match settings.embedding_provider:
        case "ollama":
            from langchain_ollama import OllamaEmbeddings
            return OllamaEmbeddings(model=settings.embedding_model)
        case "openai":
            from langchain_openai import OpenAIEmbeddings
            return OpenAIEmbeddings(model=settings.embedding_model, api_key=settings.openai_api_key)
        case other:                                              # pragma: no cover
            raise ValueError(f"Unknown EMBEDDING_PROVIDER: {other!r}")


# ─── Extraction prompt ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """\
You extract structured business entities and relations from SEC filings.

Rules:
- Only extract entities and relations explicitly supported by the text.
- Skip boilerplate (forward-looking-statement disclaimers, safe-harbor language).
- When the filer refers to itself ('the Company', 'we', 'our'), resolve to the
  filer's name given in the user prompt context.
- For each relation, include a short verbatim quote as evidence (≤200 chars).
- Prefer fewer high-confidence extractions over exhaustive coverage.
- If a section has nothing structured to extract, return empty lists.
"""

_USER_TEMPLATE = """\
FILER: {ticker}
FILING: {filing_type} for fiscal year {fiscal_year}
SECTION: Item {item_no} — {section_name}

----- BEGIN TEXT -----
{text}
----- END TEXT -----

Extract entities and relations from the section above.
"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=1, max=10), reraise=True)
def extract_from_chunk(chunk: Document) -> FilingExtraction:
    """Run structured extraction on a single chunk. Retries on transient errors."""
    model = get_chat_model(structured_output=FilingExtraction)
    user = _USER_TEMPLATE.format(
        ticker=chunk.metadata.get("ticker", "UNKNOWN"),
        filing_type=chunk.metadata.get("filing_type", ""),
        fiscal_year=chunk.metadata.get("fiscal_year", ""),
        item_no=chunk.metadata.get("item_no", ""),
        section_name=chunk.metadata.get("section_name", ""),
        text=chunk.page_content,
    )
    return model.invoke([SystemMessage(_SYSTEM_PROMPT), HumanMessage(user)])  # type: ignore[return-value]


def extract_all(chunks: Iterable[Document]) -> list[tuple[Document, FilingExtraction]]:
    """Sequential extraction. Swap in asyncio.gather for parallelism once
    you've confirmed your provider's rate limits."""
    out: list[tuple[Document, FilingExtraction]] = []
    for c in chunks:
        try:
            ex = extract_from_chunk(c)
            out.append((c, ex))
        except Exception as e:                                    # noqa: BLE001
            log.warning("Extraction failed for %s: %s", c.metadata.get("accession_no"), e)
    return out
