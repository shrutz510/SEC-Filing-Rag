"""Centralized settings, loaded from environment / .env.

The free path requires zero secrets. Paid providers are opt-in via env vars.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- LLM ---
    llm_provider: Literal["ollama", "openai", "anthropic"] = "ollama"
    llm_model: str = "qwen2.5:14b"
    llm_temperature: float = 0.0
    openai_api_key: str | None = None
    anthropic_api_key: str | None = None

    # --- Embeddings ---
    embedding_provider: Literal["ollama", "openai"] = "ollama"
    embedding_model: str = "nomic-embed-text"

    # --- Vector store ---
    chroma_dir: Path = Path("./.chroma")
    chroma_collection: str = "filings"

    # --- Graph store ---
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "sec-filing-rag-dev"

    # --- Ingestion ---
    edgar_user_agent: str = "sec-filing-rag example@example.com"
    edgar_cache_dir: Path = Path("./.edgar")

    @property
    def is_local_only(self) -> bool:
        """True if no paid API keys are required to run."""
        return self.llm_provider == "ollama" and self.embedding_provider == "ollama"


settings = Settings()  # singleton; import as `from sec_filing_rag.config import settings`
