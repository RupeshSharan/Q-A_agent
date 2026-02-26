"""Configuration values for the RAG application."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _env_str(name: str, default: str) -> str:
    """Read a string environment variable with a non-empty fallback."""
    value = os.getenv(name)
    if value is None:
        return default
    value = value.strip()
    return value or default


def _env_int(name: str, default: int) -> int:
    """Read an integer environment variable with safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    """Read a float environment variable with safe fallback."""
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


@dataclass
class Settings:
    """Runtime settings loaded from environment variables."""

    # Use default_factory so values are read when Settings() is instantiated,
    # not at module import time. This ensures load_dotenv() values are honored.
    embedding_model_name: str = field(default_factory=lambda: _env_str("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2"))
    chunk_size: int = field(default_factory=lambda: _env_int("CHUNK_SIZE", 400))
    chunk_overlap: int = field(default_factory=lambda: _env_int("CHUNK_OVERLAP", 100))
    top_k: int = field(default_factory=lambda: _env_int("TOP_K", 5))
    chroma_dir: Path = field(default_factory=lambda: Path(_env_str("CHROMA_DIR", "data/chroma")))
    collection_name: str = field(default_factory=lambda: _env_str("CHROMA_COLLECTION", "document_chunks"))
    gemini_model_name: str = field(default_factory=lambda: _env_str("GEMINI_MODEL", "gemini-2.5-flash"))
    gemini_max_retries: int = field(default_factory=lambda: _env_int("GEMINI_MAX_RETRIES", 3))
    gemini_retry_wait_seconds: int = field(default_factory=lambda: _env_int("GEMINI_RETRY_WAIT_SECONDS", 60))
    gemini_temperature: float = field(default_factory=lambda: _env_float("GEMINI_TEMPERATURE", 0.1))
    retrieval_min_similarity: float = field(default_factory=lambda: _env_float("RETRIEVAL_MIN_SIMILARITY", 0.25))
