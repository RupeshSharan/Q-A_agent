"""RAG runtime bootstrap using existing src pipeline modules."""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

from src.config import Settings
from src.embeddings import LocalEmbeddingModel
from src.llm import GeminiAnswerGenerator
from src.rag_pipeline import RAGPipeline
from src.vector_store import ChromaVectorStore

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_STORAGE_DIR = PROJECT_ROOT / "backend" / "storage" / "pdfs"
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "backend" / "data" / "chroma"

load_dotenv(PROJECT_ROOT / ".env")


@dataclass
class RagRuntime:
    settings: Settings
    embedding_model: LocalEmbeddingModel
    vector_store: ChromaVectorStore


_runtime_lock = threading.Lock()
_runtime: RagRuntime | None = None


def get_storage_dir() -> Path:
    raw_dir = os.getenv("DOCMIND_STORAGE_DIR", "").strip()
    if raw_dir:
        return Path(raw_dir).expanduser().resolve()
    return DEFAULT_STORAGE_DIR


def get_chroma_dir() -> Path:
    raw_dir = os.getenv("CHROMA_DIR", "").strip()
    if raw_dir:
        return Path(raw_dir).expanduser().resolve()
    return DEFAULT_CHROMA_DIR


def ensure_backend_dirs() -> None:
    get_storage_dir().mkdir(parents=True, exist_ok=True)
    get_chroma_dir().mkdir(parents=True, exist_ok=True)


def _build_runtime() -> RagRuntime:
    settings = Settings(chroma_dir=get_chroma_dir())
    embedding_model = LocalEmbeddingModel(settings.embedding_model_name)
    vector_store = ChromaVectorStore(settings.chroma_dir, settings.collection_name)
    return RagRuntime(settings=settings, embedding_model=embedding_model, vector_store=vector_store)


def get_runtime() -> RagRuntime:
    global _runtime
    if _runtime is not None:
        return _runtime
    with _runtime_lock:
        if _runtime is None:
            _runtime = _build_runtime()
    return _runtime


def has_gemini_key() -> bool:
    return bool(os.getenv("GEMINI_API_KEY", "").strip())


def build_pipeline(require_llm: bool) -> RAGPipeline:
    runtime = get_runtime()

    answer_generator = None
    if require_llm and has_gemini_key():
        answer_generator = GeminiAnswerGenerator(
            api_key=os.getenv("GEMINI_API_KEY", "").strip(),
            model_name=runtime.settings.gemini_model_name,
            max_retries=runtime.settings.gemini_max_retries,
            retry_wait_seconds=runtime.settings.gemini_retry_wait_seconds,
            temperature=runtime.settings.gemini_temperature,
        )

    return RAGPipeline(
        settings=runtime.settings,
        embedding_model=runtime.embedding_model,
        vector_store=runtime.vector_store,
        answer_generator=answer_generator,
    )

