from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import List, Optional, cast

from dotenv import load_dotenv

from src.config import Settings
from src.llm import GeminiAnswerGenerator
from src.rag_pipeline import RAGPipeline
from src.types import SourceContext
from src.vector_store import ChromaVectorStore


class _NoopEmbeddingModel:
    def embed_documents(self, _texts: List[str]) -> List[List[float]]:
        raise RuntimeError("Embedding model not initialized for this command.")

    def embed_query(self, _query: str) -> List[float]:
        raise RuntimeError("Embedding model not initialized for this command.")


def _build_pipeline(settings: Settings, api_key: Optional[str], require_embedding: bool = True) -> RAGPipeline:
    """Create a pipeline instance for CLI commands."""
    if require_embedding:
        from src.embeddings import LocalEmbeddingModel

        embedding_model = LocalEmbeddingModel(settings.embedding_model_name)
    else:
        embedding_model = _NoopEmbeddingModel()
    vector_store = ChromaVectorStore(settings.chroma_dir, settings.collection_name)

    answer_generator = None
    if api_key:
        answer_generator = GeminiAnswerGenerator(
            api_key=api_key,
            model_name=settings.gemini_model_name,
            max_retries=settings.gemini_max_retries,
            retry_wait_seconds=settings.gemini_retry_wait_seconds,
            temperature=settings.gemini_temperature,
        )

    return RAGPipeline(
        settings=settings,
        embedding_model=embedding_model,
        vector_store=vector_store,
        answer_generator=answer_generator,
    )


def _warn_if_fallback_backend(pipeline: RAGPipeline) -> None:
    if pipeline.vector_store.backend_name == "fallback":
        print(
            "Warning: Chroma backend unavailable in this environment. "
            "Using local fallback store."
        )
        if pipeline.vector_store.backend_error:
            print(f"Backend error: {pipeline.vector_store.backend_error}")


def _print_sources(contexts: List[SourceContext], max_chars: int = 240) -> None:
    """Print retrieved source snippets for transparency."""
    if not contexts:
        print("No source snippets retrieved.")
        return

    print("\nSources:")
    for idx, item in enumerate(contexts, start=1):
        source = item.get("source", "unknown")
        chunk_index = item.get("chunk_index", -1)
        distance = item.get("distance")
        snippet = str(item.get("text", "")).strip().replace("\n", " ")
        if len(snippet) > max_chars:
            snippet = snippet[:max_chars] + "..."

        if isinstance(distance, (int, float)):
            print(f"[{idx}] {source} | chunk={chunk_index} | distance={float(distance):.4f}")
        else:
            print(f"[{idx}] {source} | chunk={chunk_index}")
        print(f"    {snippet}")


def _resolve_input_files(paths: List[str]) -> List[Path]:
    """Validate and normalize input file paths."""
    resolved: List[Path] = []
    for raw in paths:
        path = Path(raw).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if path.suffix.lower() not in {".pdf", ".txt"}:
            raise ValueError(f"Unsupported file type: {path.suffix} ({path})")
        resolved.append(path)
    return resolved


def command_ingest(args: argparse.Namespace, settings: Settings) -> int:
    """Ingest one or more documents into local vector storage."""
    try:
        files = _resolve_input_files(args.files)
    except Exception as exc:
        print(f"Input validation error: {exc}", file=sys.stderr)
        return 2

    pipeline = _build_pipeline(settings, api_key=None, require_embedding=True)
    _warn_if_fallback_backend(pipeline)
    total_chunks = 0
    failed: List[str] = []

    for file_path in files:
        try:
            chunk_count = pipeline.ingest_file(file_path, source_name=file_path.name)
            total_chunks += chunk_count
            print(f"Ingested {file_path.name}: {chunk_count} chunks")
        except Exception as exc:
            failed.append(f"{file_path.name}: {exc}")

    print(f"\nTotal chunks stored/updated: {total_chunks}")
    if failed:
        print("Ingestion failures:", file=sys.stderr)
        for item in failed:
            print(f"- {item}", file=sys.stderr)
        return 1
    return 0


def command_ask(args: argparse.Namespace, settings: Settings) -> int:
    """Answer a single question against already-ingested documents."""
    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    pipeline = _build_pipeline(settings, api_key=api_key if api_key else None)
    _warn_if_fallback_backend(pipeline)

    if pipeline.vector_store.count() == 0:
        print("Vector store is empty. Run ingest first.", file=sys.stderr)
        return 1
    if not api_key:
        print("Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    answer, contexts = pipeline.answer_question(args.question, top_k=args.top_k)
    print("\nAnswer:")
    print(answer)
    if args.show_sources:
        _print_sources(contexts, max_chars=args.source_chars)
    return 0


def command_chat(args: argparse.Namespace, settings: Settings) -> int:
    """Start an interactive Q&A session in the terminal."""
    api_key = args.api_key or os.getenv("GEMINI_API_KEY", "")
    pipeline = _build_pipeline(settings, api_key=api_key if api_key else None)
    _warn_if_fallback_backend(pipeline)

    if pipeline.vector_store.count() == 0:
        print("Vector store is empty. Run ingest first.", file=sys.stderr)
        return 1
    if not api_key:
        print("Missing Gemini API key. Set GEMINI_API_KEY or pass --api-key.", file=sys.stderr)
        return 2

    print("Interactive chat started. Type 'exit' or 'quit' to stop.")
    while True:
        try:
            question = input("\nYou> ").strip()
        except EOFError:
            print("\nExiting chat.")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Exiting chat.")
            break

        answer, contexts = pipeline.answer_question(question, top_k=args.top_k)
        print("\nAssistant>")
        print(answer)
        if args.show_sources:
            _print_sources(contexts, max_chars=args.source_chars)
    return 0


def command_stats(_: argparse.Namespace, settings: Settings) -> int:
    """Show stored chunk count."""
    pipeline = _build_pipeline(settings, api_key=None, require_embedding=False)
    _warn_if_fallback_backend(pipeline)
    print(f"Stored chunks: {pipeline.vector_store.count()}")
    print(f"Chroma directory: {settings.chroma_dir}")
    print(f"Collection: {settings.collection_name}")
    return 0


def command_clear(_: argparse.Namespace, settings: Settings) -> int:
    """Clear all vectors from the configured collection."""
    pipeline = _build_pipeline(settings, api_key=None, require_embedding=False)
    _warn_if_fallback_backend(pipeline)
    pipeline.vector_store.clear()
    print("Vector store cleared.")
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    """Build CLI argument parser."""
    parser = argparse.ArgumentParser(description="Mini Document Q&A Agent CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Ingest PDF/TXT files into local vector store")
    ingest_parser.add_argument("files", nargs="+", help="One or more .pdf/.txt files")

    ask_parser = subparsers.add_parser("ask", help="Ask one question and print grounded answer")
    ask_parser.add_argument("question", help="Question text")
    ask_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    ask_parser.add_argument("--api-key", default="", help="Gemini API key (overrides GEMINI_API_KEY)")
    ask_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved context snippets",
    )
    ask_parser.add_argument(
        "--source-chars",
        type=int,
        default=240,
        help="Max characters shown per source snippet",
    )

    chat_parser = subparsers.add_parser("chat", help="Start interactive terminal Q&A")
    chat_parser.add_argument("--top-k", type=int, default=None, help="Number of chunks to retrieve")
    chat_parser.add_argument("--api-key", default="", help="Gemini API key (overrides GEMINI_API_KEY)")
    chat_parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved context snippets",
    )
    chat_parser.add_argument(
        "--source-chars",
        type=int,
        default=240,
        help="Max characters shown per source snippet",
    )

    subparsers.add_parser("stats", help="Show vector store statistics")
    subparsers.add_parser("clear", help="Clear all vectors from current collection")
    return parser


def main() -> int:
    """CLI entrypoint."""
    load_dotenv()
    settings = Settings()
    parser = build_arg_parser()
    args = parser.parse_args()

    top_k = cast(Optional[int], getattr(args, "top_k", None))
    if top_k is None:
        setattr(args, "top_k", settings.top_k)

    if args.command == "ingest":
        return command_ingest(args, settings)
    if args.command == "ask":
        return command_ask(args, settings)
    if args.command == "chat":
        return command_chat(args, settings)
    if args.command == "stats":
        return command_stats(args, settings)
    if args.command == "clear":
        return command_clear(args, settings)

    parser.print_help()
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
