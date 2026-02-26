"""Local persistent vector store with Chroma backend and safe fallback."""
from __future__ import annotations

import hashlib
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import numpy as np

from .ingestion import ChunkRecord
from .types import SourceContext


@dataclass
class FallbackItem:
    """Item stored in the local pickle-based fallback store."""

    embedding: List[float]
    document: str
    metadata: Dict[str, object]


class ChromaVectorStore:
    """Stores and retrieves chunk embeddings with metadata."""

    def __init__(self, persist_dir: Path, collection_name: str) -> None:
        persist_dir.mkdir(parents=True, exist_ok=True)
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.backend_name = "chroma"
        self.backend_error: Optional[str] = None

        # Chroma types are not stable across versions; keep them as Any.
        self.collection: Any = None
        self.client: Any = None

        self._fallback_path = self.persist_dir / f"{self.collection_name}_fallback.pkl"
        self._fallback_items: Dict[str, FallbackItem] = {}

        try:
            # Chroma can fail on Python 3.14 because of upstream pydantic.v1 internals.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="Core Pydantic V1 functionality isn't compatible with Python 3.14 or greater.",
                    category=UserWarning,
                )
                import chromadb  # type: ignore

            self.client = chromadb.PersistentClient(path=str(persist_dir))
            self.collection = self.client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
        except Exception as exc:
            self.backend_name = "fallback"
            self.backend_error = str(exc)
            self._load_fallback()

    @staticmethod
    def _as_int(value: object, default: int = -1) -> int:
        try:
            return int(value)  # type: ignore[arg-type]
        except Exception:
            return default

    @staticmethod
    def _as_float(value: object) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)  # type: ignore[arg-type]
        except Exception:
            return None

    def _chunk_id(self, chunk: ChunkRecord) -> str:
        chunk_fingerprint = f"{chunk.source}:{chunk.chunk_index}:{chunk.text}"
        return hashlib.sha1(chunk_fingerprint.encode("utf-8")).hexdigest()

    def _load_fallback(self) -> None:
        if not self._fallback_path.exists():
            self._fallback_items = {}
            return
        try:
            with self._fallback_path.open("rb") as fh:
                raw_data = pickle.load(fh)
            if not isinstance(raw_data, dict):
                self._fallback_items = {}
                return

            loaded: Dict[str, FallbackItem] = {}
            for item_id, item_value in raw_data.items():
                if not isinstance(item_id, str) or not isinstance(item_value, dict):
                    continue
                embedding = item_value.get("embedding")
                document = item_value.get("document")
                metadata = item_value.get("metadata")
                if not isinstance(embedding, list) or not isinstance(document, str) or not isinstance(metadata, dict):
                    continue
                loaded[item_id] = FallbackItem(
                    embedding=[float(x) for x in embedding],
                    document=document,
                    metadata=cast(Dict[str, object], metadata),
                )
            self._fallback_items = loaded
        except Exception:
            self._fallback_items = {}

    def _save_fallback(self) -> None:
        serializable: Dict[str, Dict[str, object]] = {}
        for item_id, item in self._fallback_items.items():
            serializable[item_id] = {
                "embedding": item.embedding,
                "document": item.document,
                "metadata": item.metadata,
            }
        with self._fallback_path.open("wb") as fh:
            pickle.dump(serializable, fh)

    def _fallback_similarity_search(self, query_embedding: List[float], k: int) -> List[SourceContext]:
        if k <= 0 or not self._fallback_items:
            return []

        query = np.array(query_embedding, dtype=np.float32)
        if query.ndim != 1:
            return []

        ids = list(self._fallback_items.keys())
        vectors = np.array(
            [self._fallback_items[item_id].embedding for item_id in ids],
            dtype=np.float32,
        )
        if vectors.ndim != 2:
            return []
        if vectors.shape[1] != query.shape[0]:
            raise RuntimeError(
                "Stored embedding dimension does not match query embedding dimension. "
                "Clear and re-ingest documents with the current embedding model."
            )

        vector_norms = np.linalg.norm(vectors, axis=1)
        query_norm = float(np.linalg.norm(query))
        denom = vector_norms * (query_norm if query_norm != 0 else 1.0)
        denom = np.where(denom == 0, 1e-12, denom)
        scores = np.dot(vectors, query) / denom

        top_indices = np.argsort(-scores)[: min(k, len(ids))]
        matches: List[SourceContext] = []
        for idx in top_indices:
            item = self._fallback_items[ids[int(idx)]]
            score = float(scores[int(idx)])
            matches.append(
                SourceContext(
                    text=item.document,
                    source=str(item.metadata.get("source", "unknown")),
                    chunk_index=self._as_int(item.metadata.get("chunk_index", -1), -1),
                    distance=1.0 - score,
                )
            )
        return matches

    def upsert_chunks(self, chunks: List[ChunkRecord], embeddings: List[List[float]]) -> int:
        """Insert or update chunk vectors and metadata."""
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must be the same length")
        if not chunks:
            return 0

        ids: List[str] = []
        documents: List[str] = []
        metadatas: List[Dict[str, object]] = []

        for chunk in chunks:
            ids.append(self._chunk_id(chunk))
            documents.append(chunk.text)
            metadatas.append(
                {
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                }
            )

        if self.backend_name == "chroma" and self.collection is not None:
            self.collection.upsert(
                ids=ids,
                embeddings=embeddings,
                documents=documents,
                metadatas=metadatas,
            )
            return len(ids)

        for item_id, embedding, document, metadata in zip(ids, embeddings, documents, metadatas):
            self._fallback_items[item_id] = FallbackItem(
                embedding=embedding,
                document=document,
                metadata=metadata,
            )
        self._save_fallback()
        return len(ids)

    def similarity_search(self, query_embedding: List[float], k: int = 5) -> List[SourceContext]:
        """Return top-k nearest chunks using cosine distance."""
        if self.backend_name == "chroma" and self.collection is not None:
            if k <= 0:
                return []

            total_items = self.collection.count()
            if total_items == 0:
                return []

            raw_results: Dict[str, Any] = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=min(k, total_items),
                include=["documents", "metadatas", "distances"],
            )

            raw_documents = raw_results.get("documents", [[]])
            raw_metadatas = raw_results.get("metadatas", [[]])
            raw_distances = raw_results.get("distances", [[]])

            documents = raw_documents[0] if isinstance(raw_documents, list) and raw_documents else []
            metadatas = raw_metadatas[0] if isinstance(raw_metadatas, list) and raw_metadatas else []
            distances = raw_distances[0] if isinstance(raw_distances, list) and raw_distances else []

            matches: List[SourceContext] = []
            for document, metadata, distance in zip(documents, metadatas, distances):
                metadata_dict = metadata if isinstance(metadata, dict) else {}
                matches.append(
                    SourceContext(
                        text=str(document or ""),
                        source=str(metadata_dict.get("source", "unknown")),
                        chunk_index=self._as_int(metadata_dict.get("chunk_index", -1), -1),
                        distance=self._as_float(distance),
                    )
                )
            return matches

        return self._fallback_similarity_search(query_embedding, k)

    def count(self) -> int:
        """Return number of stored chunks."""
        if self.backend_name == "chroma" and self.collection is not None:
            return int(self.collection.count())
        return len(self._fallback_items)

    def list_sources(self, limit: int = 50) -> List[str]:
        """List unique source filenames currently indexed."""
        if limit <= 0:
            return []

        sources: List[str] = []
        if self.backend_name == "chroma" and self.collection is not None:
            all_items: Dict[str, Any] = self.collection.get(include=["metadatas"])
            raw_metadatas = all_items.get("metadatas", [])

            # Chroma can return list[dict] or list[list[dict]] depending on query/get shape.
            flattened: List[object] = []
            if isinstance(raw_metadatas, list):
                for item in raw_metadatas:
                    if isinstance(item, list):
                        flattened.extend(item)
                    else:
                        flattened.append(item)

            for metadata in flattened:
                if isinstance(metadata, dict):
                    source = metadata.get("source")
                    if isinstance(source, str) and source.strip():
                        sources.append(source.strip())
        else:
            for item in self._fallback_items.values():
                source = item.metadata.get("source")
                if isinstance(source, str) and source.strip():
                    sources.append(source.strip())

        unique_sorted = sorted(set(sources))
        return unique_sorted[:limit]

    def clear(self) -> None:
        """Delete all vectors from this collection."""
        if self.backend_name == "chroma" and self.collection is not None:
            all_items: Dict[str, Any] = self.collection.get()
            ids = all_items.get("ids", [])
            if isinstance(ids, list) and ids:
                self.collection.delete(ids=ids)
            return

        self._fallback_items = {}
        if self._fallback_path.exists():
            self._fallback_path.unlink()
