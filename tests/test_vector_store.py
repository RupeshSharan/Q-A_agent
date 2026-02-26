from __future__ import annotations

import uuid
from pathlib import Path

from src.ingestion import ChunkRecord
from src.vector_store import ChromaVectorStore


def test_vector_store_similarity_search_orders_by_cosine(tmp_path: Path) -> None:
    collection_name = f"test_{uuid.uuid4().hex}"
    store = ChromaVectorStore(tmp_path / "chroma", collection_name=collection_name)

    chunks = [
        ChunkRecord(text="alpha content", source="a.txt", chunk_index=0),
        ChunkRecord(text="beta content", source="b.txt", chunk_index=0),
        ChunkRecord(text="gamma content", source="c.txt", chunk_index=0),
    ]
    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
        [-1.0, 0.0],
    ]
    store.upsert_chunks(chunks, embeddings)

    results = store.similarity_search([1.0, 0.0], k=5)

    assert len(results) == 3
    assert results[0]["source"] == "a.txt"
    assert results[0]["chunk_index"] == 0
    assert results[0]["text"] == "alpha content"


def test_vector_store_similarity_search_handles_non_positive_k(tmp_path: Path) -> None:
    collection_name = f"test_{uuid.uuid4().hex}"
    store = ChromaVectorStore(tmp_path / "chroma", collection_name=collection_name)
    assert store.similarity_search([1.0, 0.0], k=0) == []
