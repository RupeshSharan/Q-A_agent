from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from src.config import Settings
from src.rag_pipeline import RAGPipeline


class FakeEmbeddingModel:
    def __init__(self) -> None:
        self.document_calls: List[List[str]] = []
        self.query_calls: List[str] = []

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        self.document_calls.append(texts)
        return [[1.0, 0.0] for _ in texts]

    def embed_query(self, query: str) -> List[float]:
        self.query_calls.append(query)
        return [1.0, 0.0]


class FakeVectorStore:
    def __init__(self) -> None:
        self.records: List[Dict[str, object]] = []

    def upsert_chunks(self, chunks, embeddings) -> int:
        for chunk, _embedding in zip(chunks, embeddings):
            self.records.append(
                {
                    "text": chunk.text,
                    "source": chunk.source,
                    "chunk_index": chunk.chunk_index,
                }
            )
        return len(chunks)

    def similarity_search(self, _query_embedding: List[float], k: int = 5) -> List[Dict[str, object]]:
        return self.records[:k]

    def count(self) -> int:
        return len(self.records)


class FakeAnswerGenerator:
    def __init__(self) -> None:
        self.calls: List[Dict[str, object]] = []

    def generate_answer(self, question: str, contexts: List[Dict[str, object]]) -> str:
        self.calls.append({"question": question, "contexts": contexts})
        return "mocked grounded answer"


class LowSimilarityVectorStore(FakeVectorStore):
    def similarity_search(self, _query_embedding: List[float], k: int = 5) -> List[Dict[str, object]]:
        del k
        return [
            {
                "text": "Unrelated content",
                "source": "unrelated.txt",
                "chunk_index": 0,
                "distance": 1.25,  # similarity = -0.25
            }
        ]


def test_pipeline_ingest_retrieve_answer(tmp_path: Path) -> None:
    file_path = tmp_path / "doc.txt"
    file_path.write_text(
        "First point about topic. Second point about details. Third point about results. "
        "Fourth point about conclusion. Fifth point about summary.",
        encoding="utf-8",
    )

    settings = Settings(
        chunk_size=5,
        chunk_overlap=2,
        top_k=2,
        chroma_dir=tmp_path / "chroma",
        collection_name="unused",
    )
    embedding_model = FakeEmbeddingModel()
    vector_store = FakeVectorStore()
    answer_generator = FakeAnswerGenerator()
    pipeline = RAGPipeline(settings, embedding_model, vector_store, answer_generator)

    inserted = pipeline.ingest_file(file_path, source_name="doc.txt")
    assert inserted >= 1
    assert len(embedding_model.document_calls) == 1
    assert len(vector_store.records) >= 1

    answer, contexts = pipeline.answer_question("What is in the document?", top_k=2)
    assert answer == "mocked grounded answer"
    assert len(contexts) <= 2
    assert len(answer_generator.calls) == 1
    assert "What is in the document?" in embedding_model.query_calls


def test_pipeline_returns_no_context_message_when_empty(tmp_path: Path) -> None:
    settings = Settings(
        chunk_size=5,
        chunk_overlap=2,
        top_k=2,
        chroma_dir=tmp_path / "chroma",
        collection_name="unused",
    )
    pipeline = RAGPipeline(settings, FakeEmbeddingModel(), FakeVectorStore(), FakeAnswerGenerator())

    answer, contexts = pipeline.answer_question("Anything here?")
    assert answer == "No relevant context found in the ingested documents."
    assert contexts == []


def test_pipeline_filters_low_similarity_contexts(tmp_path: Path) -> None:
    settings = Settings(
        chunk_size=5,
        chunk_overlap=2,
        top_k=2,
        chroma_dir=tmp_path / "chroma",
        collection_name="unused",
        retrieval_min_similarity=0.1,
    )
    pipeline = RAGPipeline(settings, FakeEmbeddingModel(), LowSimilarityVectorStore(), FakeAnswerGenerator())

    answer, contexts = pipeline.answer_question("What is the vacation policy?")
    assert answer == "No relevant context found in the ingested documents."
    assert contexts == []
