"""End-to-end RAG pipeline orchestration."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from .config import Settings
from .embeddings import LocalEmbeddingModel
from .ingestion import ingest_document
from .llm import GeminiAnswerGenerator
from .retriever import retrieve_with_expansion
from .types import SourceContext
from .vector_store import ChromaVectorStore


class RAGPipeline:
    """Coordinates ingestion, retrieval, and answer generation."""

    def __init__(
        self,
        settings: Settings,
        embedding_model: LocalEmbeddingModel,
        vector_store: ChromaVectorStore,
        answer_generator: Optional[GeminiAnswerGenerator] = None,
    ) -> None:
        self.settings = settings
        self.embedding_model = embedding_model
        self.vector_store = vector_store
        self.answer_generator = answer_generator

    def ingest_file(self, file_path: Path, source_name: Optional[str] = None) -> int:
        """Parse file, chunk it, embed chunks, and persist vectors."""
        chunks = ingest_document(
            file_path=file_path,
            chunk_size=self.settings.chunk_size,
            chunk_overlap=self.settings.chunk_overlap,
            source_name=source_name,
        )
        if not chunks:
            return 0

        embeddings = self.embedding_model.embed_documents([chunk.text for chunk in chunks])
        return self.vector_store.upsert_chunks(chunks, embeddings)

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[SourceContext]:
        """Retrieve most relevant chunks for a query using expansion."""
        k = top_k or self.settings.top_k
        # Use expanded retrieval for better recall.
        contexts = retrieve_with_expansion(query, self.embedding_model, self.vector_store, k=k)
        filtered_contexts: List[SourceContext] = []
        for item in contexts:
            distance = item.get("distance")
            if isinstance(distance, (int, float)):
                similarity = 1.0 - float(distance)
                if similarity < self.settings.retrieval_min_similarity:
                    continue
            filtered_contexts.append(item)
        return filtered_contexts

    def answer_question(self, question: str, top_k: Optional[int] = None) -> Tuple[str, List[SourceContext]]:
        """Retrieve context and generate a grounded answer.

        If an answer generator is available, the question is first corrected
        for spelling/grammar errors before retrieval.
        """
        # Correct spelling/grammar in the query before retrieval.
        corrected_question = question
        if self.answer_generator is not None:
            try:
                corrected_question = self.answer_generator.correct_query(question)
            except Exception:
                corrected_question = question

        contexts = self.retrieve(corrected_question, top_k=top_k)
        if not contexts:
            return "No relevant context found in the ingested documents.", []

        if self.answer_generator is None:
            return "Answer generator is not configured.", contexts

        answer = self.answer_generator.generate_answer(corrected_question, contexts)
        return answer, contexts
