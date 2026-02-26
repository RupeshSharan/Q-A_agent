"""Retrieval helpers."""
from __future__ import annotations

import re
from typing import List, Set

from .embeddings import LocalEmbeddingModel
from .types import SourceContext
from .vector_store import ChromaVectorStore


def _extract_keywords(query: str) -> Set[str]:
    """Extract meaningful keywords from a query for expansion."""
    tokens = re.findall(r"[a-zA-Z0-9]{3,}", query.lower())
    stopwords = {
        "what", "which", "when", "where", "who", "whom", "whose", "why",
        "how", "much", "many", "does", "did", "have", "with", "from",
        "about", "that", "this", "these", "those", "into", "your", "their",
        "there", "then", "the", "and", "for", "are", "but", "not", "you",
        "all", "can", "her", "was", "one", "our", "out", "has", "been",
        "will", "would", "could", "should", "may", "might", "shall",
    }
    return {t for t in tokens if t not in stopwords}


def _build_keyword_query(keywords: Set[str]) -> str:
    """Build a keyword-focused query string from extracted keywords."""
    if not keywords:
        return ""
    return " ".join(sorted(keywords))


def retrieve_top_k(
    query: str,
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
    k: int = 5,
) -> List[SourceContext]:
    """Embed a query and fetch the top-k similar chunks."""
    query_embedding = embedding_model.embed_query(query)
    return vector_store.similarity_search(query_embedding, k=k)


def retrieve_with_expansion(
    query: str,
    embedding_model: LocalEmbeddingModel,
    vector_store: ChromaVectorStore,
    k: int = 5,
) -> List[SourceContext]:
    """Retrieve chunks using the original query plus a keyword-expanded variant.

    This increases recall by running two retrieval passes:
    1. Original query (captures semantic meaning).
    2. Keyword-only query (captures exact term matches that embedding might miss).

    Results are deduplicated and merged, keeping the best distance score for
    each unique chunk.
    """
    # Primary retrieval with original query.
    primary_results = retrieve_top_k(query, embedding_model, vector_store, k=k)

    # Build keyword-focused query for expansion.
    keywords = _extract_keywords(query)
    keyword_query = _build_keyword_query(keywords)

    if not keyword_query or keyword_query.lower().strip() == query.lower().strip():
        return primary_results

    # Secondary retrieval with keyword query.
    expanded_results = retrieve_top_k(keyword_query, embedding_model, vector_store, k=k)

    # Merge and deduplicate: use (source, chunk_index) as identity key.
    seen: dict[tuple[str, int], SourceContext] = {}
    for item in primary_results:
        key = (item["source"], item["chunk_index"])
        seen[key] = item

    for item in expanded_results:
        key = (item["source"], item["chunk_index"])
        if key not in seen:
            seen[key] = item
        else:
            # Keep the one with lower distance (better similarity).
            existing_dist = seen[key].get("distance")
            new_dist = item.get("distance")
            if (
                isinstance(existing_dist, (int, float))
                and isinstance(new_dist, (int, float))
                and new_dist < existing_dist
            ):
                seen[key] = item

    # Sort by distance (ascending = most similar first) and return top-k.
    merged = sorted(
        seen.values(),
        key=lambda x: float(x.get("distance", 999)) if isinstance(x.get("distance"), (int, float)) else 999,
    )
    return merged[:k]
