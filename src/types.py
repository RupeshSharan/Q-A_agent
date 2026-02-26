"""Shared type declarations for application state and retrieval results."""
from __future__ import annotations

from typing import List, Literal, Optional, TypedDict


class SourceContext(TypedDict):
    """Single retrieved chunk returned from the vector store."""

    text: str
    source: str
    chunk_index: int
    distance: Optional[float]


class ChatMessage(TypedDict, total=False):
    """Streamlit chat transcript item."""

    role: Literal["user", "assistant"]
    content: str
    sources: List[SourceContext]
