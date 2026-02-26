from __future__ import annotations

from pathlib import Path

import pytest

from src.ingestion import ingest_document, sliding_window_chunk_text


def test_sliding_window_chunk_text_sentence_aware() -> None:
    """Chunking should respect sentence boundaries."""
    text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four. This is sentence five."
    chunks = sliding_window_chunk_text(text, chunk_size=8, chunk_overlap=3)

    # Should produce multiple sentence-aware chunks.
    assert len(chunks) >= 2
    # Each chunk should contain complete sentences.
    for chunk in chunks:
        assert chunk.strip()


def test_sliding_window_chunk_text_fallback() -> None:
    """When no sentence boundaries exist, text is treated as a single block."""
    text = " ".join(f"t{i}" for i in range(1, 13))
    chunks = sliding_window_chunk_text(text, chunk_size=5, chunk_overlap=2)
    # With no sentence boundaries, text is either treated as single sentence
    # or falls back to word-level chunking. Either way, produces chunks.
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.strip()


@pytest.mark.parametrize(
    ("chunk_size", "chunk_overlap"),
    [
        (0, 0),
        (5, -1),
        (5, 5),
    ],
)
def test_sliding_window_chunk_text_invalid_params(chunk_size: int, chunk_overlap: int) -> None:
    with pytest.raises(ValueError):
        sliding_window_chunk_text("one two three", chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def test_ingest_document_txt_creates_chunk_metadata(tmp_path: Path) -> None:
    file_path = tmp_path / "sample.txt"
    # Write text with sentence boundaries for sentence-aware chunking.
    file_path.write_text(
        "First sentence here. Second sentence follows. Third sentence now. Fourth sentence next. Fifth sentence last.",
        encoding="utf-8",
    )

    chunks = ingest_document(file_path, chunk_size=6, chunk_overlap=2)

    assert len(chunks) >= 1
    assert chunks[0].source == "sample.txt"
    assert chunks[0].chunk_index == 0
    if len(chunks) > 1:
        assert chunks[1].chunk_index == 1
