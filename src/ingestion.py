"""Document loading and chunking helpers."""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pdfplumber


@dataclass
class ChunkRecord:
    """A single chunk with source metadata."""

    text: str
    source: str
    chunk_index: int


def extract_text_from_pdf(file_path: Path) -> str:
    """Extract plain text from a PDF file page-by-page.

    Uses pdfplumber first. If no text is found, falls back to PyMuPDF when available.
    """
    page_text_parts: List[str] = []
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                page_text_parts.append(page_text.strip())
    text = "\n".join(page_text_parts).strip()
    if text:
        return text

    # Fallback for PDFs where pdfplumber returns little/no text.
    try:
        import fitz  # type: ignore
    except Exception:
        return ""

    fallback_parts: List[str] = []
    with fitz.open(file_path) as doc:
        for page in doc:
            page_text = page.get_text("text") or ""
            if page_text.strip():
                fallback_parts.append(page_text.strip())
    return "\n".join(fallback_parts).strip()


def extract_text(file_path: Path) -> str:
    """Extract raw text from supported file types."""
    suffix = file_path.suffix.lower()
    if suffix == ".pdf":
        return extract_text_from_pdf(file_path)
    if suffix == ".txt":
        return file_path.read_text(encoding="utf-8", errors="ignore").strip()
    raise ValueError(f"Unsupported file type: {suffix}")


def _normalize_text(text: str) -> str:
    """Clean up text before chunking: fix whitespace, encoding artifacts."""
    # Collapse multiple newlines into double newline (paragraph break).
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Collapse multiple spaces/tabs into single space.
    text = re.sub(r"[ \t]+", " ", text)
    # Fix common encoding artifacts.
    text = text.replace("\u00a0", " ")  # non-breaking space
    text = text.replace("\u2019", "'").replace("\u2018", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    text = text.replace("\u2014", "—").replace("\u2013", "–")
    return text.strip()


def _split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using regex boundary detection.

    Handles abbreviations (Mr., Dr., etc.), decimal numbers, and URLs
    better than a naive period-split.
    """
    # Split on sentence-ending punctuation followed by space + uppercase,
    # or on double newlines (paragraph boundaries).
    sentence_pattern = r"(?<=[.!?])\s+(?=[A-Z\"\'\(\[])|(?:\n\n+)"
    raw_sentences = re.split(sentence_pattern, text)
    sentences: List[str] = []
    for s in raw_sentences:
        s = s.strip()
        if s:
            sentences.append(s)
    return sentences


def sliding_window_chunk_text(text: str, chunk_size: int = 400, chunk_overlap: int = 100) -> List[str]:
    """Split text into overlapping chunks using sentence-aware boundaries.

    Instead of splitting on raw words (which breaks mid-sentence), this
    accumulates complete sentences until approaching the token limit.
    Overlap is achieved by carrying over trailing sentences from the
    previous chunk into the next one.

    Args:
        text: Input text to chunk.
        chunk_size: Target maximum number of tokens per chunk.
        chunk_overlap: Target number of overlap tokens between consecutive chunks.

    Returns:
        List of text chunks.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than 0")
    if chunk_overlap < 0:
        raise ValueError("chunk_overlap must be non-negative")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    text = _normalize_text(text)
    sentences = _split_into_sentences(text)
    if not sentences:
        # Fallback to word-level if no sentences detected (e.g. single words).
        tokens = text.split()
        if not tokens:
            return []
        chunks: List[str] = []
        step = chunk_size - chunk_overlap
        for start in range(0, len(tokens), step):
            end = start + chunk_size
            chunk_tokens = tokens[start:end]
            if not chunk_tokens:
                break
            chunks.append(" ".join(chunk_tokens))
            if end >= len(tokens):
                break
        return chunks

    def _token_count(s: str) -> int:
        return len(s.split())

    chunks = []
    current_sentences: List[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = _token_count(sentence)

        # If a single sentence exceeds chunk_size, include it as its own chunk.
        if sent_tokens > chunk_size and not current_sentences:
            chunks.append(sentence)
            continue

        if current_tokens + sent_tokens > chunk_size and current_sentences:
            # Flush current chunk.
            chunks.append(" ".join(current_sentences))

            # Build overlap from trailing sentences of current chunk.
            overlap_sentences: List[str] = []
            overlap_tokens = 0
            for s in reversed(current_sentences):
                s_tok = _token_count(s)
                if overlap_tokens + s_tok > chunk_overlap and overlap_sentences:
                    break
                overlap_sentences.insert(0, s)
                overlap_tokens += s_tok

            current_sentences = overlap_sentences
            current_tokens = overlap_tokens

        current_sentences.append(sentence)
        current_tokens += sent_tokens

    # Flush remaining sentences.
    if current_sentences:
        remaining_text = " ".join(current_sentences)
        # Only add if it's not a near-duplicate of the last chunk.
        if not chunks or remaining_text != chunks[-1]:
            chunks.append(remaining_text)

    return chunks


def ingest_document(
    file_path: Path,
    chunk_size: int = 400,
    chunk_overlap: int = 100,
    source_name: Optional[str] = None,
) -> List[ChunkRecord]:
    """Load a document and convert it to chunk records."""
    raw_text = extract_text(file_path)
    if not raw_text.strip():
        return []

    source = source_name or file_path.name
    chunks = sliding_window_chunk_text(raw_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return [ChunkRecord(text=chunk_text, source=source, chunk_index=index) for index, chunk_text in enumerate(chunks)]
