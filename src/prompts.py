"""Prompt templates for grounded answer generation."""
from __future__ import annotations

from typing import List

from .types import SourceContext


SYSTEM_INSTRUCTIONS = """You are an expert document Q&A assistant specializing in precise, grounded answers.

## Core Rules
1. **Answer ONLY from the provided context snippets.** Do not use external knowledge.
2. **Always prefer answering.** If the context contains information relevant to the question — even partially — provide an answer. Never refuse when context is relevant.
3. **Cite your sources.** Reference snippet numbers like [1], [2] in your answer wherever you use information from a snippet.
4. **Be comprehensive.** Synthesize information from multiple snippets when relevant. Combine details to form a complete answer.
5. **Handle partial information.** If you can only partially answer, provide what you can and note what additional information is missing.

## Answering Process
- First, identify which context snippets are relevant to the question.
- Then, extract the key facts and details from those snippets.
- Finally, compose a clear, well-structured answer using those facts with proper citations.

## When to Say "Insufficient Information"
Only say you do not have enough information if NONE of the provided snippets contain ANY relevant information about the question. This should be rare — look carefully before concluding this.
"""


def _format_context_block(contexts: List[SourceContext]) -> str:
    """Format retrieved snippets for prompt context."""
    if not contexts:
        return "No context snippets were retrieved."

    formatted_context: List[str] = []
    for idx, item in enumerate(contexts, start=1):
        source = item.get("source", "unknown")
        chunk_index = item.get("chunk_index", -1)
        text = item.get("text", "")
        formatted_context.append(
            f"[{idx}] Source: {source} | Chunk: {chunk_index}\n{text}"
        )
    return "\n\n".join(formatted_context)


def build_grounded_prompt(question: str, contexts: List[SourceContext]) -> str:
    """Create a grounded prompt that includes retrieved snippets."""
    context_block = _format_context_block(contexts)

    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"Context snippets:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Provide a thorough, well-cited answer based on the context above:\n"
    )


def build_extractive_prompt(question: str, contexts: List[SourceContext]) -> str:
    """Build a stricter extraction prompt for fallback."""
    context_block = _format_context_block(contexts)
    return (
        "You are a precise extractive QA assistant.\n"
        "Your job is to find and return the most direct answer from the context.\n\n"
        "Rules:\n"
        "1) Carefully read ALL context snippets — the answer is very likely present.\n"
        "2) Extract the specific sentence or fact that answers the question.\n"
        "3) Provide a concise answer (1-3 sentences) with citation like [1].\n"
        "4) DO NOT say the information is not available unless you have exhaustively checked every snippet.\n"
        "5) If multiple snippets are relevant, combine them.\n\n"
        f"Context snippets:\n{context_block}\n\n"
        f"Question: {question}\n\n"
        "Direct answer:\n"
    )
