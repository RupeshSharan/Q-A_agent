"""Gemini API wrapper for answer generation."""
from __future__ import annotations

import re
import time
from typing import Optional

from google import genai
from google.genai import errors
from google.genai import types as genai_types

from .prompts import build_extractive_prompt, build_grounded_prompt
from .types import SourceContext


class GeminiAnswerGenerator:
    """Generates grounded answers with Gemini."""

    def __init__(
        self,
        api_key: str,
        model_name: str,
        max_retries: int = 3,
        retry_wait_seconds: int = 60,
        temperature: float = 0.1,
    ) -> None:
        if not api_key:
            raise ValueError("Gemini API key is required")
        self.client = genai.Client(api_key=api_key)
        self.model_name = model_name
        self.max_retries = max_retries
        self.retry_wait_seconds = retry_wait_seconds
        self.temperature = temperature

    def _generation_config(self, max_tokens: int = 2048) -> genai_types.GenerateContentConfig:
        """Build generation config with low temperature for factual answers."""
        return genai_types.GenerateContentConfig(
            temperature=self.temperature,
            top_p=0.9,
            max_output_tokens=max_tokens,
        )

    def correct_query(self, raw_query: str) -> str:
        """Fix spelling and grammar errors in a user query using Gemini.

        Uses a lightweight prompt with minimal tokens so it's fast.
        Falls back to the original query if anything goes wrong.
        """
        if not raw_query.strip():
            return raw_query

        correction_prompt = (
            "Fix any spelling and grammar errors in the following question. "
            "Return ONLY the corrected question, nothing else. "
            "If the question is already correct, return it unchanged.\n\n"
            f"Question: {raw_query}\n\n"
            "Corrected question:"
        )
        try:
            config = self._generation_config(max_tokens=256)
            response = self.client.models.generate_content(
                model=self.model_name,
                contents=correction_prompt,
                config=config,
            )
            text = getattr(response, "text", None)
            if text and text.strip():
                corrected = text.strip().strip('"').strip("'")
                # Sanity check: corrected text shouldn't be wildly different in length.
                if len(corrected) < len(raw_query) * 4 and len(corrected) > 0:
                    return corrected
        except Exception:
            pass
        return raw_query

    @staticmethod
    def _status_code(error: Exception) -> Optional[int]:
        """Extract HTTP-like status code from Gemini SDK errors."""
        status_code = getattr(error, "status_code", None)
        if isinstance(status_code, int):
            return status_code
        code = getattr(error, "code", None)
        if isinstance(code, int):
            return code
        return None

    @staticmethod
    def _looks_insufficient(text: str) -> bool:
        """Heuristic check for refusal-style answers."""
        normalized = text.lower()
        markers = [
            "i do not have enough information",
            "not enough information",
            "cannot determine",
            "can't determine",
            "not provided in the context",
            "not in the provided context",
            "the context does not",
            "the provided context does not",
            "the context doesn't",
            "no information about",
            "i cannot find",
            "i can't find",
            "does not contain",
            "doesn't contain",
            "is not mentioned",
            "isn't mentioned",
            "no relevant information",
            "insufficient information",
        ]
        return any(marker in normalized for marker in markers)

    @staticmethod
    def _tokenize_query(question: str) -> set[str]:
        """Tokenize question into searchable keywords."""
        tokens = re.findall(r"[a-zA-Z0-9]{3,}", question.lower())
        stopwords = {
            "what",
            "which",
            "when",
            "where",
            "who",
            "whom",
            "whose",
            "why",
            "how",
            "much",
            "many",
            "does",
            "did",
            "have",
            "with",
            "from",
            "about",
            "that",
            "this",
            "these",
            "those",
            "into",
            "your",
            "their",
            "there",
            "then",
            "the",
            "and",
            "for",
            "are",
            "but",
            "not",
            "you",
            "all",
            "can",
            "her",
            "was",
            "one",
            "our",
            "out",
        }
        return {t for t in tokens if t not in stopwords}

    def _extractive_sentence_fallback(self, question: str, contexts: list[SourceContext]) -> Optional[str]:
        """Fallback extractor when model incorrectly refuses despite relevant context.

        Weights sentences by keyword overlap AND position (earlier sentences
        in a chunk are more likely to contain key information).
        """
        query_terms = self._tokenize_query(question)
        if not query_terms:
            return None

        best_sentence = ""
        best_score: float = 0
        best_idx = 0

        for idx, ctx in enumerate(contexts, start=1):
            text = ctx.get("text", "")
            if not text:
                continue
            # Lightweight sentence split for local extractive fallback.
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+|\n+", text) if s.strip()]
            for sent_pos, sentence in enumerate(sentences):
                sentence_terms = set(re.findall(r"[a-zA-Z0-9]{3,}", sentence.lower()))
                keyword_score = len(query_terms.intersection(sentence_terms))
                # Position bonus: earlier sentences get slight boost.
                position_bonus = max(0, 0.5 - (sent_pos * 0.05))
                score = keyword_score + position_bonus
                if score > best_score:
                    best_score = score
                    best_sentence = sentence
                    best_idx = idx

        if best_score < 1.0 or not best_sentence:
            return None
        return f"{best_sentence} [{best_idx}]"

    def generate_answer(self, question: str, contexts: list[SourceContext]) -> str:
        """Build prompt from retrieved context and call Gemini with retry on 429."""
        prompt = build_grounded_prompt(question, contexts)
        config = self._generation_config()

        for attempt in range(self.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=config,
                )
                text = getattr(response, "text", None)
                if text and text.strip():
                    answer = text.strip()
                    if self._looks_insufficient(answer) and contexts:
                        # Second pass with stricter extraction instructions.
                        extractive_prompt = build_extractive_prompt(question, contexts)
                        extractive_response = self.client.models.generate_content(
                            model=self.model_name,
                            contents=extractive_prompt,
                            config=config,
                        )
                        extractive_text = getattr(extractive_response, "text", None)
                        if extractive_text and extractive_text.strip():
                            extracted_answer = extractive_text.strip()
                            if not self._looks_insufficient(extracted_answer):
                                return extracted_answer

                        # Final local fallback: pick best matching sentence from contexts.
                        local_fallback = self._extractive_sentence_fallback(question, contexts)
                        if local_fallback:
                            return local_fallback
                    return answer
                return "I could not generate a response from the model output."
            except Exception as exc:
                status_code = self._status_code(exc)
                is_last_attempt = attempt == (self.max_retries - 1)

                if status_code == 429:
                    print(
                        f"Rate limit hit (attempt {attempt + 1} of {self.max_retries})."
                    )
                    if is_last_attempt:
                        return "Max retries reached due to Gemini rate limits. Please retry later."
                    print(
                        f"Waiting {self.retry_wait_seconds} seconds before retrying Gemini request..."
                    )
                    time.sleep(self.retry_wait_seconds)
                    continue

                # Keep non-rate-limit failures user-visible for easier debugging.
                if isinstance(exc, errors.APIError):
                    return f"Unexpected Gemini API error ({status_code or 'unknown'}): {exc}"
                return f"Unexpected Gemini client error: {exc}"

        return "Gemini request failed after retries."
