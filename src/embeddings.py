"""Local sentence-transformer embedding model wrapper."""
from __future__ import annotations

import hashlib
import math
import re
import warnings
from typing import List

from sentence_transformers import SentenceTransformer


class LocalEmbeddingModel:
    """Encodes text locally using sentence-transformers."""

    def __init__(self, model_name: str) -> None:
        self.model: SentenceTransformer | None
        self._fallback_dim = 384
        self.backend: str

        try:
            self.model = SentenceTransformer(model_name)
            self.backend = "sentence-transformers"
        except Exception as exc:
            self.model = None
            self.backend = "hash-fallback"
            warnings.warn(
                (
                    "Falling back to hash-based embeddings because sentence-transformer model "
                    f"'{model_name}' could not be loaded: {exc}"
                ),
                RuntimeWarning,
                stacklevel=2,
            )

    def _normalize(self, values: List[float]) -> List[float]:
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]

    def _hash_embed(self, text: str) -> List[float]:
        # Deterministic lightweight fallback that works without external model downloads.
        vector = [0.0] * self._fallback_dim
        tokens = re.findall(r"\w+", text.lower())
        for token in tokens:
            digest = hashlib.sha256(token.encode("utf-8")).digest()
            index = int.from_bytes(digest[:4], "big") % self._fallback_dim
            sign = 1.0 if (digest[4] & 1) else -1.0
            vector[index] += sign
        return self._normalize(vector)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of chunk texts."""
        if not texts:
            return []
        if self.model is None:
            return [self._hash_embed(text) for text in texts]
        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embeddings.tolist()

    def embed_query(self, query: str) -> List[float]:
        """Embed a query string."""
        if self.model is None:
            return self._hash_embed(query)
        embedding = self.model.encode(
            [query],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        return embedding[0].tolist()
