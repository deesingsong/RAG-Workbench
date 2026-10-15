"""Deterministic test doubles.

``FakeEmbeddingProvider`` produces stable hashed bag-of-words vectors so the dense
and hybrid pipelines can be exercised end-to-end (real Qdrant store, real SQLite,
real fusion) **without** downloading a model. It is deliberately simple and is
never used to claim semantic retrieval quality — it only lets tests assert that
the plumbing, ranking order, provenance and idempotency behave correctly.
"""

from __future__ import annotations

import hashlib
import math
import re
from collections.abc import Sequence

from rag_workbench.domain.schemas import EmbeddingBatch

_TOKEN = re.compile(r"[a-z0-9]+")


def _stable_hash(token: str, dimension: int) -> int:
    """Process-independent hash bucket (Python's str hash is randomized)."""

    digest = hashlib.md5(token.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % dimension


class FakeEmbeddingProvider:
    """A deterministic, offline stand-in for a Sentence Transformers model."""

    def __init__(
        self,
        *,
        dimension: int = 64,
        model_id: str = "fake/hash-bow",
        revision: str | None = "v1",
        normalize: bool = True,
        max_tokens: int = 512,
    ) -> None:
        self._dimension = dimension
        self._model_id = model_id
        self._revision = revision
        self._normalize = normalize
        self._max_tokens = max_tokens

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def dimension(self) -> int:
        return self._dimension

    @property
    def normalize(self) -> bool:
        return self._normalize

    @property
    def max_tokens(self) -> int | None:
        return self._max_tokens

    def _vector(self, text: str) -> list[float]:
        vector = [0.0] * self._dimension
        for token in _TOKEN.findall(text.lower()):
            vector[_stable_hash(token, self._dimension)] += 1.0
        if self._normalize:
            norm = math.sqrt(sum(value * value for value in vector))
            if norm:
                vector = [value / norm for value in vector]
        return vector

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        prepared = list(texts)
        return EmbeddingBatch(
            vectors=[self._vector(text) for text in prepared],
            dimension=self._dimension,
            max_tokens_seen=None,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)
