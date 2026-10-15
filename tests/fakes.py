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

from rag_workbench.domain.schemas import EmbeddingBatch, RerankBatch

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


class FakeReranker:
    """Deterministic, offline stand-in for a cross-encoder.

    It scores each passage by the number of distinct query terms it contains
    (plus optional per-needle boosts), which is reproducible and lets tests assert
    that reranking reorders candidates, separates the rerank score from the
    retrieval score, and reports truncation. It makes **no** claim about semantic
    relevance quality.
    """

    def __init__(
        self,
        *,
        model_id: str = "fake/cross-encoder",
        revision: str | None = "v1",
        max_length: int | None = None,
        boosts: dict[str, float] | None = None,
        init_ms: float = 1.5,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._max_length = max_length
        self._boosts = dict(boosts or {})
        self._init_ms = init_ms
        self._loaded = False
        self.ensure_calls = 0
        self.score_calls = 0
        self.last_passages: list[str] = []

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def max_length(self) -> int | None:
        return self._max_length

    def ensure_loaded(self) -> float:
        self.ensure_calls += 1
        if self._loaded:
            return 0.0
        self._loaded = True
        return self._init_ms

    def score(self, query: str, passages: Sequence[str]) -> RerankBatch:
        self.score_calls += 1
        prepared = list(passages)
        self.last_passages = prepared
        query_terms = set(_TOKEN.findall(query.lower()))
        scores: list[float] = []
        for passage in prepared:
            lowered = passage.lower()
            passage_terms = set(_TOKEN.findall(lowered))
            score = float(len(query_terms & passage_terms))
            for needle, boost in self._boosts.items():
                if needle.lower() in lowered:
                    score += boost
            scores.append(score)

        truncated = 0
        diagnostics: list[str] = []
        if self._max_length is not None:
            truncated = sum(
                1 for passage in prepared if len(_TOKEN.findall(passage)) > self._max_length
            )
            if truncated:
                diagnostics.append(
                    f"{truncated} query-passage pair(s) exceeded the model max length "
                    f"({self._max_length} tokens) and were truncated by the model"
                )
        return RerankBatch(
            scores=scores,
            max_length=self._max_length,
            truncated_count=truncated,
            diagnostics=diagnostics,
        )
