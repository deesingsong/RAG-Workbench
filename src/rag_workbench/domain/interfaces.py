"""Explicit component interfaces.

These are :class:`typing.Protocol` definitions used where more than one concrete
implementation exists or is planned. Phase 2 ships plain-text/Markdown/PDF
parsers, fixed-token and paragraph chunkers, a BM25 sparse retriever, a local
Qdrant vector store and a Sentence Transformers embedding provider; the fake
embedding provider used in tests implements the same ``EmbeddingProvider``
protocol, which is the seam that keeps tests offline.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from rag_workbench.config import ChunkerConfig
from rag_workbench.domain.schemas import (
    ChunkData,
    EmbeddingBatch,
    GenerationRequest,
    GenerationResult,
    ParsedDocument,
    RerankBatch,
    SearchHit,
    VectorMatch,
    VectorPoint,
)


@runtime_checkable
class DocumentParser(Protocol):
    """Parse raw bytes into ordered blocks with provenance."""

    def parse(self, data: bytes, *, source_name: str, mime_type: str) -> ParsedDocument: ...


@runtime_checkable
class Chunker(Protocol):
    """Turn a parsed document into deterministic, profile-scoped chunks."""

    def chunk(
        self,
        document: ParsedDocument,
        config: ChunkerConfig,
        *,
        profile_key: str,
        document_id: str,
    ) -> list[ChunkData]: ...


@runtime_checkable
class SparseRetriever(Protocol):
    """Lexical retrieval over a profile-scoped index."""

    def search(
        self,
        session: Session,
        *,
        query: str,
        profile_id: str,
        top_k: int,
        collection: str | None = None,
    ) -> list[SearchHit]: ...


@runtime_checkable
class EmbeddingProvider(Protocol):
    """Turn text into vectors with explicit, inspectable model identity.

    ``dimension``, ``model_id``, ``revision``, ``normalize`` and ``max_tokens``
    describe exactly what produced a vector. Implementations may be lazy: the
    model is not loaded until the first embedding call.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def dimension(self) -> int: ...

    @property
    def normalize(self) -> bool: ...

    @property
    def max_tokens(self) -> int | None: ...

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch: ...

    def embed_query(self, text: str) -> list[float]: ...


@runtime_checkable
class VectorStore(Protocol):
    """Persistent vector storage scoped by named collection.

    Only the state needed for retrieval lives here; SQLite remains authoritative
    for documents, chunk text and provenance.
    """

    def ensure_collection(self, name: str, *, dimension: int, metric: str = "cosine") -> None: ...

    def has_collection(self, name: str) -> bool: ...

    def collection_dimension(self, name: str) -> int | None: ...

    def upsert(self, name: str, points: Sequence[VectorPoint]) -> int: ...

    def search(
        self,
        name: str,
        vector: Sequence[float],
        *,
        limit: int,
        collection: str | None = None,
    ) -> list[VectorMatch]: ...

    def delete_ids(self, name: str, ids: Sequence[str]) -> int: ...

    def list_ids(self, name: str) -> set[str]: ...

    def count(self, name: str) -> int: ...

    def drop(self, name: str) -> None: ...


@runtime_checkable
class Reranker(Protocol):
    """Re-score query-passage pairs with a cross-encoder.

    Implementations may be lazy: the model is not loaded until the first scoring
    call. ``model_id``, ``revision`` and ``max_length`` describe exactly what
    produced a score. Scores are relevance logits (larger is more relevant) and
    are **not** calibrated probabilities.
    """

    @property
    def model_id(self) -> str: ...

    @property
    def revision(self) -> str | None: ...

    @property
    def max_length(self) -> int | None: ...

    def ensure_loaded(self) -> float:
        """Load the model if needed and return the one-off initialization time in ms."""

    def score(self, query: str, passages: Sequence[str]) -> RerankBatch: ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """Produce a grounded, structured answer from trusted instructions and evidence.

    Implementations may be lazy and must not be contacted until generation is
    actually requested. ``generate`` returns a :class:`GenerationResult` that
    carries the parsed payload (or ``None`` plus an ``error`` when the model's
    output could not be recovered) together with model metadata and timings;
    failures never raise out of a well-behaved call except for an *unavailable*
    generator, which raises :class:`~rag_workbench.domain.errors.GeneratorUnavailableError`.
    """

    @property
    def model_id(self) -> str: ...

    def ensure_available(self) -> None: ...

    def generate(self, request: GenerationRequest) -> GenerationResult: ...


__all__ = [
    "AnswerGenerator",
    "Chunker",
    "DocumentParser",
    "EmbeddingProvider",
    "Reranker",
    "SparseRetriever",
    "VectorStore",
]
