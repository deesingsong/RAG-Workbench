"""Explicit component interfaces.

These are :class:`typing.Protocol` definitions used where more than one concrete
implementation exists or is planned. Phase 1 ships concrete plain-text/Markdown/
PDF parsers, fixed-token and paragraph chunkers, and a BM25 sparse retriever; the
dense/hybrid/reranker interfaces arrive with their implementations in later
phases rather than as empty placeholders.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from sqlalchemy.orm import Session

from rag_workbench.config import ChunkerConfig
from rag_workbench.domain.schemas import ChunkData, ParsedDocument, SearchHit


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


__all__ = ["Chunker", "DocumentParser", "SparseRetriever"]
