"""Domain schemas shared by ingestion, retrieval, the API and the CLI.

These are plain Pydantic models. They carry the provenance contract: a chunk
knows which page(s) and section it came from *when that information was
extractable*, and ``None`` means genuinely unknown rather than a guess.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentStatus = Literal["indexed", "empty", "failed"]
JobState = Literal["queued", "running", "completed", "failed", "duplicate"]


class ParsedBlock(BaseModel):
    """A contiguous unit of parsed text (usually a paragraph) with provenance."""

    model_config = ConfigDict(frozen=True)

    text: str
    order: int
    page: int | None = None
    section_path: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class ParsedDocument(BaseModel):
    """Ordered parsed blocks for one source document, plus parsing diagnostics."""

    source_name: str
    mime_type: str
    checksum: str
    parser_version: str
    blocks: list[ParsedBlock] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    text_length: int = 0

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)


class ChunkData(BaseModel):
    """A chunk ready to be persisted, with deterministic ID and provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    order: int
    text: str
    token_count: int
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None


class SearchHit(BaseModel):
    """A single retrieval result with provenance and a normalized rank.

    ``rank`` is 1-based and always means "1 is best", independent of the
    underlying scoring direction. ``score`` is the raw backend score; for BM25
    over FTS5 a *smaller* (more negative) score is better, and the retriever
    sorts ascending accordingly.
    """

    rank: int
    chunk_id: str
    document_id: str
    source_name: str
    collection: str
    chunk_order: int
    text: str
    score: float
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None


class IngestionResult(BaseModel):
    """Outcome of ingesting one document into one index profile."""

    job_id: str
    document_id: str
    profile_id: str
    state: JobState
    is_duplicate: bool
    chunk_count: int
    source_name: str
    checksum: str
    collection: str
    diagnostics: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentDeleteResult(BaseModel):
    document_id: str
    profile_id: str | None
    chunks_deleted: int
    document_deleted: bool


class SearchResponse(BaseModel):
    """A retrieval response with configuration fingerprint and stage timings."""

    query: str
    mode: str
    profile_id: str
    profile_name: str
    config_hash: str
    collection: str | None = None
    top_k: int
    timings_ms: dict[str, float] = Field(default_factory=dict)
    hits: list[SearchHit] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    """A document row as returned by the API/CLI."""

    document_id: str
    collection: str
    source_name: str
    source_checksum: str
    mime_type: str
    byte_size: int
    imported_at: datetime
    status: DocumentStatus
    diagnostics: list[str] = Field(default_factory=list)
    chunk_count: int


class ProfileRecord(BaseModel):
    profile_id: str
    name: str
    config_hash: str
    parser_version: str
    chunker_strategy: str
    chunker_params: dict[str, object]
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    index_version: str
    created_at: datetime
    chunk_count: int
