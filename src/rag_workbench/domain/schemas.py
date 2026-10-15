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
JobState = Literal["queued", "running", "completed", "partial", "failed", "duplicate"]

#: How to read a hit's ``score``. BM25 over FTS5 is ascending (smaller is
#: better); cosine similarity and RRF are descending (larger is better).
ScoreKind = Literal["bm25", "cosine", "rrf"]


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
    # Clause-aware provenance. Populated only when it could be derived
    # deterministically; ``None`` means genuinely unknown, never a guess.
    section_number: str | None = None
    section_title: str | None = None
    parent_clause: str | None = None
    section_level: int | None = None
    diagnostics: list[str] = Field(default_factory=list)


class RerankBatch(BaseModel):
    """Cross-encoder scores for a batch of query-passage pairs.

    Scores are raw cross-encoder relevance logits: **larger is more relevant**,
    but they are not calibrated probabilities and must not be described as such.
    """

    scores: list[float]
    max_length: int | None = None
    truncated_count: int = 0
    diagnostics: list[str] = Field(default_factory=list)


class EmbeddingBatch(BaseModel):
    """Result of embedding a batch of texts, with truncation diagnostics."""

    vectors: list[list[float]]
    dimension: int
    diagnostics: list[str] = Field(default_factory=list)
    truncated_count: int = 0
    max_tokens_seen: int | None = None


class VectorPoint(BaseModel):
    """A vector ready to be upserted, keyed by its stable chunk ID."""

    id: str
    vector: list[float]
    payload: dict[str, object] = Field(default_factory=dict)


class VectorMatch(BaseModel):
    """A raw vector-store search result before provenance resolution."""

    id: str
    score: float
    payload: dict[str, object] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """A single retrieval result with provenance and a normalized rank.

    ``rank`` is 1-based and always means "1 is best", independent of the
    underlying scoring direction. ``score`` is the score of the selected mode and
    ``score_kind`` documents how to read it:

    * ``bm25``  — FTS5 score, *smaller* (more negative) is better.
    * ``cosine`` — semantic similarity, *larger* is better.
    * ``rrf``   — Reciprocal Rank Fusion score, *larger* is better.

    For hybrid results the original component ranks/scores are preserved in the
    ``bm25_*`` and ``dense_*`` fields so rank movements can be inspected.

    When reranking is enabled ``rank`` is the *final* (post-rerank) position and
    the pre-rerank position is captured in ``retrieval_rank``; ``rerank_score``
    holds the cross-encoder score and ``rank_movement`` is
    ``retrieval_rank - rank`` (positive means the passage moved up).
    """

    rank: int
    chunk_id: str
    document_id: str
    source_name: str
    collection: str
    chunk_order: int
    text: str
    score: float
    score_kind: ScoreKind = "bm25"
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    fused_score: float | None = None
    section_number: str | None = None
    section_title: str | None = None
    parent_clause: str | None = None
    # Reranking diagnostics (populated only when reranking ran).
    rerank_score: float | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    retrieval_score_kind: ScoreKind | None = None
    rank_movement: int | None = None


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
    candidates: int | None = None
    rerank: bool = False
    rerank_candidates: int | None = None
    reranker: dict[str, object] | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_ms: float | None = None
    hits: list[SearchHit] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """A stored chunk with full provenance, for the chunking laboratory."""

    chunk_id: str
    document_id: str
    profile_id: str
    source_name: str
    collection: str
    chunk_order: int
    text: str
    token_count: int
    section_path: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    parent_clause: str | None = None
    section_level: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    diagnostics: list[str] = Field(default_factory=list)


class VectorIndexStatus(BaseModel):
    """Reconciliation view of one profile's vector index versus SQLite."""

    profile_id: str
    collection: str
    embedding_model: str | None = None
    dimension: int | None = None
    chunk_count: int
    point_count: int
    missing_points: int
    orphan_points: int
    in_sync: bool


class VectorRebuildResult(BaseModel):
    """Outcome of rebuilding (or repairing) one profile's vector index."""

    profile_id: str
    collection: str
    chunks: int
    points_written: int
    orphan_points_deleted: int = 0
    duration_ms: float
    diagnostics: list[str] = Field(default_factory=list)


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
    embedding_revision: str | None = None
    similarity_metric: str = "cosine"
    normalize_embeddings: bool = True
    vector_collection: str | None = None
    index_version: str
    created_at: datetime
    chunk_count: int
