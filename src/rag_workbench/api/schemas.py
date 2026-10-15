"""API request/response models.

These are thin HTTP contracts around the domain schemas; they add pagination
envelopes and error shapes but no business logic.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from rag_workbench.domain.schemas import (
    ChunkRecord,
    DocumentRecord,
    IngestionResult,
    ProfileRecord,
    SearchResponse,
    VectorIndexStatus,
    VectorRebuildResult,
)


class HealthResponse(BaseModel):
    status: str
    app_name: str
    version: str
    database: str
    document_count: int
    chunk_count: int
    profile_count: int


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=2000)
    profile_id: str | None = None
    mode: str = "bm25"
    top_k: int = Field(default=5, ge=1, le=100)
    collection: str | None = None
    candidates: int | None = Field(default=None, ge=1, le=500)
    query_mode: str = "and"
    rerank: bool = False
    reranker_model: str | None = Field(default=None, max_length=200)
    rerank_candidates: int | None = Field(default=None, ge=1, le=500)


class VectorRebuildRequest(BaseModel):
    profile_id: str | None = None
    all_profiles: bool = False
    drop: bool = True


class VectorReconcileRequest(BaseModel):
    profile_id: str
    repair: bool = False


class VectorRebuildResponse(BaseModel):
    results: list[VectorRebuildResult]


class VectorStatusResponse(BaseModel):
    statuses: list[VectorIndexStatus]


class DocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    documents: list[DocumentRecord]


class ProfileListResponse(BaseModel):
    total: int
    profiles: list[ProfileRecord]


class ChunkListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    chunks: list[ChunkRecord]


class IngestionJobResponse(BaseModel):
    job_id: str
    document_id: str | None
    profile_id: str
    collection: str
    state: str
    progress: float
    chunk_count: int
    started_at: datetime
    finished_at: datetime | None = None
    duration_ms: float | None = None
    error_code: str | None = None
    error_message: str | None = None
    diagnostics: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class ErrorResponse(BaseModel):
    error: str
    message: str


class DatasetSummary(BaseModel):
    name: str
    slug: str
    version: str
    source: str
    license: str
    description: str = ""
    relevance_mapping: str
    corpus_count: int
    query_count: int
    qrel_count: int
    evidence_count: int
    checksum: str
    created_at: str
    notes: list[str] = []
    directory: str | None = None


class DatasetListResponse(BaseModel):
    datasets: list[DatasetSummary]


class CorpusIngestRequest(BaseModel):
    dataset: str
    collection: str | None = None
    strategy: str | None = None
    max_tokens: int | None = Field(default=None, ge=8, le=4096)
    overlap_tokens: int | None = Field(default=None, ge=0, le=2048)
    profile_name: str | None = None
    embed: bool = False
    embedding_model: str | None = None
    embedding_dimension: int | None = Field(default=None, ge=8, le=8192)
    limit: int | None = Field(default=None, ge=1)


class CorpusIngestResponse(BaseModel):
    dataset: str
    collection: str
    profile_id: str
    documents: int
    new_documents: int
    duplicates: int
    chunks: int
    diagnostics: list[str] = []


class EvaluationRunRequest(BaseModel):
    dataset: str
    profile_id: str
    collection: str | None = None
    mode: str = "bm25"
    query_mode: str = "or"
    candidate_depth: int = Field(default=100, ge=1, le=500)
    top_k: int = Field(default=10, ge=1, le=100)
    k_values: list[int] = Field(default_factory=lambda: [1, 3, 5, 10])
    rerank: bool = False
    rerank_candidates: int | None = Field(default=None, ge=1, le=500)
    reranker_model: str | None = None
    candidates: int | None = Field(default=None, ge=1, le=500)
    max_queries: int | None = Field(default=None, ge=1)
    label: str | None = None
    subset: bool = False
    related_baseline: str | None = None


class ExperimentSummary(BaseModel):
    experiment_id: str
    source: str
    created_at: str | None = None
    dataset: str | None = None
    mode: str | None = None
    label: str | None = None
    rerank: bool | None = None
    subset: bool | None = None
    aggregate: dict[str, float] = {}
    candidate_metrics: dict[str, float] = {}
    counts: dict[str, int] = {}
    latency: dict = {}


class ExperimentListResponse(BaseModel):
    experiments: list[dict]


class AppConfigResponse(BaseModel):
    read_only: bool
    app_name: str
    version: str
    dense_available: bool
    reranking_available: bool
    supported_modes: list[str]
    default_k_values: list[int]
    default_candidate_depth: int
    recorded_experiments: int
    datasets: int


__all__ = [
    "AppConfigResponse",
    "ChunkListResponse",
    "CorpusIngestRequest",
    "CorpusIngestResponse",
    "DatasetListResponse",
    "DatasetSummary",
    "DocumentListResponse",
    "ErrorResponse",
    "EvaluationRunRequest",
    "ExperimentListResponse",
    "ExperimentSummary",
    "HealthResponse",
    "IngestionJobResponse",
    "IngestionResult",
    "ProfileListResponse",
    "SearchRequest",
    "SearchResponse",
    "VectorRebuildRequest",
    "VectorRebuildResponse",
    "VectorReconcileRequest",
    "VectorStatusResponse",
]
