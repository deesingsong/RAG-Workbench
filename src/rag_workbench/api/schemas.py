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


class AnswerRequest(BaseModel):
    """A grounded-answer request: retrieval config plus generation bounds.

    ``candidate_depth`` (first-stage pool) and ``max_evidence`` (how many passages
    reach the model) are deliberately separate, so context selection can be
    inspected independently of retrieval depth.
    """

    question: str = Field(min_length=1, max_length=2000)
    profile_id: str | None = None
    mode: str = "bm25"
    collection: str | None = None
    query_mode: str = "and"
    candidate_depth: int | None = Field(default=None, ge=1, le=100)
    candidates: int | None = Field(default=None, ge=1, le=500)
    rerank: bool = False
    reranker_model: str | None = Field(default=None, max_length=200)
    rerank_candidates: int | None = Field(default=None, ge=1, le=500)
    max_evidence: int | None = Field(default=None, ge=1, le=50)
    context_tokens: int | None = Field(default=None, ge=128, le=32768)
    temperature: float | None = Field(default=None, ge=0.0, le=2.0)
    num_predict: int | None = Field(default=None, ge=1, le=8192)
    seed: int | None = None


class AnswerDemoSummary(BaseModel):
    demo_id: str | None = None
    created_at: str | None = None
    source: str = "recorded"
    question: str | None = None
    status: str | None = None
    answer_preview: str | None = None
    claim_count: int = 0
    citation_valid: bool | None = None
    structural_valid: bool | None = None
    semantic_support: str | None = None
    evidence_selected: int | None = None
    evidence_retrieved: int | None = None
    collection: str | None = None
    mode: str | None = None
    model_id: str | None = None
    total_ms: float | None = None
    generation_ms: float | None = None
    diagnostic_codes: list[str] = Field(default_factory=list)


class AnswerDemoListResponse(BaseModel):
    total: int
    demos: list[AnswerDemoSummary]


class AnswerDemoResponse(BaseModel):
    """An ``answer_demo`` artifact's raw record (already validated on load)."""

    demo: dict


class AnswerQualityReportSummary(BaseModel):
    """A compact summary of one recorded answer-quality report, for comparison."""

    report_id: str | None = None
    created_at: str | None = None
    source: str = "recorded"
    label: str | None = None
    recommended: bool = False
    dataset: str | None = None
    model_id: str | None = None
    mode: str | None = None
    query_mode: str | None = None
    max_evidence: int | None = None
    context_tokens: int | None = None
    seed: int | None = None
    prompt_variant: str | None = None
    counts: dict | None = None
    structural_validity_rate: float | None = None
    citation_validity_rate: float | None = None
    complete_reference_coverage_rate: float | None = None
    reference_evidence_unit_coverage_rate: float | None = None
    correct_abstention_rate: float | None = None
    status_accuracy_rate: float | None = None
    generation_ms_p50: float | None = None


class AnswerQualityReportListResponse(BaseModel):
    total: int
    reports: list[AnswerQualityReportSummary]


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
    autoseed_sample: bool = False
    app_name: str
    version: str
    revision: str | None = None
    revision_source: str = "unavailable"
    dense_available: bool
    reranking_available: bool
    supported_modes: list[str]
    default_k_values: list[int]
    default_candidate_depth: int
    recorded_experiments: int
    datasets: int
    repo_root: str = ""
    bundled_datasets_dir: str = ""
    bundled_datasets_exists: bool = False
    recorded_experiments_dir: str = ""
    recorded_experiments_exists: bool = False
    generation_enabled: bool = False
    generation_available: bool = False
    generation_available_detail: str = ""
    ollama_model: str = ""
    ollama_base_url: str = ""
    recorded_answer_demos: int = 0
    recorded_answers_dir: str = ""
    recorded_answers_exists: bool = False
    recorded_scaling_benchmarks: int = 0
    recorded_scaling_dir: str = ""
    recorded_scaling_exists: bool = False
    scaling_max_documents: int = 0


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
