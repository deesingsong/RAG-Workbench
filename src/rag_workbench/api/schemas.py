"""API request/response models.

These are thin HTTP contracts around the domain schemas; they add pagination
envelopes and error shapes but no business logic.
"""

from __future__ import annotations

import json
from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rag_workbench.domain.schemas import (
    ChunkRecord,
    DocumentRecord,
    IngestionResult,
    ProfileRecord,
    SearchResponse,
    VectorIndexStatus,
    VectorRebuildResult,
)

#: Absolute upper bound on a single authenticator response payload. Real WebAuthn
#: responses are a few kilobytes; this keeps a hostile request from forcing large
#: JSON parsing before the cryptographic checks even start.
_MAX_CREDENTIAL_BYTES = 64 * 1024


def _validate_credential_payload(value: dict) -> dict:
    """Reject oversized or structurally implausible authenticator responses.

    Only shape and size are checked here; the cryptographic validity is decided
    by the WebAuthn verification library, never by this validation.
    """

    if "id" not in value or "response" not in value:
        raise ValueError("a WebAuthn credential needs at least 'id' and 'response'")
    try:
        encoded = json.dumps(value)
    except (TypeError, ValueError) as exc:
        raise ValueError("WebAuthn credential is not JSON-serialisable") from exc
    if len(encoded) > _MAX_CREDENTIAL_BYTES:
        raise ValueError("WebAuthn credential payload is too large")
    return value


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


class OwnerEvaluationRequest(BaseModel):
    """A bounded owner-evaluation request (Laboratory Controls §5/§6).

    Deliberately narrow: no retrieval mode, no reranking and no profile choice.
    The server forces BM25 and enforces the stricter configured bounds; the
    field ceilings here mirror the absolute maxima so an oversized request is a
    validation error before any work starts.
    """

    dataset: str | None = None
    query_mode: Literal["and", "or"] = "or"
    max_queries: int | None = Field(default=None, ge=1, le=100)
    candidate_depth: int | None = Field(default=None, ge=1, le=200)
    top_k: int | None = Field(default=None, ge=1, le=50)
    label: str | None = Field(default=None, max_length=120)


class WebAuthnOptionsResponse(BaseModel):
    """A server-generated WebAuthn challenge plus the options for the browser."""

    handle: str
    options: str


class WebAuthnRegisterVerifyRequest(BaseModel):
    """The owner's authenticator response to a registration challenge."""

    handle: str = Field(min_length=1, max_length=256)
    credential: dict
    label: str | None = Field(default=None, max_length=80)

    @field_validator("credential")
    @classmethod
    def _bounded_credential(cls, value: dict) -> dict:
        return _validate_credential_payload(value)


class WebAuthnLoginVerifyRequest(BaseModel):
    """The owner's authenticator assertion response for authentication."""

    handle: str = Field(min_length=1, max_length=256)
    credential: dict

    @field_validator("credential")
    @classmethod
    def _bounded_credential(cls, value: dict) -> dict:
        return _validate_credential_payload(value)


class WebAuthnCredentialResponse(BaseModel):
    """The verified *public* credential record to persist in trusted config."""

    credential: dict


class OwnerSessionResponse(BaseModel):
    """Visitor owner-authentication status. Never includes the session bearer."""

    authenticated: bool
    session_token: str | None = None
    credential_label: str | None = None
    issued_at: str | None = None
    expires_at: str | None = None
    absolute_expires_at: str | None = None


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
    # Laboratory Controls: per-capability availability/permission (never the
    # session toggle, which belongs to the frontend) and the non-sensitive owner
    # policy. No secret, session token or visitor authentication status is ever
    # included here.
    laboratory: dict = {}
    owner_evaluations_enabled: bool = False
    owner_evaluation_policy: dict = {}
    passkey: dict = {}


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
    "OwnerEvaluationRequest",
    "OwnerSessionResponse",
    "ProfileListResponse",
    "SearchRequest",
    "SearchResponse",
    "VectorRebuildRequest",
    "VectorRebuildResponse",
    "VectorReconcileRequest",
    "VectorStatusResponse",
    "WebAuthnCredentialResponse",
    "WebAuthnLoginVerifyRequest",
    "WebAuthnOptionsResponse",
    "WebAuthnRegisterVerifyRequest",
]
