"""API request/response models.

These are thin HTTP contracts around the domain schemas; they add pagination
envelopes and error shapes but no business logic.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from rag_workbench.domain.schemas import (
    DocumentRecord,
    IngestionResult,
    ProfileRecord,
    SearchResponse,
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
    top_k: int = Field(default=5, ge=1, le=50)
    collection: str | None = None


class DocumentListResponse(BaseModel):
    total: int
    limit: int
    offset: int
    documents: list[DocumentRecord]


class ProfileListResponse(BaseModel):
    total: int
    profiles: list[ProfileRecord]


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


__all__ = [
    "DocumentListResponse",
    "ErrorResponse",
    "HealthResponse",
    "IngestionJobResponse",
    "IngestionResult",
    "ProfileListResponse",
    "SearchRequest",
    "SearchResponse",
]
