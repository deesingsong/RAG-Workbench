"""Document ingestion, catalog, job and profile endpoints."""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile

from rag_workbench.api.deps import get_ingestion_service, get_settings
from rag_workbench.api.schemas import (
    ChunkListResponse,
    DocumentListResponse,
    DocumentRecord,
    IngestionJobResponse,
    IngestionResult,
    ProfileListResponse,
)
from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.domain.errors import FileTooLargeError, ValidationError
from rag_workbench.domain.schemas import DocumentDeleteResult
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.persistence.models import IngestionJob

router = APIRouter()


def _sanitize_filename(name: str) -> str:
    """Reduce a client-supplied name to a safe basename."""

    base = Path(name.replace("\\", "/")).name
    return base or "upload"


async def _read_limited(upload: UploadFile, max_bytes: int) -> bytes:
    """Read an upload, rejecting anything over ``max_bytes`` without buffering it all."""

    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await upload.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > max_bytes:
            raise FileTooLargeError("file exceeds the configured upload limit")
        chunks.append(chunk)
    return b"".join(chunks)


def _build_profile(
    settings: Settings,
    *,
    strategy: str | None,
    max_tokens: int | None,
    overlap_tokens: int | None,
    name: str | None,
    embed: bool = False,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
) -> IndexProfileSpec:
    try:
        chunker = ChunkerConfig(
            strategy=strategy or settings.default_chunk_strategy,  # type: ignore[arg-type]
            max_tokens=max_tokens or settings.default_max_tokens,
            overlap_tokens=(
                overlap_tokens if overlap_tokens is not None else settings.default_overlap_tokens
            ),
        )
    except ValueError as exc:
        raise ValidationError(f"invalid chunking configuration: {exc}") from exc

    if embed:
        try:
            return settings.dense_profile(
                name=name or f"dense-{chunker.strategy}-{chunker.max_tokens}",
                chunker=chunker,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
        except ValueError as exc:
            raise ValidationError(f"invalid dense configuration: {exc}") from exc

    profile_name = name or f"{chunker.strategy}-{chunker.max_tokens}"
    return IndexProfileSpec(name=profile_name, chunker=chunker)


@router.post("/documents", response_model=IngestionResult, status_code=201, tags=["documents"])
async def upload_document(
    file: UploadFile = File(...),
    collection: str | None = Form(default=None),
    strategy: str | None = Form(default=None),
    max_tokens: int | None = Form(default=None),
    overlap_tokens: int | None = Form(default=None),
    profile_name: str | None = Form(default=None),
    embed: bool = Form(default=False),
    embedding_model: str | None = Form(default=None),
    embedding_dimension: int | None = Form(default=None),
    settings: Settings = Depends(get_settings),
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionResult:
    data = await _read_limited(file, settings.max_upload_bytes)
    profile = _build_profile(
        settings,
        strategy=strategy,
        max_tokens=max_tokens,
        overlap_tokens=overlap_tokens,
        name=profile_name,
        embed=embed,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
    )
    return service.ingest_bytes(
        data,
        source_name=_sanitize_filename(file.filename or "upload"),
        collection=collection or settings.default_collection,
        profile=profile,
    )


@router.get("/documents", response_model=DocumentListResponse, tags=["documents"])
def list_documents(
    collection: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentListResponse:
    documents, total = service.list_documents(collection=collection, limit=limit, offset=offset)
    return DocumentListResponse(total=total, limit=limit, offset=offset, documents=documents)


@router.get("/documents/{document_id}", response_model=DocumentRecord, tags=["documents"])
def get_document(
    document_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentRecord:
    return service.get_document(document_id)


@router.delete("/documents/{document_id}", response_model=DocumentDeleteResult, tags=["documents"])
def delete_document(
    document_id: str,
    profile_id: str | None = Query(default=None),
    service: IngestionService = Depends(get_ingestion_service),
) -> DocumentDeleteResult:
    return service.delete_document(document_id, profile_id=profile_id)


@router.get("/ingestion-jobs/{job_id}", response_model=IngestionJobResponse, tags=["documents"])
def get_ingestion_job(
    job_id: str,
    service: IngestionService = Depends(get_ingestion_service),
) -> IngestionJobResponse:
    job: IngestionJob = service.get_job(job_id)
    return IngestionJobResponse(
        job_id=job.id,
        document_id=job.document_id,
        profile_id=job.profile_id,
        collection=job.collection,
        state=job.state,
        progress=job.progress,
        chunk_count=job.chunk_count,
        started_at=job.started_at,
        finished_at=job.finished_at,
        duration_ms=job.duration_ms,
        error_code=job.error_code,
        error_message=job.error_message,
        diagnostics=list(job.diagnostics or []),
        timings_ms=dict(job.timings_ms or {}),
    )


@router.get("/profiles", response_model=ProfileListResponse, tags=["documents"])
def list_profiles(
    service: IngestionService = Depends(get_ingestion_service),
) -> ProfileListResponse:
    profiles = service.list_profiles()
    return ProfileListResponse(total=len(profiles), profiles=profiles)


@router.get(
    "/profiles/{profile_id}/chunks",
    response_model=ChunkListResponse,
    tags=["documents"],
)
def list_chunks(
    profile_id: str,
    document_id: str | None = Query(default=None),
    limit: int = Query(default=200, ge=1, le=1000),
    offset: int = Query(default=0, ge=0),
    service: IngestionService = Depends(get_ingestion_service),
) -> ChunkListResponse:
    chunks, total = service.list_chunks(
        profile_id, document_id=document_id, limit=limit, offset=offset
    )
    return ChunkListResponse(total=total, limit=limit, offset=offset, chunks=chunks)
