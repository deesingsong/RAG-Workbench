"""Vector index management endpoints (status, rebuild, reconcile)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from rag_workbench.api.deps import get_vector_indexer
from rag_workbench.api.schemas import (
    VectorRebuildRequest,
    VectorRebuildResponse,
    VectorReconcileRequest,
    VectorStatusResponse,
)
from rag_workbench.domain.errors import ValidationError
from rag_workbench.domain.schemas import VectorIndexStatus, VectorRebuildResult
from rag_workbench.ingestion.vector_index import VectorIndexer

router = APIRouter()


def _target_profiles(
    indexer: VectorIndexer, profile_id: str | None, all_profiles: bool
) -> list[str]:
    if all_profiles:
        return indexer.dense_profile_ids()
    if profile_id:
        return [profile_id]
    raise ValidationError(
        "specify either profile_id or all_profiles=true", code="missing_target"
    )


@router.post("/vectors/rebuild", response_model=VectorRebuildResponse, tags=["vectors"])
def rebuild_vectors(
    request: VectorRebuildRequest,
    indexer: VectorIndexer = Depends(get_vector_indexer),
) -> VectorRebuildResponse:
    results: list[VectorRebuildResult] = [
        indexer.rebuild_profile(profile_id, drop=request.drop)
        for profile_id in _target_profiles(indexer, request.profile_id, request.all_profiles)
    ]
    return VectorRebuildResponse(results=results)


@router.post("/vectors/reconcile", response_model=VectorIndexStatus, tags=["vectors"])
def reconcile_vectors(
    request: VectorReconcileRequest,
    indexer: VectorIndexer = Depends(get_vector_indexer),
) -> VectorIndexStatus:
    return indexer.reconcile_profile(request.profile_id, repair=request.repair)


@router.get("/vectors/status", response_model=VectorStatusResponse, tags=["vectors"])
def vector_status(
    profile_id: str | None = Query(default=None),
    indexer: VectorIndexer = Depends(get_vector_indexer),
) -> VectorStatusResponse:
    targets = [profile_id] if profile_id else indexer.dense_profile_ids()
    return VectorStatusResponse(
        statuses=[indexer.reconcile_profile(target, repair=False) for target in targets]
    )
