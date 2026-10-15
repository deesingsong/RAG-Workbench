"""Search endpoint (Phase 1: BM25 only)."""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_workbench.api.deps import get_search_service
from rag_workbench.api.schemas import SearchRequest, SearchResponse
from rag_workbench.retrieval.service import SearchService

router = APIRouter()


@router.post("/search", response_model=SearchResponse, tags=["search"])
def search(
    request: SearchRequest,
    service: SearchService = Depends(get_search_service),
) -> SearchResponse:
    return service.search(
        query=request.query,
        profile_id=request.profile_id,
        mode=request.mode,
        top_k=request.top_k,
        collection=request.collection,
    )
