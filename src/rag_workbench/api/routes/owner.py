"""Owner-only evaluation endpoint (Laboratory Controls §5/§6).

A dedicated, explicitly authorized boundary: the ordinary public evaluation
endpoint stays writable-to-nobody on a read-only deployment, while the operator
can run a small, bounded BM25 evaluation here. Authorization is enforced by the
:func:`~rag_workbench.api.deps.require_owner` dependency, never by the frontend,
and the server-side policy caps queries, depth and runtime.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_workbench.api.deps import (
    get_owner_evaluation_service,
    require_owner,
)
from rag_workbench.api.schemas import OwnerEvaluationRequest
from rag_workbench.evaluation.owner import OwnerEvaluationService

router = APIRouter()


@router.get("/owner/capabilities", tags=["owner"])
def owner_capabilities(
    service: OwnerEvaluationService = Depends(get_owner_evaluation_service),
    _: None = Depends(require_owner),
) -> dict:
    """Return the enforced bounds to an authenticated owner (never the token)."""

    return {"enabled": True, "policy": service.policy.summary()}


@router.post("/owner/evaluations", tags=["owner"])
def run_owner_evaluation(
    request: OwnerEvaluationRequest,
    service: OwnerEvaluationService = Depends(get_owner_evaluation_service),
    _: None = Depends(require_owner),
) -> dict:
    """Run one bounded owner evaluation against an allowed bundled dataset."""

    return service.run(
        dataset=request.dataset,
        query_mode=request.query_mode,
        max_queries=request.max_queries,
        candidate_depth=request.candidate_depth,
        top_k=request.top_k,
        label=request.label,
    )
