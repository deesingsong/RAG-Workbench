"""Evaluation endpoints: datasets, corpus ingestion, experiments and exports.

These are thin HTTP contracts over
:class:`~rag_workbench.evaluation.service.EvaluationService`; all behaviour lives
in the service so the CLI and Streamlit produce identical results.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query, Request
from fastapi.responses import PlainTextResponse, Response
from pydantic import ValidationError as PydanticValidationError

from rag_workbench.answering.ollama import probe_generation
from rag_workbench.answering.recorded import load_answer_demos
from rag_workbench.api.deps import (
    get_evaluation_service,
    get_settings,
    require_writable,
)
from rag_workbench.api.schemas import (
    CorpusIngestRequest,
    CorpusIngestResponse,
    DatasetListResponse,
    DatasetSummary,
    EvaluationRunRequest,
)
from rag_workbench.config import Settings, find_repo_root
from rag_workbench.domain.errors import ValidationError
from rag_workbench.evaluation.runner import EvalRunConfig
from rag_workbench.evaluation.service import EvaluationService
from rag_workbench.ingestion.optional import dense_feature_available

router = APIRouter()


def _capability(*, available: bool, permitted: bool, reason: str) -> dict[str, object]:
    """One non-sensitive capability status for ``/v1/config`` (Laboratory Controls §8).

    ``available`` is whether the dependency/model exists, ``permitted`` is whether
    the deployment's configuration and authorization policy allow the operation.
    The *session-enabled* state lives only in the frontend and is never reported
    here, so it can never be confused with a deployment permission.
    """

    return {"available": bool(available), "permitted": bool(permitted), "reason": reason}


@router.get("/datasets", response_model=DatasetListResponse, tags=["evaluation"])
def list_datasets(
    service: EvaluationService = Depends(get_evaluation_service),
) -> DatasetListResponse:
    return DatasetListResponse.model_validate({"datasets": service.list_datasets()})


@router.get("/datasets/{name}", response_model=DatasetSummary, tags=["evaluation"])
def dataset_detail(
    name: str,
    service: EvaluationService = Depends(get_evaluation_service),
) -> DatasetSummary:
    return DatasetSummary.model_validate(service.dataset_summary(name))


@router.post(
    "/datasets/ingest",
    response_model=CorpusIngestResponse,
    tags=["evaluation"],
    dependencies=[Depends(require_writable)],
)
def ingest_dataset(
    request: CorpusIngestRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> CorpusIngestResponse:
    report = service.ingest_dataset(
        request.dataset,
        collection=request.collection,
        strategy=request.strategy,
        max_tokens=request.max_tokens,
        overlap_tokens=request.overlap_tokens,
        profile_name=request.profile_name,
        embed=request.embed,
        embedding_model=request.embedding_model,
        embedding_dimension=request.embedding_dimension,
        limit=request.limit,
    )
    return CorpusIngestResponse.model_validate(report.model_dump())


@router.post(
    "/evaluations",
    tags=["evaluation"],
    dependencies=[Depends(require_writable)],
)
def run_evaluation(
    request: EvaluationRunRequest,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    config = _build_run_config(request)
    return service.run(config)


def _build_run_config(request: EvaluationRunRequest) -> EvalRunConfig:
    """Translate a run request, surfacing validation as a typed API error.

    ``EvalRunConfig`` enforces cross-field rules (for example that the rerank
    pool is not smaller than ``top_k``); those must reach the client as a 422
    with a stable code, not an opaque 500.
    """

    try:
        return EvalRunConfig(
            dataset=request.dataset,
            profile_id=request.profile_id,
            collection=request.collection,
            mode=request.mode,
            query_mode=request.query_mode,
            candidate_depth=request.candidate_depth,
            top_k=request.top_k,
            k_values=tuple(request.k_values),
            rerank=request.rerank,
            rerank_candidates=request.rerank_candidates,
            reranker_model=request.reranker_model,
            candidates=request.candidates,
            max_queries=request.max_queries,
            label=request.label,
            subset=request.subset,
            related_baseline=request.related_baseline,
        )
    except PydanticValidationError as exc:
        details = "; ".join(
            error["msg"] for error in exc.errors(include_url=False, include_input=False)
        )
        code = (
            "invalid_candidates"
            if "rerank_candidates" in details
            else "invalid_evaluation_config"
        )
        raise ValidationError(details or "invalid evaluation configuration", code=code) from exc


@router.get("/experiments", tags=["evaluation"])
def list_experiments(
    dataset: str | None = Query(default=None),
    include_recorded: bool = Query(default=True),
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    return {
        "experiments": service.list_experiments(
            dataset=dataset, include_recorded=include_recorded
        )
    }


@router.get("/experiments/{experiment_id}", tags=["evaluation"])
def get_experiment(
    experiment_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    return service.get_experiment(experiment_id)


@router.get("/experiments/{experiment_id}/queries/{query_id}", tags=["evaluation"])
def get_experiment_query(
    experiment_id: str,
    query_id: str,
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    return service.get_query(experiment_id, query_id)


@router.get(
    "/experiments/{experiment_id}/export",
    tags=["evaluation"],
    response_model=None,
)
def export_experiment(
    experiment_id: str,
    format: str = Query(default="json", pattern="^(json|csv)$"),
    service: EvaluationService = Depends(get_evaluation_service),
) -> Response:
    if format == "csv":
        csv_text = service.export_csv(experiment_id)
        return PlainTextResponse(
            content=csv_text,
            media_type="text/csv",
            headers={
                "Content-Disposition": f'attachment; filename="{experiment_id}.csv"'
            },
        )
    return service.export_json(experiment_id)


@router.get("/config", tags=["system"])
def app_config(
    request: Request,
    settings: Settings = Depends(get_settings),
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    available = dense_feature_available()
    generation = probe_generation(settings)
    answer_demos = load_answer_demos(settings.recorded_answers_dir)
    scaling_service = request.app.state.scaling_service
    scaling_benchmarks = scaling_service.list_benchmarks()
    owner_service = request.app.state.owner_evaluation_service
    owner_auth = request.app.state.owner_auth_service
    generation_available = bool(generation.get("available"))

    # Laboratory Controls capability status: available vs permitted, with a clear
    # reason when an operation cannot be enabled. Session-enabled is frontend-only.
    eval_available = True  # BM25 retrieval and the bundled datasets ship in core
    eval_capability = _capability(
        available=eval_available,
        permitted=not settings.read_only,
        reason=(
            "evaluation runs are disabled for public visitors"
            if settings.read_only
            else ""
        ),
    )
    if not settings.generation_enabled:
        generation_reason = (
            "live generation requires an Ollama server; set "
            "RAG_WORKBENCH_GENERATION_ENABLED=true and run Ollama to enable it"
        )
    elif settings.read_only:
        generation_reason = "live generation is disabled for public visitors"
    elif not generation_available:
        generation_reason = str(
            generation.get("detail") or "no local model is currently available"
        )
    else:
        generation_reason = ""
    generation_capability = _capability(
        available=generation_available,
        permitted=(
            settings.generation_enabled
            and not settings.read_only
            and generation_available
        ),
        reason=generation_reason,
    )
    ingestion_capability = _capability(
        available=True,
        permitted=not settings.read_only,
        reason=(
            "document ingestion is disabled for public visitors"
            if settings.read_only
            else ""
        ),
    )
    # Owner Mode: the capability is whether a passkey could be used at all
    # (relying party configured and at least one credential registered). The
    # visitor's *own* authentication status is deliberately not reported here.
    owner_available = owner_auth.available and owner_auth.credential_count > 0
    if not owner_auth.available:
        owner_reason = (
            "owner passkey authentication is not configured on this deployment"
        )
    elif owner_auth.credential_count == 0:
        owner_reason = "no owner passkey is registered yet"
    elif not settings.owner_evaluations_enabled:
        owner_reason = (
            "owner evaluations are disabled; set "
            "RAG_WORKBENCH_OWNER_EVALUATIONS_ENABLED=true to enable them"
        )
    else:
        owner_reason = ""
    owner_capability = _capability(
        available=owner_available,
        permitted=settings.owner_evaluations_enabled and owner_available,
        reason=owner_reason,
    )

    # Public capability payload: only non-sensitive facts the frontend needs.
    # Absolute filesystem paths and internal endpoints are deliberately absent --
    # they are served by the operator-only ``/v1/config/diagnostics`` route below.
    # The deployment revision *is* kept here: it is useful for detecting a stale
    # backend and discloses nothing about the host filesystem.
    return {
        "read_only": settings.read_only,
        "autoseed_sample": settings.autoseed_sample,
        "app_name": settings.app_name,
        "version": __import__("rag_workbench").__version__,
        # The revision this process built at startup, distinct from the package
        # version; a stale (hot-reloaded) backend reports an older revision than
        # the source tree on disk and the UI warns.
        "revision": getattr(request.app.state, "revision", None),
        "revision_source": getattr(request.app.state, "revision_source", "unavailable"),
        "dense_available": available,
        "reranking_available": available,
        "supported_modes": ["bm25", "dense", "hybrid"],
        "default_k_values": list(settings.default_eval_k),
        "default_candidate_depth": settings.default_candidate_depth,
        "recorded_experiments": len(service.list_experiments(include_recorded=True)),
        "datasets": len(service.list_datasets()),
        # Whether the committed/recorded assets were found, as plain booleans (no
        # paths). Useful to the UI and harmless to disclose.
        "bundled_datasets_exists": settings.bundled_datasets_dir.is_dir(),
        "recorded_experiments_exists": settings.recorded_experiments_dir.is_dir(),
        "recorded_answers_exists": settings.recorded_answers_dir.is_dir(),
        "recorded_scaling_exists": settings.recorded_scaling_dir.is_dir(),
        # Optional answer generation (Phase 5). Live generation is opt-in; the
        # public demo reports it disabled and serves recorded demos instead.
        "generation_enabled": settings.generation_enabled,
        "generation_available": bool(generation.get("available")),
        "recorded_answer_demos": len(answer_demos),
        # Scalability (Phase 6). Recorded locally measured benchmarks plus the
        # design data are exposed read-only; the capacity calculator is pure
        # arithmetic and needs no model or extra dependency.
        "recorded_scaling_benchmarks": len(scaling_benchmarks),
        "scaling_max_documents": settings.scaling_max_documents,
        # Laboratory Controls (runtime capability status) and the optional owner
        # evaluation boundary. Only non-sensitive facts are exposed; the owner
        # secret, session token and any visitor authentication status are never
        # included.
        "laboratory": {
            "evaluation": eval_capability,
            "generation": generation_capability,
            "ingestion": ingestion_capability,
            "owner_evaluation": owner_capability,
        },
        "owner_evaluations_enabled": settings.owner_evaluations_enabled,
        "owner_evaluation_policy": owner_service.policy.summary(),
        "passkey": owner_auth.public_summary(),
    }


@router.get(
    "/config/diagnostics",
    tags=["system"],
    dependencies=[Depends(require_writable)],
)
def config_diagnostics(
    settings: Settings = Depends(get_settings),
) -> dict:
    """Resolved asset locations and internal endpoints for the operator.

    Gated behind :func:`~rag_workbench.api.deps.require_writable`, so the public
    read-only deployment returns 403 and cannot disclose its filesystem layout or
    internal endpoints. A local/self-hosted (writable) deployment can still see
    where its datasets, recorded artifacts and Ollama endpoint resolve, which is
    what makes a misconfigured deploy obvious. Nothing here is a secret.
    """

    return {
        "repo_root": str(find_repo_root()),
        "bundled_datasets_dir": str(settings.bundled_datasets_dir),
        "recorded_experiments_dir": str(settings.recorded_experiments_dir),
        "recorded_answers_dir": str(settings.recorded_answers_dir),
        "recorded_scaling_dir": str(settings.recorded_scaling_dir),
        "ollama_base_url": settings.ollama_base_url,
        "ollama_model": settings.ollama_model,
    }
