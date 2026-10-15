"""Evaluation endpoints: datasets, corpus ingestion, experiments and exports.

These are thin HTTP contracts over
:class:`~rag_workbench.evaluation.service.EvaluationService`; all behaviour lives
in the service so the CLI and Streamlit produce identical results.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from fastapi.responses import PlainTextResponse, Response
from pydantic import ValidationError as PydanticValidationError

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
    settings: Settings = Depends(get_settings),
    service: EvaluationService = Depends(get_evaluation_service),
) -> dict:
    available = dense_feature_available()
    return {
        "read_only": settings.read_only,
        "app_name": settings.app_name,
        "version": __import__("rag_workbench").__version__,
        "dense_available": available,
        "reranking_available": available,
        "supported_modes": ["bm25", "dense", "hybrid"],
        "default_k_values": list(settings.default_eval_k),
        "default_candidate_depth": settings.default_candidate_depth,
        "recorded_experiments": len(service.list_experiments(include_recorded=True)),
        "datasets": len(service.list_datasets()),
        # Deployment diagnostics: the resolved asset locations. These are paths,
        # not secrets, and make it obvious when a deployment cannot see the
        # committed datasets/experiments (e.g. a wrong repository root).
        "repo_root": str(find_repo_root()),
        "bundled_datasets_dir": str(settings.bundled_datasets_dir),
        "bundled_datasets_exists": settings.bundled_datasets_dir.is_dir(),
        "recorded_experiments_dir": str(settings.recorded_experiments_dir),
        "recorded_experiments_exists": settings.recorded_experiments_dir.is_dir(),
    }
