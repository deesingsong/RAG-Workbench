"""Answer endpoints: live graded generation plus recorded demonstrations.

``POST /v1/answer`` runs the optional local generation pipeline; it is refused in
read-only mode and when generation is disabled. The ``/v1/answer-demos`` endpoints
serve committed *recorded* answer runs so the public, read-only deployment can
still demonstrate the answer pipeline without a model.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends

from rag_workbench.answering.pipeline import AnswerService
from rag_workbench.answering.recorded import (
    answer_demo_summary,
    find_answer_demo,
    load_answer_demos,
    load_quality_report,
    load_quality_reports,
    quality_report_summary,
)
from rag_workbench.api.deps import get_answer_service, get_settings, require_generation
from rag_workbench.api.schemas import (
    AnswerDemoListResponse,
    AnswerDemoResponse,
    AnswerDemoSummary,
    AnswerQualityReportListResponse,
    AnswerQualityReportSummary,
    AnswerRequest,
)
from rag_workbench.config import Settings
from rag_workbench.domain.errors import AnswerNotFoundError
from rag_workbench.domain.schemas import AnswerResponse

router = APIRouter()


@router.post("/answer", response_model=AnswerResponse, tags=["answer"])
def answer(
    request: AnswerRequest,
    service: AnswerService = Depends(get_answer_service),
    _: None = Depends(require_generation),
) -> AnswerResponse:
    return service.answer(
        question=request.question,
        profile_id=request.profile_id,
        mode=request.mode,
        collection=request.collection,
        query_mode=request.query_mode,
        candidate_depth=request.candidate_depth,
        candidates=request.candidates,
        rerank=request.rerank,
        reranker_model=request.reranker_model,
        rerank_candidates=request.rerank_candidates,
        max_evidence=request.max_evidence,
        context_tokens=request.context_tokens,
        temperature=request.temperature,
        num_predict=request.num_predict,
        seed=request.seed,
    )


@router.get("/answer-demos", response_model=AnswerDemoListResponse, tags=["answer"])
def list_answer_demos(settings: Settings = Depends(get_settings)) -> AnswerDemoListResponse:
    demos = [
        AnswerDemoSummary.model_validate(answer_demo_summary(record))
        for record in load_answer_demos(settings.recorded_answers_dir)
    ]
    return AnswerDemoListResponse(total=len(demos), demos=demos)


@router.get("/answer-demos/{demo_id}", response_model=AnswerDemoResponse, tags=["answer"])
def get_answer_demo(
    demo_id: str, settings: Settings = Depends(get_settings)
) -> AnswerDemoResponse:
    record = find_answer_demo(settings.recorded_answers_dir, demo_id)
    if record is None:
        raise AnswerNotFoundError(f"no recorded answer demo with id '{demo_id}'")
    return AnswerDemoResponse(demo=record)


@router.get("/answer-quality", tags=["answer"])
def answer_quality(settings: Settings = Depends(get_settings)) -> dict:
    """Return the newest recorded grounded-answer quality report, if any."""

    report = load_quality_report(settings.recorded_answers_dir)
    if report is None:
        raise AnswerNotFoundError("no recorded answer-quality report is available")
    return report


@router.get(
    "/answer-quality-reports",
    response_model=AnswerQualityReportListResponse,
    tags=["answer"],
)
def list_answer_quality_reports(
    settings: Settings = Depends(get_settings),
) -> AnswerQualityReportListResponse:
    """List every recorded quality report, newest first, for configuration comparison."""

    reports = [
        AnswerQualityReportSummary.model_validate(quality_report_summary(report))
        for report in load_quality_reports(settings.recorded_answers_dir)
    ]
    return AnswerQualityReportListResponse(total=len(reports), reports=reports)
