"""Grounded-answer quality evaluation (Phase 5 §13).

A small, deterministic evaluation over the synthetic contracts that measures the
*answer pipeline*, not legal accuracy:

* structured-output validity (did the model satisfy the answer schema?);
* citation-id validity (do cited ids resolve to selected passages?);
* invalid-citation frequency;
* abstention behaviour (does the pipeline abstain on unanswerable questions?);
* whether reference evidence was selected into the context;
* context exclusions (retrieved but dropped);
* retrieval and generation latency, measured separately.

It deliberately does **not** score whether a cited passage truly supports a claim:
that is semantic support and is not deterministically decidable. The report says so
rather than inventing a support score.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from rag_workbench.answering.pipeline import AnswerService
from rag_workbench.answering.recorded import DEMO_SCHEMA_VERSION
from rag_workbench.domain.errors import DatasetError
from rag_workbench.domain.schemas import AnswerResponse
from rag_workbench.evaluation.runtime import latency_stats, runtime_environment

ExpectedStatus = Literal["answered", "partially_answered", "insufficient_evidence"]
_ANSWERABLE = ("answered", "partially_answered")


class AnswerQualityQuestion(BaseModel):
    id: str
    question: str
    expected_status: ExpectedStatus
    reference_answer: str = ""
    reference_keywords: list[str] = Field(default_factory=list)
    reference_sections: list[str] = Field(default_factory=list)
    note: str = ""


class AnswerQualityDataset(BaseModel):
    schema_version: str
    name: str
    version: str
    collection: str
    source: str = ""
    license: str = ""
    description: str = ""
    limitations: list[str] = Field(default_factory=list)
    questions: list[AnswerQualityQuestion]

    @field_validator("questions")
    @classmethod
    def _unique_ids(cls, questions: list[AnswerQualityQuestion]) -> list[AnswerQualityQuestion]:
        ids = [q.id for q in questions]
        if len(ids) != len(set(ids)):
            raise ValueError("answer-quality dataset has duplicate question ids")
        if not questions:
            raise ValueError("answer-quality dataset has no questions")
        return questions


def load_answer_quality_dataset(path: Path) -> AnswerQualityDataset:
    """Load and validate a grounded-answer dataset from a JSON file."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DatasetError(f"could not read answer-quality dataset: {exc}") from exc
    try:
        return AnswerQualityDataset.model_validate(payload)
    except Exception as exc:  # noqa: BLE001 - surface a single actionable error
        raise DatasetError(f"invalid answer-quality dataset '{path.name}': {exc}") from exc


class AnswerQualityRunner:
    """Run a grounded-answer dataset through :class:`AnswerService`."""

    def __init__(self, answer_service: AnswerService) -> None:
        self._service = answer_service

    def run(
        self,
        dataset: AnswerQualityDataset,
        **kwargs,
    ) -> dict:
        report, _ = self.run_with_responses(dataset, **kwargs)
        return report

    def run_with_responses(
        self,
        dataset: AnswerQualityDataset,
        *,
        profile_id: str | None = None,
        mode: str = "bm25",
        query_mode: str = "and",
        candidate_depth: int | None = None,
        rerank: bool = False,
        rerank_candidates: int | None = None,
        max_evidence: int | None = None,
        context_tokens: int | None = None,
        label: str | None = None,
    ) -> tuple[dict, list[tuple[AnswerQualityQuestion, AnswerResponse]]]:
        """Run the set once, returning both the report and the full responses."""

        cases: list[dict] = []
        responses: list[tuple[AnswerQualityQuestion, AnswerResponse]] = []
        retrieval_ms: list[float] = []
        generation_ms: list[float] = []
        total_ms: list[float] = []
        model_id: str | None = None

        for item in dataset.questions:
            response = self._service.answer(
                question=item.question,
                profile_id=profile_id,
                mode=mode,
                collection=dataset.collection,
                query_mode=query_mode,
                candidate_depth=candidate_depth,
                rerank=rerank,
                rerank_candidates=rerank_candidates,
                max_evidence=max_evidence,
                context_tokens=context_tokens,
            )
            responses.append((item, response))
            model_id = model_id or str(response.generation.get("model_id") or "")
            timings = response.timings_ms
            retrieval_ms.append(timings.get("retrieval", 0.0))
            if "generation" in timings:
                generation_ms.append(timings["generation"])
            total_ms.append(timings.get("total", 0.0))

            validation = response.citation_validation
            keywords = [k.lower() for k in item.reference_keywords]
            evidence_text = " ".join(e.text.lower() for e in response.evidence)
            present = (
                any(keyword in evidence_text for keyword in keywords) if keywords else None
            )
            cases.append(
                {
                    "id": item.id,
                    "question": item.question,
                    "expected_status": item.expected_status,
                    "status": response.status,
                    "status_correct": response.status == item.expected_status,
                    "answerable": item.expected_status in _ANSWERABLE,
                    "abstained": response.status == "insufficient_evidence",
                    "structural_valid": validation.structural_valid,
                    "citation_valid": validation.citation_valid,
                    "unknown_citation_ids": validation.unknown_source_ids,
                    "semantic_support": validation.semantic_support,
                    "reference_evidence_present": present,
                    "retrieved": response.context.retrieved,
                    "selected": response.context.selected,
                    "excluded": len(response.excluded_evidence),
                    "answer": response.answer,
                    "claims": [claim.model_dump() for claim in response.claims],
                    "diagnostics": response.diagnostics,
                    "timings_ms": dict(timings),
                }
            )

        metrics = self._aggregate(cases)
        report = {
            "schema_version": DEMO_SCHEMA_VERSION,
            "kind": "answer_quality",
            "report_id": f"aq-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S')}",
            "created_at": datetime.now(UTC).isoformat(),
            "source": "recorded",
            "label": label,
            "dataset": {
                "name": dataset.name,
                "version": dataset.version,
                "collection": dataset.collection,
                "source": dataset.source,
                "license": dataset.license,
            },
            "config": {
                "profile_id": profile_id,
                "mode": mode,
                "query_mode": query_mode,
                "candidate_depth": candidate_depth,
                "rerank": rerank,
                "rerank_candidates": rerank_candidates,
                "max_evidence": max_evidence,
                "context_tokens": context_tokens,
                "model_id": model_id,
            },
            "counts": {
                "questions": len(cases),
                "answered": sum(1 for c in cases if c["status"] == "answered"),
                "partially_answered": sum(
                    1 for c in cases if c["status"] == "partially_answered"
                ),
                "insufficient_evidence": sum(
                    1 for c in cases if c["status"] == "insufficient_evidence"
                ),
                "generation_error": sum(
                    1 for c in cases if c["status"] == "generation_error"
                ),
            },
            "metrics": metrics,
            "latency": {
                "retrieval_ms": latency_stats(retrieval_ms),
                "generation_ms": latency_stats(generation_ms),
                "total_ms": latency_stats(total_ms),
            },
            "runtime": runtime_environment(),
            "limitations": [
                *dataset.limitations,
                "Retrieval is BM25 by default; dense/rerank results depend on the "
                "installed optional extras and the chosen configuration.",
                "Generation depends on the local model; answer quality is not "
                "comparable across models without re-running the set.",
            ],
            "cases": cases,
        }
        return report, responses

    @staticmethod
    def _rate(numerator: int, denominator: int) -> float | None:
        return round(numerator / denominator, 4) if denominator else None

    @classmethod
    def _aggregate(cls, cases: list[dict]) -> dict:
        total = len(cases)
        answerable = [c for c in cases if c["answerable"]]
        unanswerable = [c for c in cases if c["expected_status"] == "insufficient_evidence"]
        with_keywords = [c for c in cases if c["reference_evidence_present"] is not None]
        structural_ok = sum(1 for c in cases if c["structural_valid"])
        citation_ok = sum(1 for c in cases if c["citation_valid"])
        invalid = sum(1 for c in cases if c["unknown_citation_ids"])
        abstained = sum(1 for c in cases if c["abstained"])
        correct_abstention = sum(
            1 for c in unanswerable if c["abstained"]
        )
        over_abstention = sum(1 for c in answerable if c["abstained"])
        evidence_present = sum(
            1 for c in with_keywords if c["reference_evidence_present"]
        )
        excluded = sum(1 for c in cases if c["excluded"])
        failures = sum(1 for c in cases if c["status"] == "generation_error")
        return {
            "structural_validity_rate": cls._rate(structural_ok, total),
            "citation_validity_rate": cls._rate(citation_ok, total),
            "invalid_citation_rate": cls._rate(invalid, total),
            "abstention_rate": cls._rate(abstained, total),
            "correct_abstention_rate": cls._rate(correct_abstention, len(unanswerable)),
            "over_abstention_rate": cls._rate(over_abstention, len(answerable)),
            "reference_evidence_selected_rate": cls._rate(evidence_present, len(with_keywords)),
            "context_exclusion_rate": cls._rate(excluded, total),
            "generation_failure_rate": cls._rate(failures, total),
            "semantic_support": "not_determined",
        }


__all__ = [
    "AnswerQualityDataset",
    "AnswerQualityQuestion",
    "AnswerQualityRunner",
    "load_answer_quality_dataset",
]
