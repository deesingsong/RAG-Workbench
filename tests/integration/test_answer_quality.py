"""Tests for the grounded-answer quality evaluation (offline, fake generator)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_workbench.answering.pipeline import AnswerService
from rag_workbench.answering.quality import (
    AnswerQualityDataset,
    AnswerQualityQuestion,
    AnswerQualityRunner,
    load_answer_quality_dataset,
)
from rag_workbench.api.app import create_app
from rag_workbench.config import Settings, find_repo_root
from rag_workbench.domain.schemas import AnswerPayload, GenerationResult
from rag_workbench.retrieval.service import SearchService
from tests.fakes import FakeAnswerGenerator


class _QuestionAwareGenerator(FakeAnswerGenerator):
    """Deterministic generator whose status depends on the question text."""

    def generate(self, request):  # type: ignore[override]
        question = request.question.lower()
        if "salary" in question:
            payload = AnswerPayload(
                status="insufficient_evidence",
                answer="",
                claims=[],
                reason="not stated in the passages",
            )
            return GenerationResult(
                model_id=self.model_id,
                payload=payload,
                structured=True,
                timings_ms={"generation": 1.0},
            )
        if "governing" in question and "termination" in question:
            payload = AnswerPayload(
                status="partially_answered",
                answer="Partially answerable [S1].",
                claims=[],
                unanswered=["governing law"],
            )
            return GenerationResult(
                model_id=self.model_id,
                payload=payload,
                structured=True,
                timings_ms={"generation": 1.0},
            )
        return super().generate(request)


def _dataset() -> AnswerQualityDataset:
    return AnswerQualityDataset(
        schema_version="1",
        name="small-answers",
        version="1",
        collection="contracts",
        questions=[
            AnswerQualityQuestion(
                id="q1",
                question="termination for convenience",
                expected_status="answered",
                reference_answer="sixty (60) days written notice",
                reference_keywords=["sixty (60) days"],
            ),
            AnswerQualityQuestion(
                id="q2",
                question="termination for convenience and governing law",
                expected_status="partially_answered",
            ),
            AnswerQualityQuestion(
                id="q3",
                question="salary of the chief executive officer",
                expected_status="insufficient_evidence",
            ),
        ],
    )


def test_quality_runner_aggregates_metrics(settings: Settings, contract_bytes) -> None:
    tuned = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        generation_enabled=True,
        _env_file=None,
    )
    app = create_app(tuned)
    with TestClient(app) as client:
        client.post(
            "/v1/documents",
            files={"file": ("sample_contract.txt", contract_bytes)},
            data={"collection": "contracts"},
        )
    database = app.state.database
    service = AnswerService(
        database,
        tuned,
        SearchService(database, tuned),
        generator_factory=lambda: _QuestionAwareGenerator(),
    )
    runner = AnswerQualityRunner(service)
    report, responses = runner.run_with_responses(
        _dataset(), profile_id=None, mode="bm25", query_mode="or", max_evidence=6
    )

    assert report["kind"] == "answer_quality"
    assert report["dataset"]["name"] == "small-answers"
    assert report["counts"]["questions"] == 3
    assert report["counts"]["insufficient_evidence"] >= 1
    assert report["metrics"]["correct_abstention_rate"] == 1.0
    assert report["metrics"]["semantic_support"] == "not_determined"
    assert report["metrics"]["citation_validity_rate"] is not None
    assert report["latency"]["generation_ms"]["count"] >= 1
    assert report["limitations"]
    assert len(responses) == 3
    database.dispose()


def test_quality_runner_offline_fake_never_abstains_when_evidence_present(
    settings: Settings, contract_bytes
) -> None:
    tuned = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        generation_enabled=True,
        _env_file=None,
    )
    app = create_app(tuned)
    with TestClient(app) as client:
        client.post(
            "/v1/documents",
            files={"file": ("sample_contract.txt", contract_bytes)},
            data={"collection": "contracts"},
        )
    database = app.state.database
    service = AnswerService(
        database,
        tuned,
        SearchService(database, tuned),
        generator_factory=lambda: FakeAnswerGenerator(),
    )
    report = AnswerQualityRunner(service).run(
        _dataset(), profile_id=None, mode="bm25", query_mode="or"
    )
    assert report["metrics"]["over_abstention_rate"] is not None
    database.dispose()


def test_committed_answer_quality_dataset_parses() -> None:
    path = find_repo_root() / "datasets" / "answer-quality" / "legal-contracts-answers.json"
    assert path.is_file(), "the bundled answer-quality dataset must be committed"
    dataset = load_answer_quality_dataset(path)
    assert dataset.name == "legal-contracts-answers"
    assert dataset.collection == "legal-contracts"
    assert len(dataset.questions) >= 8
    assert any(q.expected_status == "insufficient_evidence" for q in dataset.questions)
    assert any(q.expected_status == "partially_answered" for q in dataset.questions)
