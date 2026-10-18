"""API tests for the answer endpoint, read-only enforcement and recorded demos.

The real FastAPI app is driven over an isolated SQLite database with a
deterministic fake generator injected, so the whole pipeline runs offline.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_workbench.answering.pipeline import AnswerService
from rag_workbench.answering.recorded import write_answer_demo, write_quality_report
from rag_workbench.api.app import create_app
from rag_workbench.config import Settings
from tests.fakes import FakeAnswerGenerator


def _upload(client: TestClient, name: str, data: bytes, **form):
    return client.post(
        "/v1/documents",
        files={"file": (name, data)},
        data={k: str(v) for k, v in form.items()},
    )


def _generation_settings(settings: Settings, **overrides) -> Settings:
    return Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        generation_enabled=True,
        _env_file=None,
        **overrides,
    )


def _install_generator(client: TestClient, generator: FakeAnswerGenerator) -> None:
    app = client.app
    app.state.answer_service = AnswerService(
        app.state.database,
        app.state.settings,
        app.state.search_service,
        generator_factory=lambda: generator,
    )


@pytest.fixture
def answer_client(settings: Settings):
    tuned = _generation_settings(settings)
    app = create_app(tuned)
    app.state.answer_service = AnswerService(
        app.state.database,
        tuned,
        app.state.search_service,
        generator_factory=lambda: FakeAnswerGenerator(),
    )
    with TestClient(app) as client:
        yield client


def test_answer_returns_grounded_answer_with_citations(answer_client, contract_bytes) -> None:
    _upload(answer_client, "sample_contract.txt", contract_bytes, collection="contracts")
    response = answer_client.post(
        "/v1/answer",
        json={"question": "termination for convenience", "candidate_depth": 10},
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["status"] == "answered"
    assert body["claims"] and body["claims"][0]["citations"]
    assert body["evidence"], "selected context must be returned for inspection"
    assert body["retrieved_hits"], "the full retrieval must be returned"
    assert body["citation_validation"]["citation_valid"] is True
    assert body["citation_validation"]["semantic_support"] == "not_determined"
    assert body["context"]["selected"] >= 1
    assert "retrieval" in body["timings_ms"]
    assert "generation" in body["timings_ms"]
    assert body["retrieval"]["mode"] == "bm25"
    assert body["generation"]["model_id"] == "fake-answerer"
    assert body["generator_available"] is True


def test_unknown_citation_is_flagged(answer_client, contract_bytes) -> None:
    _install_generator(answer_client, FakeAnswerGenerator(citations=["S99"]))
    _upload(answer_client, "sample_contract.txt", contract_bytes)
    response = answer_client.post("/v1/answer", json={"question": "termination"})
    assert response.status_code == 200
    body = response.json()
    assert body["citation_validation"]["citation_valid"] is False
    assert body["citation_validation"]["unknown_source_ids"] == ["S99"]
    assert any(d["code"] == "unknown_citation" for d in body["diagnostics"])


def test_malformed_generation_is_a_generation_error(answer_client, contract_bytes) -> None:
    _install_generator(answer_client, FakeAnswerGenerator(mode="malformed"))
    _upload(answer_client, "sample_contract.txt", contract_bytes)
    response = answer_client.post("/v1/answer", json={"question": "termination"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "generation_error"
    assert body["citation_validation"]["structural_valid"] is False
    assert any(d["code"] == "generation_failed" for d in body["diagnostics"])


def test_generator_timeout_is_reported_not_raised(answer_client, contract_bytes) -> None:
    _install_generator(answer_client, FakeAnswerGenerator(mode="timeout"))
    _upload(answer_client, "sample_contract.txt", contract_bytes)
    response = answer_client.post("/v1/answer", json={"question": "termination"})
    assert response.status_code == 200
    assert response.json()["status"] == "generation_error"


def test_unavailable_generator_returns_503(answer_client, contract_bytes) -> None:
    _install_generator(answer_client, FakeAnswerGenerator(available=False))
    _upload(answer_client, "sample_contract.txt", contract_bytes)
    response = answer_client.post("/v1/answer", json={"question": "termination"})
    assert response.status_code == 503
    assert response.json()["error"] == "generator_unavailable"


def test_no_evidence_abstains_without_calling_the_model(answer_client, contract_bytes) -> None:
    generator = FakeAnswerGenerator()
    _install_generator(answer_client, generator)
    _upload(answer_client, "sample_contract.txt", contract_bytes)
    response = answer_client.post(
        "/v1/answer", json={"question": "zzzqqq unrelated gibberish xyzzy"}
    )
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "insufficient_evidence"
    assert body["generation"]["attempted"] is False
    assert generator.generate_calls == 0
    assert any(d["code"] == "no_evidence_retrieved" for d in body["diagnostics"])


def test_generation_disabled_returns_503(client) -> None:
    response = client.post("/v1/answer", json={"question": "anything"})
    assert response.status_code == 503
    assert response.json()["error"] == "generator_unavailable"


def test_read_only_refuses_generation(settings: Settings) -> None:
    read_only = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        read_only=True,
        generation_enabled=True,
        _env_file=None,
    )
    with TestClient(create_app(read_only)) as client:
        response = client.post("/v1/answer", json={"question": "anything"})
    assert response.status_code == 403
    assert response.json()["error"] == "read_only_mode"


def test_search_still_works_alongside_answer(answer_client, contract_bytes) -> None:
    _upload(answer_client, "sample_contract.txt", contract_bytes)
    search = answer_client.post(
        "/v1/search", json={"query": "termination for convenience", "top_k": 3}
    )
    assert search.status_code == 200
    assert search.json()["hits"]


def test_recorded_answer_demos_endpoints(settings: Settings, tmp_path: Path) -> None:
    answers_dir = tmp_path / "recorded_answers"
    demo = {
        "schema_version": "1",
        "kind": "answer_demo",
        "demo_id": "ans-20260101T000000-deadbeef",
        "created_at": "2026-01-01T00:00:00Z",
        "question": "How long is the term?",
        "status": "answered",
        "answer": "Three years [S1].",
        "claims": [],
        "citation_validation": {"structural_valid": True, "citation_valid": True},
        "context": {"retrieved": 1, "selected": 1},
        "retrieval": {"collection": "legal", "mode": "bm25"},
        "generation": {"model_id": "qwen2.5:1.5b-instruct"},
    }
    write_answer_demo(answers_dir, demo)
    write_quality_report(
        answers_dir,
        {
            "kind": "answer_quality",
            "report_id": "aq-test",
            "created_at": "2026-01-02T00:00:00Z",
            "recommended": True,
            "config": {"model_id": "qwen2.5:1.5b-instruct", "max_evidence": 4},
            "metrics": {
                "citation_validity_rate": 0.9,
                "semantic_support": "not_determined",
                "complete_reference_coverage_rate": 0.5,
            },
            "counts": {"questions": 2},
        },
    )
    write_quality_report(
        answers_dir,
        {
            "kind": "answer_quality",
            "report_id": "aq-test-2",
            "created_at": "2026-01-03T00:00:00Z",
            "config": {"model_id": "qwen2.5:1.5b-instruct", "max_evidence": 6},
            "metrics": {
                "citation_validity_rate": 1.0,
                "semantic_support": "not_determined",
                "complete_reference_coverage_rate": 0.75,
            },
            "counts": {"questions": 2},
        },
    )
    tuned = _generation_settings(settings, recorded_answers_dir=answers_dir)
    with TestClient(create_app(tuned)) as client:
        listing = client.get("/v1/answer-demos")
        assert listing.status_code == 200
        body = listing.json()
        assert body["total"] == 1
        assert body["demos"][0]["demo_id"] == "ans-20260101T000000-deadbeef"

        single = client.get("/v1/answer-demos/ans-20260101T000000-deadbeef")
        assert single.status_code == 200
        assert single.json()["demo"]["question"] == "How long is the term?"

        missing = client.get("/v1/answer-demos/does-not-exist")
        assert missing.status_code == 404
        assert missing.json()["error"] == "answer_not_found"

        quality = client.get("/v1/answer-quality")
        assert quality.status_code == 200
        # An explicitly recommended report wins even when a newer one exists.
        assert quality.json()["report_id"] == "aq-test"

        reports = client.get("/v1/answer-quality-reports")
        assert reports.status_code == 200
        report_body = reports.json()
        assert report_body["total"] == 2
        assert [r["report_id"] for r in report_body["reports"]] == ["aq-test-2", "aq-test"]
        assert report_body["reports"][0]["max_evidence"] == 6
        assert report_body["reports"][0]["recommended"] is False
        assert report_body["reports"][1]["recommended"] is True
        assert report_body["reports"][1]["complete_reference_coverage_rate"] == 0.5


def test_config_reports_generation_capabilities(settings: Settings, tmp_path: Path) -> None:
    # Point at a closed port so the availability probe fails fast and the test is
    # deterministic regardless of whether a local Ollama happens to be running.
    tuned = _generation_settings(
        settings,
        recorded_answers_dir=tmp_path / "answers",
        ollama_base_url="http://127.0.0.1:9",
    )
    with TestClient(create_app(tuned)) as client:
        body = client.get("/v1/config").json()
        diagnostics = client.get("/v1/config/diagnostics").json()
    assert body["generation_enabled"] is True
    assert body["generation_available"] is False
    assert body["recorded_answer_demos"] == 0
    # The public config exposes capabilities, not infrastructure details. The
    # resolved paths and the Ollama endpoint are operator-only and served by a
    # separate route that the public read-only deployment refuses.
    assert "recorded_answers_dir" not in body
    assert "ollama_model" not in body
    assert "ollama_base_url" not in body
    assert diagnostics["recorded_answers_dir"].endswith("answers")
    assert diagnostics["ollama_base_url"] == "http://127.0.0.1:9"
