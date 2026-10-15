"""Opt-in real-Ollama test.

Excluded from the default suite (``-m 'not ... and not ollama'``). Run with a
local Ollama serving the configured model::

    uv run pytest -m ollama

It skips (does not fail) when Ollama or the model is unavailable, so it never
misreports a real generation result.
"""

from __future__ import annotations

import pytest

from rag_workbench.answering.ollama import build_answer_generator
from rag_workbench.answering.pipeline import AnswerService
from rag_workbench.config import Settings
from rag_workbench.domain.errors import GeneratorUnavailableError
from rag_workbench.retrieval.service import SearchService


@pytest.mark.ollama
def test_real_ollama_grounded_answer(settings: Settings, client, contract_bytes) -> None:
    upload = client.post(
        "/v1/documents",
        files={"file": ("sample_contract.txt", contract_bytes)},
        data={"collection": "contracts"},
    )
    assert upload.status_code == 201

    generator = build_answer_generator(settings)
    try:
        generator.ensure_available()
    except GeneratorUnavailableError as exc:
        pytest.skip(f"local Ollama is not available: {exc.message}")

    database = client.app.state.database
    service = AnswerService(
        database,
        settings,
        SearchService(database, settings),
        generator_factory=lambda: generator,
    )
    response = service.answer(
        question="What is the notice period for termination for convenience?",
        collection="contracts",
        query_mode="or",
    )
    assert response.status in {
        "answered",
        "partially_answered",
        "insufficient_evidence",
        "generation_error",
    }
    assert response.timings_ms.get("retrieval", 0) >= 0
    if response.status in {"answered", "partially_answered"}:
        assert response.citation_validation.structural_valid is True
        assert response.citation_validation.citation_valid is True
