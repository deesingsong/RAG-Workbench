"""Unit tests for the bounded owner-evaluation policy (Laboratory Controls §5/§6).

These exercise the server-side bounds in isolation from FastAPI: defaults,
rejection of out-of-range requests, the dataset allowlist and the concurrency
gate. The HTTP authorization boundary is covered separately in
``tests/integration/test_laboratory_controls_api.py``.
"""

from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from rag_workbench.config import Settings
from rag_workbench.domain.errors import OwnerLimitError
from rag_workbench.evaluation.owner import (
    OwnerEvaluationPolicy,
    OwnerEvaluationService,
)


class _FakeEvaluationService:
    """Records the ingested dataset and the config handed to the runner."""

    def __init__(self) -> None:
        self.ingested: list[tuple[str, dict]] = []
        self.configs: list[object] = []

    def ingest_dataset(self, name: str, **kwargs):
        self.ingested.append((name, kwargs))
        return SimpleNamespace(profile_id="p-test", collection=name)

    def run(self, config):
        self.configs.append(config)
        return {"experiment_id": "exp-test", "config": config.model_dump(mode="json")}


def _settings(tmp_path, **overrides) -> Settings:
    return Settings(data_dir=tmp_path, log_level="WARNING", _env_file=None, **overrides)


def test_policy_defaults_come_from_settings(tmp_path) -> None:
    policy = OwnerEvaluationPolicy.from_settings(
        _settings(
            tmp_path,
            owner_allowed_datasets=("legal-contracts",),
            owner_eval_max_queries=7,
            owner_eval_max_candidate_depth=40,
            owner_eval_max_top_k=5,
            owner_eval_max_documents=10,
            owner_eval_max_concurrency=2,
            owner_eval_max_runtime_s=12.5,
        )
    )
    assert policy.allowed_datasets == ("legal-contracts",)
    assert (policy.max_queries, policy.max_candidate_depth, policy.max_top_k) == (7, 40, 5)
    assert (policy.max_documents, policy.max_concurrency) == (10, 2)
    assert policy.max_runtime_s == 12.5


def test_policy_summary_is_non_sensitive(tmp_path) -> None:
    policy = OwnerEvaluationPolicy.from_settings(_settings(tmp_path))
    summary = policy.summary()
    assert summary["modes"] == ["bm25"]
    assert summary["rerank"] is False
    assert "token" not in {key.lower() for key in summary}


def test_defaults_are_clamped_to_the_policy_and_bm25_only(tmp_path) -> None:
    fake = _FakeEvaluationService()
    settings = _settings(tmp_path, owner_eval_max_queries=9, owner_eval_max_top_k=4)
    service = OwnerEvaluationService(settings, fake)

    result = service.run(dataset=None)

    assert result["experiment_id"] == "exp-test"
    config = fake.configs[0]
    assert config.mode == "bm25"
    assert config.rerank is False
    assert config.max_queries == 9
    assert config.max_runtime_s == service.policy.max_runtime_s
    assert max(config.k_values) <= config.top_k <= service.policy.max_top_k
    # The trusted fixture was ingested with a bounded document limit, never bulk.
    name, kwargs = fake.ingested[0]
    assert name == "legal-contracts"
    assert kwargs["limit"] == service.policy.max_documents
    assert kwargs["embed"] is False


def test_over_limit_queries_are_rejected(tmp_path) -> None:
    service = OwnerEvaluationService(
        _settings(tmp_path, owner_eval_max_queries=5), _FakeEvaluationService()
    )
    with pytest.raises(OwnerLimitError):
        service.run(max_queries=6)


def test_candidate_depth_and_top_k_bounds(tmp_path) -> None:
    service = OwnerEvaluationService(
        _settings(
            tmp_path,
            owner_eval_max_candidate_depth=20,
            owner_eval_max_top_k=5,
        ),
        _FakeEvaluationService(),
    )
    with pytest.raises(OwnerLimitError):
        service.run(top_k=6)
    with pytest.raises(OwnerLimitError):
        service.run(candidate_depth=21)
    with pytest.raises(OwnerLimitError):
        service.run(top_k=5, candidate_depth=4)


def test_disallowed_dataset_is_rejected(tmp_path) -> None:
    service = OwnerEvaluationService(_settings(tmp_path), _FakeEvaluationService())
    with pytest.raises(OwnerLimitError):
        service.run(dataset="scifact")


def test_concurrency_gate_rejects_a_second_job(tmp_path) -> None:
    settings = _settings(tmp_path, owner_eval_max_concurrency=1)
    fake = _FakeEvaluationService()
    started = threading.Event()
    release = threading.Event()

    def blocking_run(config):
        started.set()
        assert release.wait(timeout=5)
        return {"experiment_id": "exp"}

    fake.run = blocking_run  # type: ignore[method-assign]
    service = OwnerEvaluationService(settings, fake)
    results: list[dict] = []

    worker = threading.Thread(target=lambda: results.append(service.run(max_queries=1)))
    worker.start()
    try:
        assert started.wait(timeout=5)
        with pytest.raises(OwnerLimitError):
            service.run(max_queries=1)
    finally:
        release.set()
        worker.join(timeout=5)

    assert results == [{"experiment_id": "exp"}]
    # The gate releases cleanly, so a later job is admitted again.
    fake.run = lambda config: {"experiment_id": "exp-2"}  # type: ignore[method-assign]
    assert service.run(max_queries=1)["experiment_id"] == "exp-2"
