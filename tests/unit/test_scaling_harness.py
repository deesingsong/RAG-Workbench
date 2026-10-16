"""Guard rails and helpers of the scalability performance harness.

These tests exercise the cheap, deterministic parts of the harness (input
validation and query construction). They deliberately do **not** run a full
benchmark, which would be slow and write throwaway databases.
"""

from __future__ import annotations

import pytest

from rag_workbench.config import Settings
from rag_workbench.domain.errors import ScalingError
from rag_workbench.scaling import harness
from rag_workbench.scaling.corpus import CorpusConfig, generate_corpus
from rag_workbench.scaling.harness import build_retrieval_queries, run_scaling_benchmark


def test_workload_above_cap_is_rejected_before_running(settings: Settings) -> None:
    capped = settings.model_copy(update={"scaling_max_documents": 50})

    with pytest.raises(ScalingError) as excinfo:
        run_scaling_benchmark(
            capped, config=CorpusConfig(documents=100), workloads=[100]
        )

    assert excinfo.value.code == "workload_too_large"


def test_empty_workloads_rejected(settings: Settings) -> None:
    with pytest.raises(ScalingError):
        run_scaling_benchmark(settings, config=CorpusConfig(documents=10), workloads=[])


def test_corpus_smaller_than_workload_rejected(settings: Settings) -> None:
    with pytest.raises(ScalingError):
        run_scaling_benchmark(settings, config=CorpusConfig(documents=5), workloads=[10])


def test_dense_modes_require_the_optional_extra(
    settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness, "dense_feature_available", lambda: False)

    with pytest.raises(ScalingError) as excinfo:
        run_scaling_benchmark(
            settings,
            config=CorpusConfig(documents=10),
            workloads=[10],
            modes=["dense"],
        )

    assert excinfo.value.code == "dense_unavailable"


def test_build_retrieval_queries_is_deterministic_and_non_empty() -> None:
    documents = generate_corpus(CorpusConfig(documents=50, seed=0))

    first = build_retrieval_queries(documents, 5)
    second = build_retrieval_queries(documents, 5)

    assert first == second
    assert len(first) == 5
    assert all(query.strip() for query in first)


def test_build_retrieval_queries_handles_empty_input() -> None:
    assert build_retrieval_queries([], 5) == []
    assert build_retrieval_queries(generate_corpus(CorpusConfig(documents=1)), 0) == []
