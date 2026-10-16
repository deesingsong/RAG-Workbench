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
from rag_workbench.scaling.harness import (
    RssSampler,
    build_retrieval_queries,
    describe_provenance,
    run_scaling_benchmark,
)


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


def test_describe_provenance_flags_a_dirty_tree() -> None:
    text, limits = describe_provenance("abc123", True)
    assert "abc123" in text
    assert "uncommitted" in text
    assert any("not the exact executable" in note for note in limits)


def test_describe_provenance_accepts_a_clean_tree() -> None:
    text, limits = describe_provenance("abc123", False)
    assert "clean" in text
    assert limits == []


def test_describe_provenance_never_claims_unknown_state_is_clean() -> None:
    text, limits = describe_provenance("abc123", None)
    assert "unknown" in text
    assert limits

    text, limits = describe_provenance(None, None)
    assert "unknown" in text
    assert limits


def test_default_limitations_contain_no_absolute_local_paths() -> None:
    joined = " ".join(harness._default_limitations())
    assert ":\\" not in joined  # no Windows drive paths
    assert "/Users/" not in joined


def test_rss_sampler_reports_sampled_peak_not_a_single_read() -> None:
    sampler = RssSampler(lambda: 0)
    # Readings in bytes, deliberately not monotonic: the peak is not the last one.
    sampler._samples = [mebibytes * 1024 * 1024 for mebibytes in (100, 950, 300, 700, 200)]
    footprint = sampler.footprint(0)

    assert footprint["available"] is True
    assert footprint["sampled_peak_rss_mb"] == 950.0
    assert footprint["rss_mb"] == 200.0  # instantaneous final reading
    assert footprint["tracemalloc_peak_mb"] == 0.0


def test_rss_sampler_without_psutil_does_not_fake_process_memory() -> None:
    sampler = RssSampler(None)
    footprint = sampler.footprint(4 * 1024 * 1024)

    assert footprint["available"] is False
    assert footprint["rss_mb"] is None
    assert footprint["sampled_peak_rss_mb"] is None
    assert footprint["tracemalloc_peak_mb"] == 4.0
    assert "not total process memory" in footprint["note"]
