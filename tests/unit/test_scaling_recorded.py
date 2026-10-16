"""Recorded scalability benchmark loading and validation."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rag_workbench.domain.errors import ScalingError
from rag_workbench.scaling.recorded import (
    find_scaling_benchmark,
    load_scaling_benchmarks,
    scaling_summary,
    validate_scaling_benchmark,
)


def _artifact(benchmark_id: str = "scale-test") -> dict:
    return {
        "schema_version": "2",
        "benchmark_id": benchmark_id,
        "created_at": "2026-10-16T10:00:00+00:00",
        "git_commit": "abc123",
        "git_dirty": False,
        "provenance": "abc123 (clean working tree)",
        "environment": {"python": "3.12.0", "cpu_count": 8},
        "corpus": {"documents": 100, "config": {"seed": 0}, "fingerprint": "deadbeef"},
        "workloads": [
            {
                "documents": 100,
                "chunks": 1000,
                "ingestion": {"documents_per_second": 42.0, "chunks_total": 1000},
                "index": {"db_bytes": 4096, "fts_bytes": 1024},
                "retrieval": [
                    {
                        "mode": "bm25",
                        "candidate_depth": 10,
                        "latency_ms": {"p50": 1.0, "p95": 2.0},
                    }
                ],
            }
        ],
        "limitations": ["single machine"],
    }


def test_validate_accepts_a_well_formed_artifact() -> None:
    model = validate_scaling_benchmark(_artifact())
    assert model.benchmark_id == "scale-test"
    assert model.workloads[0].documents == 100


def test_validate_rejects_missing_benchmark_id() -> None:
    payload = _artifact()
    del payload["benchmark_id"]
    with pytest.raises(ScalingError):
        validate_scaling_benchmark(payload)


def test_validate_rejects_missing_workloads() -> None:
    payload = _artifact()
    payload["workloads"] = []
    with pytest.raises(ScalingError):
        validate_scaling_benchmark(payload)


def test_validate_rejects_wrongly_typed_field() -> None:
    payload = _artifact()
    payload["workloads"][0]["documents"] = "not-a-number"
    with pytest.raises(ScalingError):
        validate_scaling_benchmark(payload)


def test_load_skips_files_that_are_not_scaling_artifacts(tmp_path: Path) -> None:
    (tmp_path / "readme.json").write_text(json.dumps({"unrelated": True}), encoding="utf-8")
    (tmp_path / "scale-good.json").write_text(json.dumps(_artifact("scale-good")), encoding="utf-8")

    records = load_scaling_benchmarks(tmp_path)

    assert [record["benchmark_id"] for record in records] == ["scale-good"]
    assert records[0]["source"] == "recorded"


def test_load_marks_a_malformed_artifact_instead_of_hiding_it(tmp_path: Path) -> None:
    broken = _artifact("scale-broken")
    broken["workloads"][0]["documents"] = "oops"
    (tmp_path / "scale-broken.json").write_text(json.dumps(broken), encoding="utf-8")

    records = load_scaling_benchmarks(tmp_path)

    assert len(records) == 1
    assert records[0]["invalid"] is True
    assert records[0]["invalid_reason"]


def test_find_returns_none_for_unknown_id(tmp_path: Path) -> None:
    (tmp_path / "scale-good.json").write_text(json.dumps(_artifact("scale-good")), encoding="utf-8")

    assert find_scaling_benchmark(tmp_path, "scale-good") is not None
    assert find_scaling_benchmark(tmp_path, "missing") is None


def test_summary_reports_workload_and_modes() -> None:
    record = _artifact()
    record["source"] = "recorded"
    summary = scaling_summary(record)

    assert summary["benchmark_id"] == "scale-test"
    assert summary["workload_count"] == 1
    assert summary["documents_max"] == 100
    assert summary["modes"] == ["bm25"]
    assert summary["corpus_seed"] == 0


def test_summary_exposes_code_revision_provenance() -> None:
    record = _artifact("scale-prov")
    record["git_dirty"] = True
    record["provenance"] = "abc123 + uncommitted changes"

    summary = scaling_summary(record)

    assert summary["git_dirty"] is True
    assert summary["provenance"] == "abc123 + uncommitted changes"


def test_validate_accepts_split_memory_labels() -> None:
    payload = _artifact("scale-mem")
    payload["workloads"][0]["memory"] = {
        "available": True,
        "method": "psutil RSS sampled every 50 ms",
        "note": "sampled peak, not OS high-water mark",
        "rss_mb": 210.5,
        "sampled_peak_rss_mb": 305.2,
        "tracemalloc_peak_mb": 4.1,
    }

    model = validate_scaling_benchmark(payload)

    memory = model.workloads[0].memory
    assert memory is not None
    assert memory.rss_mb == 210.5
    assert memory.sampled_peak_rss_mb == 305.2
    assert memory.tracemalloc_peak_mb == 4.1


def test_learned_schema_version_defaults_to_current() -> None:
    from rag_workbench.scaling.recorded import SCALING_SCHEMA_VERSION, ScalingBenchmark

    model = ScalingBenchmark(benchmark_id="scale-x", workloads=[])
    assert model.schema_version == SCALING_SCHEMA_VERSION == "2"
