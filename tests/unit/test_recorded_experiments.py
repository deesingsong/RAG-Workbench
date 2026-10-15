"""Unit tests for recorded experiment artifacts (commit-friendly compaction)."""

from __future__ import annotations

import json
from pathlib import Path

from rag_workbench.evaluation.experiments import (
    compact_record,
    load_recorded,
    load_recorded_summary,
)


def _record() -> dict:
    return {
        "schema_version": 1,
        "experiment_id": "exp-test",
        "source": "live",
        "created_at": "2026-01-01T00:00:00+00:00",
        "dataset": {"name": "demo", "version": "1", "corpus_sha256": "abc"},
        "config": {
            "mode": "bm25",
            "query_mode": "or",
            "profile_id": "p-1",
            "collection": "demo",
            "rerank": False,
            "candidate_depth": 20,
            "top_k": 10,
            "label": "demo run",
        },
        "aggregate": {"recall@10": 0.5},
        "candidate_metrics": {"candidate_recall@20": 0.7},
        "counts": {"queries_evaluated": 1},
        "latency": {"total_ms": {"p50": 1.0}},
        "notes": [],
        "per_query": [
            {
                "query_id": "Q1",
                "query": "alpha",
                "status": "ok",
                "trace": [
                    {"rank": i, "excerpt": "x" * 1000, "key": f"D{i}"} for i in range(1, 21)
                ],
            },
            {
                "query_id": "Q2",
                "query": "beta",
                "status": "excluded_no_qrels",
                "trace": [{"rank": 1, "excerpt": "y" * 500, "key": "D1"}],
            },
        ],
    }


def test_compact_record_caps_evaluated_traces_and_drops_excluded() -> None:
    compact = compact_record(_record(), trace_limit=5, excerpt_chars=240)
    q1, q2 = compact["per_query"]
    assert len(q1["trace"]) == 5
    assert q1["trace_truncated"] == 15
    assert all(len(item["excerpt"]) == 240 for item in q1["trace"])
    # Unjudged queries contribute nothing; their traces are dropped entirely.
    assert q2["trace"] == []
    assert q2["trace_truncated"] == 1
    # Aggregates are untouched.
    assert compact["aggregate"] == {"recall@10": 0.5}
    assert any("committed artifact" in note for note in compact["notes"])


def test_load_recorded_summary_lifts_config_fields() -> None:
    record = _record()
    record["source"] = "recorded"
    summary = load_recorded_summary(record)
    assert "per_query" not in summary
    assert summary["source"] == "recorded"
    assert summary["dataset"] == "demo"
    assert summary["mode"] == "bm25"
    assert summary["query_mode"] == "or"
    assert summary["profile_id"] == "p-1"
    assert summary["label"] == "demo run"
    assert summary["query_count"] == 2


def test_load_recorded_reads_artifacts_and_labels_source(tmp_path: Path) -> None:
    artifact = tmp_path / "exp-test.json"
    artifact.write_text(json.dumps(_record()), encoding="utf-8")
    # A file without an aggregate is ignored rather than raising.
    (tmp_path / "not-an-experiment.json").write_text('{"hello": 1}', encoding="utf-8")

    records = load_recorded(tmp_path)
    assert len(records) == 1
    assert records[0]["experiment_id"] == "exp-test"
    assert records[0]["source"] == "recorded"
    assert records[0]["artifact_path"] == "exp-test.json"
