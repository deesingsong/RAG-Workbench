"""Unit tests for recorded answer-demo artifacts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from rag_workbench.answering.recorded import (
    answer_demo_summary,
    find_answer_demo,
    load_answer_demos,
    load_quality_report,
    new_demo_id,
    slugify,
    write_answer_demo,
    write_quality_report,
)


def _demo(demo_id: str = "ans-20260101T000000-deadbeef") -> dict:
    return {
        "schema_version": "1",
        "kind": "answer_demo",
        "demo_id": demo_id,
        "created_at": "2026-01-01T00:00:00+00:00",
        "source": "recorded",
        "question": "How long is the term?",
        "status": "answered",
        "answer": "Three years [S1].",
        "claims": [{"text": "Three years.", "citations": ["S1"]}],
        "citation_validation": {
            "structural_valid": True,
            "citation_valid": True,
            "semantic_support": "not_determined",
        },
        "context": {"retrieved": 3, "selected": 1},
        "retrieval": {"collection": "legal", "mode": "bm25"},
        "generation": {"model_id": "qwen2.5:1.5b-instruct"},
        "timings_ms": {"total": 12.0, "generation": 8.0},
        "diagnostics": [{"code": "semantic_support_unverified", "detail": "x"}],
    }


def test_write_and_load_round_trip(tmp_path: Path) -> None:
    path = write_answer_demo(tmp_path, _demo())
    assert path.exists()
    demos = load_answer_demos(tmp_path)
    assert len(demos) == 1
    assert demos[0]["demo_id"] == "ans-20260101T000000-deadbeef"
    assert demos[0]["source"] == "recorded"


def test_loader_ignores_non_demo_files(tmp_path: Path) -> None:
    write_answer_demo(tmp_path, _demo())
    (tmp_path / "notes.json").write_text(json.dumps({"hello": "world"}), encoding="utf-8")
    (tmp_path / "broken.json").write_text("{ not json", encoding="utf-8")
    assert len(load_answer_demos(tmp_path)) == 1


def test_summary_shape(tmp_path: Path) -> None:
    write_answer_demo(tmp_path, _demo())
    summary = answer_demo_summary(load_answer_demos(tmp_path)[0])
    assert summary["status"] == "answered"
    assert summary["citation_valid"] is True
    assert summary["claim_count"] == 1
    assert summary["collection"] == "legal"
    assert summary["model_id"] == "qwen2.5:1.5b-instruct"
    assert summary["diagnostic_codes"] == ["semantic_support_unverified"]


def test_find_demo(tmp_path: Path) -> None:
    write_answer_demo(tmp_path, _demo())
    assert find_answer_demo(tmp_path, "ans-20260101T000000-deadbeef") is not None
    assert find_answer_demo(tmp_path, "missing") is None


def test_new_demo_id_is_stable_and_sortable() -> None:
    when = datetime(2026, 1, 1, tzinfo=UTC)
    first = new_demo_id("How long?", created_at=when)
    second = new_demo_id("How long?", created_at=when)
    assert first == second
    assert first.startswith("ans-20260101T000000-")


def test_slugify() -> None:
    assert slugify("How long is the term?") == "how-long-is-the-term"


def test_quality_report_round_trip(tmp_path: Path) -> None:
    report = {"kind": "answer_quality", "report_id": "aq-1", "created_at": "2026-01-01T00:00:00Z"}
    write_quality_report(tmp_path, report)
    assert load_quality_report(tmp_path)["report_id"] == "aq-1"
