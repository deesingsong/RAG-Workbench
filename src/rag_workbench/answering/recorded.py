"""Recorded answer demonstrations for the read-only deployment (Phase 5 §12).

The public Streamlit app cannot run a local model, so it serves *recorded* answer
runs — real executions produced locally, committed to the repo, and clearly
labelled as previously executed rather than live. Two artifact kinds live in the
recorded directory:

* ``answer_demo`` — one full answer record (question, selected evidence, generated
  answer, citations, validation, diagnostics, retrieval/generation timings);
* ``answer_quality`` — one aggregate grounded-answer evaluation report.

Both carry ``source="recorded"`` so nothing can mistake them for live inference.
"""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

DEMO_SCHEMA_VERSION = "1"
RECORDED_SOURCE = "recorded"

_SLUG = re.compile(r"[^a-z0-9]+")


def new_demo_id(question: str, *, created_at: datetime | None = None) -> str:
    """Create a stable, sortable demo id from a question and a timestamp."""

    stamp = (created_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%S")
    digest = hashlib.sha256(question.encode("utf-8")).hexdigest()[:8]
    return f"ans-{stamp}-{digest}"


def slugify(text: str, *, limit: int = 48) -> str:
    slug = _SLUG.sub("-", text.lower()).strip("-")
    return slug[:limit] or "answer"


def write_answer_demo(directory: Path, record: dict) -> Path:
    """Write one answer-demo artifact atomically and return its path."""

    directory.mkdir(parents=True, exist_ok=True)
    demo_id = str(record["demo_id"])
    path = directory / f"{demo_id}-{slugify(str(record.get('question', '')))}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_answer_demos(directory: Path) -> list[dict]:
    """Load every ``answer_demo`` artifact, newest first."""

    if not directory.is_dir():
        return []
    demos: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or payload.get("kind") != "answer_demo":
            continue
        if "question" not in payload or "status" not in payload:
            continue
        payload["source"] = RECORDED_SOURCE
        payload["artifact_path"] = path.name
        demos.append(payload)
    demos.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return demos


def find_answer_demo(directory: Path, demo_id: str) -> dict | None:
    for demo in load_answer_demos(directory):
        if demo.get("demo_id") == demo_id:
            return demo
    return None


def answer_demo_summary(record: dict) -> dict:
    """A compact summary of a demo for list views."""

    validation = record.get("citation_validation", {}) or {}
    context = record.get("context", {}) or {}
    retrieval = record.get("retrieval", {}) or {}
    generation = record.get("generation", {}) or {}
    return {
        "demo_id": record.get("demo_id"),
        "created_at": record.get("created_at"),
        "source": record.get("source", RECORDED_SOURCE),
        "question": record.get("question"),
        "status": record.get("status"),
        "answer_preview": _preview(record.get("answer", "")),
        "claim_count": len(record.get("claims", []) or []),
        "citation_valid": validation.get("citation_valid"),
        "structural_valid": validation.get("structural_valid"),
        "semantic_support": validation.get("semantic_support"),
        "evidence_selected": context.get("selected"),
        "evidence_retrieved": context.get("retrieved"),
        "collection": retrieval.get("collection"),
        "mode": retrieval.get("mode"),
        "model_id": generation.get("model_id"),
        "total_ms": (record.get("timings_ms", {}) or {}).get("total"),
        "generation_ms": (record.get("timings_ms", {}) or {}).get("generation"),
        "diagnostic_codes": sorted(
            {item.get("code", "") for item in record.get("diagnostics", []) or []}
        ),
    }


def _preview(text: str, *, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def write_quality_report(directory: Path, record: dict) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{record['report_id']}.json"
    path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    return path


def load_quality_report(directory: Path) -> dict | None:
    """Return the newest ``answer_quality`` report, or ``None``."""

    if not directory.is_dir():
        return None
    reports: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(payload, dict) and payload.get("kind") == "answer_quality":
            payload["source"] = RECORDED_SOURCE
            payload["artifact_path"] = path.name
            reports.append(payload)
    reports.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return reports[0] if reports else None


__all__ = [
    "DEMO_SCHEMA_VERSION",
    "RECORDED_SOURCE",
    "answer_demo_summary",
    "find_answer_demo",
    "load_answer_demos",
    "load_quality_report",
    "new_demo_id",
    "slugify",
    "write_answer_demo",
    "write_quality_report",
]
