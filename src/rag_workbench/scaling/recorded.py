"""Load, validate and summarise recorded scalability benchmark artifacts.

The scaling harness runs real, bounded workloads locally and writes a single JSON
artifact per run. A small, curated set is committed under
``benchmarks/recorded_scaling/`` so the read-only Streamlit deployment can show
genuine measurements without running anything. Like the Phase-4 experiment
artifacts, every record is labelled ``source="recorded"`` and its environment is
part of the record, so a displayed number can never be mistaken for a live,
cloud-side measurement.

Validation is intentionally strict enough to catch a truncated or mis-typed
artifact (a missing workload, a non-numeric timing) while tolerating unknown
future fields.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field
from pydantic import ValidationError as PydanticValidationError

from rag_workbench.domain.errors import ScalingError

#: Bumped when the meaning of a scaling record's fields changes. Version 2 adds
#: explicit code-revision provenance (``git_dirty``/``provenance``) and splits the
#: former single memory figure into instantaneous RSS, sampled-peak RSS and a
#: Python-managed allocation peak.
SCALING_SCHEMA_VERSION = "2"

SCALING_SOURCE = "recorded"


class IngestStageStats(BaseModel):
    model_config = ConfigDict(extra="ignore")

    total_ms: float | None = None
    documents_per_second: float | None = None
    chunks_total: int | None = None
    parse_ms: dict[str, Any] | None = None
    chunk_ms: dict[str, Any] | None = None
    index_ms: dict[str, Any] | None = None
    per_document_ms: dict[str, Any] | None = None


class IndexFootprint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    source_bytes: int | None = None
    db_bytes: int | None = None
    db_wal_bytes: int | None = None
    fts_bytes: int | None = None
    vector_bytes: int | None = None
    total_bytes: int | None = None


class RetrievalLatency(BaseModel):
    model_config = ConfigDict(extra="ignore")

    mode: str = "bm25"
    query_mode: str = "or"
    candidate_depth: int | None = None
    top_k: int | None = None
    queries: int | None = None
    latency_ms: dict[str, Any] = Field(default_factory=dict)
    stages: dict[str, Any] = Field(default_factory=dict)
    errors: int = 0


class MemoryFootprint(BaseModel):
    model_config = ConfigDict(extra="ignore")

    available: bool = False
    method: str = ""
    note: str = ""
    #: One instantaneous process-RSS reading at the end of the workload (psutil).
    rss_mb: float | None = None
    #: Maximum of periodic process-RSS samples during the workload (psutil).
    sampled_peak_rss_mb: float | None = None
    #: Peak of Python-managed allocations (tracemalloc); *not* total process memory.
    tracemalloc_peak_mb: float | None = None
    #: Legacy field from schema version 1; never written by the current harness.
    peak_rss_mb: float | None = None


class ScalingWorkload(BaseModel):
    model_config = ConfigDict(extra="ignore")

    documents: int
    chunks: int | None = None
    corpus_fingerprint: str | None = None
    ingestion: IngestStageStats | None = None
    index: IndexFootprint | None = None
    retrieval: list[RetrievalLatency] = Field(default_factory=list)
    memory: MemoryFootprint | None = None
    failures: list[str] = Field(default_factory=list)


class ScalingCorpusRecord(BaseModel):
    model_config = ConfigDict(extra="ignore")

    config: dict[str, Any] = Field(default_factory=dict)
    documents: int | None = None
    fingerprint: str | None = None
    size_classes: dict[str, int] | None = None


class ScalingBenchmark(BaseModel):
    """A validated scaling benchmark artifact."""

    model_config = ConfigDict(extra="ignore")

    schema_version: str = SCALING_SCHEMA_VERSION
    benchmark_id: str
    created_at: str | None = None
    git_commit: str | None = None
    #: True/False when the working tree's state was checked; None when unknown.
    git_dirty: bool | None = None
    #: Human-readable provenance of the code revision (commit + tree state).
    provenance: str | None = None
    environment: dict[str, Any] = Field(default_factory=dict)
    corpus: ScalingCorpusRecord = Field(default_factory=ScalingCorpusRecord)
    workloads: list[ScalingWorkload] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    source: str = SCALING_SOURCE


def validate_scaling_benchmark(payload: dict[str, Any]) -> ScalingBenchmark:
    """Validate one artifact, raising :class:`ScalingError` with a clear code."""

    if not isinstance(payload, dict):
        raise ScalingError(
            "scaling benchmark must be a JSON object", code="invalid_scaling_artifact"
        )
    if not payload.get("benchmark_id"):
        raise ScalingError(
            "scaling benchmark is missing 'benchmark_id'", code="missing_benchmark_id"
        )
    if not payload.get("workloads"):
        raise ScalingError("scaling benchmark has no workloads", code="missing_workloads")
    try:
        return ScalingBenchmark.model_validate(payload)
    except PydanticValidationError as exc:
        details = "; ".join(
            error["msg"] for error in exc.errors(include_url=False, include_input=False)
        )
        raise ScalingError(
            f"malformed scaling benchmark: {details or 'validation failed'}",
            code="malformed_scaling_artifact",
        ) from exc


def _looks_like_scaling_artifact(payload: Any) -> bool:
    return isinstance(payload, dict) and "workloads" in payload


def load_scaling_benchmarks(directory: Path) -> list[dict[str, Any]]:
    """Load every valid ``*.json`` scaling artifact in ``directory``.

    Files that are not scaling artifacts (for example a stray manifest) are
    skipped; a file that *is* an artifact by shape but fails validation is
    included with an ``invalid`` marker rather than crashing the reader, so a
    broken committed artifact is visible instead of silently disappearing.
    """

    if not directory.is_dir():
        return []
    records: list[dict[str, Any]] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not _looks_like_scaling_artifact(payload):
            continue
        payload["source"] = SCALING_SOURCE
        payload["artifact_path"] = path.name
        try:
            validate_scaling_benchmark(payload)
        except ScalingError as exc:
            payload["invalid"] = True
            payload["invalid_reason"] = exc.message
        records.append(payload)
    records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return records


def find_scaling_benchmark(directory: Path, benchmark_id: str) -> dict[str, Any] | None:
    for record in load_scaling_benchmarks(directory):
        if record.get("benchmark_id") == benchmark_id:
            return record
    return None


def scaling_summary(record: dict[str, Any]) -> dict[str, Any]:
    """Normalise a record into the compact shape the list endpoint/UI expects."""

    workloads = record.get("workloads") or []
    env = record.get("environment") or {}
    corpus_config = (record.get("corpus") or {}).get("config") or {}
    largest = max((int(w.get("documents", 0)) for w in workloads), default=0)
    total_chunks = sum(int(w.get("chunks", 0) or 0) for w in workloads)
    return {
        "benchmark_id": record.get("benchmark_id"),
        "source": record.get("source", SCALING_SOURCE),
        "created_at": record.get("created_at"),
        "git_commit": record.get("git_commit"),
        "git_dirty": record.get("git_dirty"),
        "provenance": record.get("provenance"),
        "invalid": bool(record.get("invalid")),
        "invalid_reason": record.get("invalid_reason"),
        "schema_version": record.get("schema_version"),
        "workload_count": len(workloads),
        "documents_max": largest,
        "chunks_total": total_chunks,
        "corpus_seed": corpus_config.get("seed"),
        "python": env.get("python"),
        "platform": env.get("platform"),
        "cpu_count": env.get("cpu_count"),
        "processor": env.get("processor"),
        "modes": sorted(
            {
                latency.get("mode")
                for workload in workloads
                for latency in (workload.get("retrieval") or [])
                if latency.get("mode")
            }
        ),
        "notes": list(record.get("notes") or []),
        "limitations": list(record.get("limitations") or []),
    }


def new_benchmark_id() -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    return f"scale-{stamp}"


__all__ = [
    "SCALING_SCHEMA_VERSION",
    "SCALING_SOURCE",
    "ScalingBenchmark",
    "ScalingCorpusRecord",
    "ScalingWorkload",
    "find_scaling_benchmark",
    "load_scaling_benchmarks",
    "new_benchmark_id",
    "scaling_summary",
    "validate_scaling_benchmark",
]
