"""Experiment persistence, export and recorded-artifact loading.

Live runs are persisted to SQLite (``experiments`` + ``experiment_queries``) so
the API, CLI and Streamlit all read the same structured record. Two interchange
formats are supported so results can travel without the database:

* JSON — the complete record, including per-query metrics and traces;
* CSV — one row per query, convenient for spreadsheets and notebooks.

The cloud demo loads a small, committed set of *recorded* artifacts through
:func:`load_recorded`, which are labelled as previously executed runs and never
presented as live inference.
"""

from __future__ import annotations

import csv
import io
import json
import uuid
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import select

from rag_workbench.domain.errors import ExperimentNotFoundError
from rag_workbench.evaluation.runner import ExperimentResult, QueryOutcome
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Experiment, ExperimentQuery

EXPERIMENT_SCHEMA_VERSION = "1"

RECORDED_SOURCE = "recorded"
LIVE_SOURCE = "live"


def _new_experiment_id() -> str:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%S")
    return f"exp-{stamp}-{uuid.uuid4().hex[:6]}"


def _diagnostics_to_json(outcome: QueryOutcome) -> list[dict]:
    return [
        {"code": item.code, "label": item.label, "detail": item.detail}
        for item in outcome.diagnostics
    ]


def _query_to_record(outcome: QueryOutcome) -> dict:
    return {
        "query_id": outcome.query_id,
        "query": outcome.query,
        "num_relevant": outcome.num_relevant,
        "status": outcome.status,
        "candidate_count": outcome.candidate_count,
        "unique_documents": outcome.unique_documents,
        "sufficient_candidates": outcome.sufficient_candidates,
        "metrics": outcome.metrics,
        "candidate_recall": outcome.candidate_recall,
        "timings_ms": outcome.timings_ms,
        "diagnostics": _diagnostics_to_json(outcome),
        "trace": [item.model_dump(mode="json") for item in outcome.trace],
    }


class ExperimentStore:
    """Read and write experiment records against the application database."""

    def __init__(self, database: Database) -> None:
        self._db = database

    # ------------------------------------------------------------------ writes

    def save(self, result: ExperimentResult) -> str:
        experiment_id = result.experiment_id or _new_experiment_id()
        created = datetime.now(tz=UTC)
        dataset = result.dataset
        config = result.config
        row = Experiment(
            id=experiment_id,
            created_at=created,
            dataset_name=str(dataset.get("name", "unknown")),
            dataset_version=str(dataset.get("version", "1")),
            dataset_checksum=dataset.get("corpus_sha256"),
            relevance_mapping=str(config.get("relevance_mapping", "document")),
            mode=str(config.get("mode", "bm25")),
            profile_id=str(config.get("profile_id", "")),
            collection=str(config.get("collection", "")),
            rerank=bool(config.get("rerank", False)),
            candidate_depth=int(config.get("candidate_depth", 0)),
            top_k=int(config.get("top_k", 0)),
            label=config.get("label"),
            subset=bool(config.get("subset", False)),
            git_commit=result.git_commit,
            config=config,
            dataset=dataset,
            runtime=result.runtime,
            aggregate=result.aggregate,
            candidate_metrics=result.candidate_metrics,
            counts=result.counts,
            latency=result.latency,
            notes=list(result.notes),
        )
        with self._db.session() as session:
            session.add(row)
            for outcome in result.per_query:
                session.add(
                    ExperimentQuery(
                        experiment_id=experiment_id,
                        query_id=outcome.query_id,
                        query=outcome.query,
                        num_relevant=outcome.num_relevant,
                        status=outcome.status,
                        candidate_count=outcome.candidate_count,
                        unique_documents=outcome.unique_documents,
                        sufficient_candidates=outcome.sufficient_candidates,
                        metrics=outcome.metrics,
                        candidate_recall=outcome.candidate_recall,
                        timings_ms=outcome.timings_ms,
                        diagnostics=_diagnostics_to_json(outcome),
                        trace=[item.model_dump(mode="json") for item in outcome.trace],
                    )
                )
        return experiment_id

    # ------------------------------------------------------------------- reads

    def list_experiments(
        self, *, dataset: str | None = None, limit: int = 100, offset: int = 0
    ) -> list[dict]:
        with self._db.session() as session:
            stmt = select(Experiment).order_by(Experiment.created_at.desc())
            if dataset is not None:
                stmt = stmt.where(Experiment.dataset_name == dataset)
            rows = session.execute(stmt.limit(limit).offset(offset)).scalars().all()
            return [self._summary(row) for row in rows]

    def get(self, experiment_id: str) -> dict:
        with self._db.session() as session:
            row = session.get(Experiment, experiment_id)
            if row is None:
                raise ExperimentNotFoundError(f"experiment not found: {experiment_id}")
            queries = (
                session.execute(
                    select(ExperimentQuery)
                    .where(ExperimentQuery.experiment_id == experiment_id)
                    .order_by(ExperimentQuery.id)
                )
                .scalars()
                .all()
            )
            return self._full_record(row, queries)

    def get_query(self, experiment_id: str, query_id: str) -> dict:
        with self._db.session() as session:
            row = session.get(Experiment, experiment_id)
            if row is None:
                raise ExperimentNotFoundError(f"experiment not found: {experiment_id}")
            query_row = session.execute(
                select(ExperimentQuery).where(
                    ExperimentQuery.experiment_id == experiment_id,
                    ExperimentQuery.query_id == query_id,
                )
            ).scalar_one_or_none()
            if query_row is None:
                raise ExperimentNotFoundError(
                    f"query '{query_id}' not found in experiment '{experiment_id}'"
                )
            record = self._full_record(row, [query_row])
            return record["per_query"][0]

    # ------------------------------------------------------------------ export

    def export_json(self, experiment_id: str) -> dict:
        return self.get(experiment_id)

    def export_csv(self, experiment_id: str) -> str:
        record = self.get(experiment_id)
        return record_to_csv(record)

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _summary(row: Experiment) -> dict:
        return {
            "experiment_id": row.id,
            "source": LIVE_SOURCE,
            "created_at": row.created_at.isoformat(),
            "dataset": row.dataset_name,
            "dataset_version": row.dataset_version,
            "relevance_mapping": row.relevance_mapping,
            "mode": row.mode,
            "profile_id": row.profile_id,
            "collection": row.collection,
            "rerank": row.rerank,
            "candidate_depth": row.candidate_depth,
            "top_k": row.top_k,
            "label": row.label,
            "subset": row.subset,
            "git_commit": row.git_commit,
            "aggregate": dict(row.aggregate or {}),
            "candidate_metrics": dict(row.candidate_metrics or {}),
            "counts": dict(row.counts or {}),
            "latency": dict(row.latency or {}),
            "config": dict(row.config or {}),
            "dataset_manifest": dict(row.dataset or {}),
            "runtime": dict(row.runtime or {}),
            "notes": list(row.notes or []),
        }

    @staticmethod
    def _full_record(row: Experiment, queries: list[ExperimentQuery]) -> dict:
        return {
            "schema_version": EXPERIMENT_SCHEMA_VERSION,
            "experiment_id": row.id,
            "source": LIVE_SOURCE,
            "created_at": row.created_at.isoformat(),
            "git_commit": row.git_commit,
            "dataset": dict(row.dataset or {}),
            "config": dict(row.config or {}),
            "runtime": dict(row.runtime or {}),
            "aggregate": dict(row.aggregate or {}),
            "candidate_metrics": dict(row.candidate_metrics or {}),
            "counts": dict(row.counts or {}),
            "latency": dict(row.latency or {}),
            "notes": list(row.notes or []),
            "per_query": [
                {
                    "query_id": query.query_id,
                    "query": query.query,
                    "num_relevant": query.num_relevant,
                    "status": query.status,
                    "candidate_count": query.candidate_count,
                    "unique_documents": query.unique_documents,
                    "sufficient_candidates": query.sufficient_candidates,
                    "metrics": dict(query.metrics) if query.metrics else None,
                    "candidate_recall": dict(query.candidate_recall or {}),
                    "timings_ms": dict(query.timings_ms or {}),
                    "diagnostics": list(query.diagnostics or []),
                    "trace": list(query.trace or []),
                }
                for query in queries
            ],
        }


# ------------------------------------------------------------- CSV formatting


def record_to_csv(record: dict) -> str:
    """Render an experiment record's per-query metrics as CSV text."""

    per_query = record.get("per_query", [])
    metric_keys: list[str] = []
    timing_keys: list[str] = []
    for query in per_query:
        for key in (query.get("metrics") or {}):
            if key not in metric_keys:
                metric_keys.append(key)
        for key in (query.get("timings_ms") or {}):
            if key not in timing_keys:
                timing_keys.append(key)
    metric_keys.sort(key=_metric_sort_key)
    timing_keys.sort()

    buffer = io.StringIO()
    writer = csv.writer(buffer)
    header = [
        "experiment_id",
        "query_id",
        "query",
        "num_relevant",
        "status",
        "candidate_count",
        "unique_documents",
        "sufficient_candidates",
        *metric_keys,
        *[f"time_{key}" for key in timing_keys],
        "diagnostics",
    ]
    writer.writerow(header)
    for query in per_query:
        metrics = query.get("metrics") or {}
        timings = query.get("timings_ms") or {}
        row = [
            record.get("experiment_id", ""),
            query.get("query_id", ""),
            query.get("query", ""),
            query.get("num_relevant", 0),
            query.get("status", ""),
            query.get("candidate_count", 0),
            query.get("unique_documents", 0),
            query.get("sufficient_candidates", True),
            *[metrics.get(key, "") for key in metric_keys],
            *[timings.get(key, "") for key in timing_keys],
            ";".join(item.get("code", "") for item in (query.get("diagnostics") or [])),
        ]
        writer.writerow(row)
    return buffer.getvalue()


def _metric_sort_key(name: str) -> tuple[str, int]:
    base, _, depth = name.partition("@")
    try:
        k = int(depth)
    except ValueError:
        k = 0
    return (base, k)


# --------------------------------------------------------- recorded artifacts


def load_recorded(directory: Path) -> list[dict]:
    """Load every ``*.json`` experiment artifact in ``directory``.

    Recorded artifacts are produced by the local runner and committed for the
    read-only cloud demo; they carry ``source="recorded"`` so the UI can label
    them as previously executed experiments.
    """

    if not directory.is_dir():
        return []
    records: list[dict] = []
    for path in sorted(directory.glob("*.json")):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if not isinstance(payload, dict) or "aggregate" not in payload:
            continue
        # Anything in the recorded directory is, by definition, a previously
        # executed run; never present it as a live one.
        payload["source"] = RECORDED_SOURCE
        payload["artifact_path"] = path.name
        records.append(payload)
    records.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
    return records


def find_recorded(directory: Path, experiment_id: str) -> dict | None:
    for record in load_recorded(directory):
        if record.get("experiment_id") == experiment_id:
            return record
    return None


def load_recorded_summary(record: dict) -> dict:
    """Normalise a recorded artifact into the same summary shape as a live row.

    Live rows expose ``dataset``/``mode``/``query_mode``/``profile_id`` etc. as
    top-level fields; a recorded record nests them under ``dataset`` and
    ``config``. Both must look identical to the UI, so the fields are lifted
    here rather than special-cased in every view.
    """

    summary = {key: value for key, value in record.items() if key != "per_query"}
    config = record.get("config", {}) or {}
    dataset = record.get("dataset", {}) or {}
    summary["source"] = RECORDED_SOURCE
    summary["dataset"] = dataset.get("name", summary.get("dataset"))
    summary["dataset_version"] = dataset.get("version")
    summary["dataset_checksum"] = dataset.get("corpus_sha256")
    summary["relevance_mapping"] = config.get("relevance_mapping")
    summary["mode"] = config.get("mode")
    summary["query_mode"] = config.get("query_mode")
    summary["profile_id"] = config.get("profile_id")
    summary["collection"] = config.get("collection")
    summary["rerank"] = config.get("rerank")
    summary["candidate_depth"] = config.get("candidate_depth")
    summary["top_k"] = config.get("top_k")
    summary["label"] = config.get("label")
    summary["subset"] = config.get("subset")
    summary["aggregate"] = record.get("aggregate", {})
    summary["candidate_metrics"] = record.get("candidate_metrics", {})
    summary["counts"] = record.get("counts", {})
    summary["latency"] = record.get("latency", {})
    summary["query_count"] = len(record.get("per_query", []))
    return summary


def compact_record(
    record: dict, *, trace_limit: int = 5, excerpt_chars: int = 240
) -> dict:
    """Return a commit-friendly copy of an experiment record.

    A full record stores the complete ranked trace (with excerpts) for every
    query, which is tens of megabytes for a full benchmark. The committed
    read-only artifact keeps every aggregate, count, latency and per-query
    metric/diagnostic, but caps each query's trace to ``trace_limit`` items and
    truncates excerpts — enough for the Failure Explorer without bloating the
    repository. The truncation is stated explicitly in the notes.
    """

    compact = dict(record)
    per_query: list[dict] = []
    for query in record.get("per_query", []):
        trimmed = dict(query)
        trace = list(query.get("trace", []))
        if query.get("status") == "excluded_no_qrels":
            # Unjudged queries contribute nothing to the metrics or the diagnoses
            # (their only signal is "no judgments"); their traces are dropped to
            # keep the committed artifact small.
            trimmed["trace"] = []
            trimmed["trace_truncated"] = len(trace)
        else:
            trimmed["trace"] = [
                {
                    **item,
                    "excerpt": " ".join(str(item.get("excerpt", "")).split())[:excerpt_chars],
                }
                for item in trace[:trace_limit]
            ]
            trimmed["trace_truncated"] = max(0, len(trace) - trace_limit)
        per_query.append(trimmed)
    compact["per_query"] = per_query
    notes = list(compact.get("notes", []))
    notes.append(
        f"committed artifact: each query's trace is capped at {trace_limit} items; "
        "aggregates and per-query metrics are complete"
    )
    compact["notes"] = notes
    return compact


__all__ = [
    "EXPERIMENT_SCHEMA_VERSION",
    "LIVE_SOURCE",
    "RECORDED_SOURCE",
    "ExperimentStore",
    "compact_record",
    "find_recorded",
    "load_recorded",
    "load_recorded_summary",
    "record_to_csv",
]
