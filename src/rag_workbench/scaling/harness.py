"""Bounded, reproducible local performance harness.

The harness measures the *actual current implementation* — parsing, chunking,
SQLite/FTS5 ingestion, BM25 search, and optionally dense/hybrid/reranking — over
a deterministic synthetic corpus at several sizes. It is deliberately explicit
and opt-in: nothing here runs during application startup or the pytest suite, and
a configurable document cap guards against accidentally exhausting local storage.

Design rules:

* one isolated database per workload, so ingestion is always measured from empty;
* wall-clock timings via ``time.perf_counter`` and per-document stage timings
  taken from the ingestion pipeline itself (parse/chunk/index);
* on-disk measurements are real file sizes plus an FTS5 shadow-table byte count
  (``chunk_fts_data``/``docsize``) taken *after* an FTS ``optimize`` and a WAL
  checkpoint;
* memory is measured with ``psutil`` when it is installed, otherwise a
  ``tracemalloc`` Python-heap peak is recorded and clearly labelled as such;
* the actual execution environment is recorded with the run, never invented.
"""

from __future__ import annotations

import re
import shutil
import tempfile
import time
import tracemalloc
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from sqlalchemy import text

from rag_workbench.config import IndexProfileSpec, Settings
from rag_workbench.domain.errors import ScalingError
from rag_workbench.evaluation.runtime import git_commit, latency_stats, runtime_environment
from rag_workbench.ingestion.optional import dense_feature_available
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.profiles import build_profile
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService
from rag_workbench.scaling.corpus import (
    CorpusConfig,
    SyntheticDocument,
    corpus_fingerprint,
    generate_corpus,
    size_class_counts,
)
from rag_workbench.scaling.recorded import SCALING_SCHEMA_VERSION, new_benchmark_id

_TOKEN_RE = re.compile(r"\btok\d+\b")

ProgressFn = Callable[[str], None]


def _noop(_message: str) -> None:  # pragma: no cover - default sink
    return None


def build_retrieval_queries(documents: Sequence[SyntheticDocument], count: int) -> list[str]:
    """Build a deterministic, reproducible set of retrieval queries.

    Each query is drawn from a sampled document's distinctive tokens plus a
    filler word, so the query has genuine lexical signal against the synthetic
    corpus rather than matching nothing.
    """

    if count < 1 or not documents:
        return []
    span = len(documents)
    step = max(1, span // count)
    filler = ("agreement", "obligation", "payment", "termination", "disclosure")
    queries: list[str] = []
    for offset, index in enumerate(range(0, span, step)):
        if len(queries) >= count:
            break
        tokens = _TOKEN_RE.findall(documents[index].text)
        if not tokens:
            continue
        queries.append(" ".join([*tokens[:2], filler[offset % len(filler)]]))
    return queries


def _memory_probe() -> dict[str, object]:
    try:
        import psutil  # type: ignore[import-not-found]
    except ImportError:
        return {
            "available": False,
            "method": "tracemalloc",
            "note": "psutil not installed; a Python-heap peak is recorded instead of RSS",
        }
    process = psutil.Process()
    return {
        "available": True,
        "method": "psutil RSS",
        "peak_rss_mb": round(process.memory_info().rss / (1024 * 1024), 3),
    }


def _fts_bytes(db: Database) -> int:
    """Best-effort size of the FTS5 inverted index in bytes.

    FTS5 keeps its data in shadow tables derived from the virtual table name.
    Summing the compressed block sizes and per-row sizes is a genuine
    measurement of the index footprint (not the source text), taken after
    ``optimize`` merges segments.
    """

    total = 0
    with db.engine.begin() as connection:
        for table, column in (("chunk_fts_data", "block"), ("chunk_fts_docsize", "sz")):
            try:
                row = connection.execute(
                    text(f"SELECT COALESCE(SUM(LENGTH({column})), 0) FROM {table}")
                ).scalar_one()
                total += int(row)
            except Exception:  # noqa: BLE001 - measurement is best effort
                return 0
    return total


def _db_footprint(db: Database, db_path: Path) -> dict[str, int]:
    try:
        with db.engine.begin() as connection:
            connection.execute(text("PRAGMA wal_checkpoint(TRUNCATE)"))
            page_count = int(connection.execute(text("PRAGMA page_count")).scalar_one())
            page_size = int(connection.execute(text("PRAGMA page_size")).scalar_one())
    except Exception:  # noqa: BLE001
        page_count = page_size = 0
    db_bytes = db_path.stat().st_size if db_path.exists() else 0
    wal_path = db_path.with_name(db_path.name + "-wal")
    return {
        "db_bytes": db_bytes,
        "db_wal_bytes": wal_path.stat().st_size if wal_path.exists() else 0,
        "db_page_bytes": page_count * page_size,
        "fts_bytes": _fts_bytes(db),
    }


def _run_ingestion(
    db: Database,
    settings: Settings,
    profile: IndexProfileSpec,
    documents: Sequence[SyntheticDocument],
    collection: str,
    progress: ProgressFn,
) -> dict[str, object]:
    service = IngestionService(db, settings)
    per_document: list[float] = []
    parse_times: list[float] = []
    chunk_times: list[float] = []
    index_times: list[float] = []
    chunk_total = 0
    failures: list[str] = []
    total = len(documents)
    started = time.perf_counter()
    for index, document in enumerate(documents):
        try:
            result = service.ingest_bytes(
                document.text.encode("utf-8"),
                source_name=f"{document.id}.txt",
                collection=collection,
                profile=profile,
                external_id=document.id,
            )
        except Exception as exc:  # noqa: BLE001 - record and continue
            failures.append(f"{document.id}: {exc}")
            continue
        chunk_total += result.chunk_count
        timings = result.timings_ms
        per_document.append(float(timings.get("total", 0.0)))
        parse_times.append(float(timings.get("parse", 0.0)))
        chunk_times.append(float(timings.get("chunk", 0.0)))
        index_times.append(float(timings.get("index", 0.0)))
        if (index + 1) % max(1, total // 10) == 0:
            progress(f"    ingested {index + 1}/{total} documents")
    total_s = time.perf_counter() - started
    return {
        "total_ms": round(total_s * 1000.0, 3),
        "documents_per_second": round(total / total_s, 3) if total_s > 0 else None,
        "chunks_total": chunk_total,
        "parse_ms": latency_stats(parse_times),
        "chunk_ms": latency_stats(chunk_times),
        "index_ms": latency_stats(index_times),
        "per_document_ms": latency_stats(per_document),
        "failures": failures,
    }


def _run_retrieval(
    settings: Settings,
    db: Database,
    profile: IndexProfileSpec,
    queries: Sequence[str],
    *,
    candidate_depths: Sequence[int],
    query_mode: str,
    modes: Sequence[str],
) -> tuple[list[dict[str, object]], list[str]]:
    service = SearchService(db, settings)
    results: list[dict[str, object]] = []
    failures: list[str] = []
    for mode in modes:
        for depth in candidate_depths:
            samples: list[float] = []
            errors = 0
            for query in queries:
                try:
                    response = service.search(
                        query=query,
                        profile_id=profile.key,
                        mode=mode,
                        top_k=depth,
                        collection=None,
                        candidates=depth,
                        query_mode=query_mode,
                    )
                    samples.append(float(response.total_ms))
                except Exception as exc:  # noqa: BLE001
                    errors += 1
                    if len(failures) < 10:
                        failures.append(f"{mode}@{depth}: {exc}")
            results.append(
                {
                    "mode": mode,
                    "query_mode": query_mode,
                    "candidate_depth": depth,
                    "top_k": depth,
                    "queries": len(queries),
                    "latency_ms": latency_stats(samples),
                    "errors": errors,
                }
            )
    return results, failures


def run_scaling_benchmark(
    settings: Settings,
    *,
    config: CorpusConfig,
    workloads: Sequence[int],
    strategy: str = "paragraph",
    max_tokens: int = 220,
    overlap_tokens: int = 40,
    candidate_depths: Sequence[int] | None = None,
    query_mode: str = "or",
    retrieval_queries: int = 20,
    modes: Sequence[str] | None = None,
    collection: str = "scaling",
    workroot: Path | None = None,
    progress: ProgressFn | None = None,
) -> dict[str, object]:
    """Run the full bounded benchmark and return a record dict.

    ``workloads`` is the list of corpus sizes; it is validated against the
    configured document cap so a mistyped size cannot index a huge corpus.
    """

    report = progress or _noop
    candidate_depths = sorted(candidate_depths or [10, 50, 100])
    modes = list(modes or ["bm25"])
    workloads = list(workloads)
    if not workloads:
        raise ScalingError("at least one workload size is required", code="invalid_workload")
    largest = max(workloads)
    if largest > settings.scaling_max_documents:
        raise ScalingError(
            f"workload size {largest} exceeds the configured cap "
            f"({settings.scaling_max_documents}); raise RAG_WORKBENCH_SCALING_MAX_DOCUMENTS "
            "deliberately if you have the resources",
            code="workload_too_large",
        )
    if any(size < 1 for size in workloads):
        raise ScalingError("workload sizes must be positive", code="invalid_workload")
    if config.documents < largest:
        raise ScalingError(
            "corpus configuration must generate at least the largest workload",
            code="invalid_corpus",
        )

    needs_dense = any(mode in ("dense", "hybrid") for mode in modes)
    if needs_dense and not dense_feature_available():
        raise ScalingError(
            "dense/hybrid modes need the optional 'dense' extra and a local "
            "Sentence Transformers model; install it with "
            "`uv sync --extra dense` before requesting those modes",
            code="dense_unavailable",
        )

    # Generate the corpus once; every workload is a prefix of it, so the
    # fingerprint is comparable across sizes and generation is not repeated.
    corpus = generate_corpus(config.model_copy(update={"documents": largest}))
    queries = build_retrieval_queries(corpus, retrieval_queries)

    workroot_existed = workroot is not None
    workroot = workroot or Path(tempfile.mkdtemp(prefix="rw_scaling_"))
    created_workroot = not workroot_existed
    workroot.mkdir(parents=True, exist_ok=True)
    workload_records: list[dict[str, object]] = []
    run_failures: list[str] = []
    tracemalloc.start()
    try:
        for size in workloads:
            report(f"  workload: {size} documents")
            documents = corpus[:size]
            workload_dir = workroot / f"wl-{size}"
            if workload_dir.exists():
                shutil.rmtree(workload_dir)
            workload_dir.mkdir(parents=True)
            db_path = workload_dir / "index.db"
            run_settings = settings.model_copy(
                update={"data_dir": workload_dir, "database_url": f"sqlite:///{db_path}"}
            )
            db = Database(run_settings.resolved_database_url)
            try:
                db.create_schema()
                profile = build_profile(
                    run_settings,
                    strategy=strategy,
                    max_tokens=max_tokens,
                    overlap_tokens=overlap_tokens,
                    name=f"scaling-{strategy}-{max_tokens}" + ("-dense" if needs_dense else ""),
                    embed=needs_dense,
                )
                ingestion = _run_ingestion(
                    db, run_settings, profile, documents, collection, report
                )
                # Consolidate the FTS index before measuring its footprint.
                with db.engine.begin() as connection:
                    connection.execute(text("INSERT INTO chunk_fts(chunk_fts) VALUES('optimize')"))
                footprint = _db_footprint(db, db_path)
                memory = _memory_probe()
                _current, peak = tracemalloc.get_traced_memory()
                memory["tracemalloc_peak_mb"] = round(peak / (1024 * 1024), 3)
                report(f"    retrieval across {len(queries)} queries, depths {candidate_depths}")
                retrieval, retrieval_failures = _run_retrieval(
                    run_settings,
                    db,
                    profile,
                    queries,
                    candidate_depths=candidate_depths,
                    query_mode=query_mode,
                    modes=modes,
                )
                run_failures.extend(retrieval_failures)
                source_bytes = sum(len(document.text.encode("utf-8")) for document in documents)
                db_failures = list(ingestion.pop("failures", []))
                workload_records.append(
                    {
                        "documents": size,
                        "chunks": ingestion["chunks_total"],
                        "corpus_fingerprint": corpus_fingerprint(documents),
                        "ingestion": ingestion,
                        "index": {
                            "source_bytes": source_bytes,
                            "db_bytes": footprint["db_bytes"],
                            "db_wal_bytes": footprint["db_wal_bytes"],
                            "db_page_bytes": footprint["db_page_bytes"],
                            "fts_bytes": footprint["fts_bytes"],
                            "total_bytes": footprint["db_bytes"]
                            + footprint["db_wal_bytes"],
                        },
                        "retrieval": retrieval,
                        "memory": memory,
                        "failures": [*db_failures, *retrieval_failures],
                    }
                )
            finally:
                db.dispose()
    finally:
        tracemalloc.stop()
        if created_workroot:
            # The harness created this throwaway directory; remove it so repeated
            # runs do not accumulate multi-hundred-MB working databases.
            shutil.rmtree(workroot, ignore_errors=True)

    return {
        "schema_version": SCALING_SCHEMA_VERSION,
        "benchmark_id": new_benchmark_id(),
        "created_at": datetime.now(tz=UTC).isoformat(),
        "git_commit": git_commit(),
        "source": "recorded",
        "environment": runtime_environment(),
        "hardware": {
            key: runtime_environment().get(key)
            for key in ("cpu_count", "processor", "machine", "platform")
        },
        "corpus": {
            "config": config.as_fingerprint_fields(),
            "documents": largest,
            "fingerprint": corpus_fingerprint(corpus),
            "size_classes": size_class_counts(corpus),
        },
        "config": {
            "strategy": strategy,
            "max_tokens": max_tokens,
            "overlap_tokens": overlap_tokens,
            "candidate_depths": list(candidate_depths),
            "query_mode": query_mode,
            "modes": modes,
            "retrieval_queries": len(queries),
        },
        "workloads": workload_records,
        "failures": run_failures,
        "limitations": _default_limitations(workroot),
        "notes": [
            "Synthetic corpus; not representative of any specific production corpus.",
            "Ingestion is synchronous and single-writer; timings reflect this prototype.",
            "Latency scales with local CPU and is not extrapolated to other hardware.",
        ],
    }


def _default_limitations(workroot: Path) -> list[str]:
    return [
        "A single local machine measured these numbers; they are not a production "
        "throughput claim.",
        f"Working databases were written under {workroot} and removed after the run; "
        "only aggregate measurements are kept.",
        "The synthetic corpus is generated, so absolute retrieval quality is not "
        "meaningful here; the measurements characterise cost, not relevance.",
        "Dense/reranker stages are only measured when their optional extras and "
        "model weights are installed.",
    ]


__all__ = [
    "build_retrieval_queries",
    "run_scaling_benchmark",
]
