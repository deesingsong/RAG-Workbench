"""Command-line interface.

    rag-workbench init-db
    rag-workbench ingest path/to/file.pdf --collection contracts
    rag-workbench ingest path/to/file.pdf --embed          # dense profile
    rag-workbench search "termination for convenience" --top-k 5
    rag-workbench search "termination" --mode hybrid       # BM25 + dense + RRF
    rag-workbench search "termination" --rerank            # + cross-encoder rerank
    rag-workbench ingest contract.txt --strategy clause_aware
    rag-workbench chunks --profile <id>
    rag-workbench documents
    rag-workbench profiles
    rag-workbench rebuild-vectors --all
    rag-workbench vectors-status
    rag-workbench datasets list
    rag-workbench datasets download-scifact
    rag-workbench datasets import --name scifact --corpus DIR/corpus.jsonl ...
    rag-workbench datasets ingest scifact --strategy paragraph
    rag-workbench eval run --dataset scifact --profile <id> --mode bm25
    rag-workbench eval run --dataset scifact --profile <id> --rerank
    rag-workbench eval list
    rag-workbench eval show <experiment-id>
    rag-workbench eval export <experiment-id> --format csv --out run.csv
    rag-workbench eval failures <experiment-id>
    rag-workbench scaling run --documents 100,1000,5000 --record
    rag-workbench scaling capacity --documents 10000000 --scenario typical
    rag-workbench scaling load-test --url http://127.0.0.1:8765 --workload mixed
    rag-workbench serve

The CLI, the API and the tests all call the same core services; the CLI adds no
business logic of its own.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import UTC, datetime
from pathlib import Path

from rag_workbench.answering.pipeline import AnswerService
from rag_workbench.answering.quality import (
    AnswerQualityRunner,
    load_answer_quality_dataset,
)
from rag_workbench.answering.recorded import (
    DEMO_SCHEMA_VERSION,
    answer_demo_summary,
    find_answer_demo,
    load_answer_demos,
    new_demo_id,
    write_answer_demo,
    write_quality_report,
)
from rag_workbench.config import IndexProfileSpec, Settings, find_repo_root, get_settings
from rag_workbench.domain.errors import (
    AnswerNotFoundError,
    GeneratorUnavailableError,
    RagWorkbenchError,
)
from rag_workbench.domain.schemas import AnswerResponse
from rag_workbench.evaluation.experiments import compact_record
from rag_workbench.evaluation.runner import EvalRunConfig
from rag_workbench.evaluation.runtime import git_commit, runtime_environment
from rag_workbench.evaluation.service import EvaluationService
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.profiles import build_profile
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.logging_config import configure_logging
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.vector_store import close_vector_stores
from rag_workbench.retrieval.service import SearchService
from rag_workbench.scaling.capacity import (
    IngestionAssumptions,
    StorageAssumptions,
    estimate_ingestion,
    estimate_storage,
    human_bytes,
    storage_scenarios,
)
from rag_workbench.scaling.corpus import CorpusConfig, generate_corpus, write_corpus
from rag_workbench.scaling.harness import run_scaling_benchmark
from rag_workbench.scaling.loadtest import run_load_test_sync
from rag_workbench.scaling.service import ScalingService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-workbench",
        description="Local-first RAG retrieval laboratory (ingestion, BM25, dense, hybrid).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the SQLite schema and FTS5 index")

    ingest = sub.add_parser("ingest", help="ingest a TXT/MD/PDF document")
    ingest.add_argument("path", type=Path, help="path to the document")
    ingest.add_argument("--collection", default=None, help="collection (dataset) name")
    ingest.add_argument(
        "--strategy",
        choices=["paragraph", "fixed_token", "clause_aware"],
        default=None,
        help="chunking strategy",
    )
    ingest.add_argument("--max-tokens", type=int, default=None)
    ingest.add_argument("--overlap-tokens", type=int, default=None)
    ingest.add_argument("--name", default=None, help="human label for the index profile")
    ingest.add_argument(
        "--embed", action="store_true", help="create a dense profile and index embeddings"
    )
    ingest.add_argument(
        "--embedding-model", default=None, help="Sentence Transformers model id (with --embed)"
    )
    ingest.add_argument(
        "--embedding-dimension",
        type=int,
        default=None,
        help="explicit vector dimension (with --embed)",
    )
    ingest.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    search = sub.add_parser("search", help="run a BM25, dense or hybrid search")
    search.add_argument("query", nargs="+", help="query text")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--profile", default=None, help="index profile ID")
    search.add_argument("--collection", default=None)
    search.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid"], default="bm25", help="retrieval mode"
    )
    search.add_argument(
        "--candidates",
        type=int,
        default=None,
        help="candidate pool for dense/hybrid",
    )
    search.add_argument(
        "--query-mode",
        choices=["and", "or"],
        default="and",
        help="lexical term combination for BM25/hybrid (default: and)",
    )
    search.add_argument(
        "--rerank", action="store_true", help="rerank candidates with a cross-encoder"
    )
    search.add_argument(
        "--reranker-model",
        default=None,
        help="cross-encoder model id (default: cross-encoder/ms-marco-MiniLM-L6-v2)",
    )
    search.add_argument(
        "--rerank-candidates",
        type=int,
        default=None,
        help="candidate depth to rerank (default 20; must be >= --top-k)",
    )
    search.add_argument("--json", action="store_true")

    chunks = sub.add_parser("chunks", help="list stored chunks and their provenance")
    chunks.add_argument("--profile", default=None, help="index profile ID (default: newest)")
    chunks.add_argument("--document", default=None, help="restrict to one document ID")
    chunks.add_argument("--limit", type=int, default=200)
    chunks.add_argument("--json", action="store_true")

    documents = sub.add_parser("documents", help="list indexed documents")
    documents.add_argument("--collection", default=None)
    documents.add_argument("--limit", type=int, default=50)
    documents.add_argument("--json", action="store_true")

    profiles = sub.add_parser("profiles", help="list index profiles")
    profiles.add_argument("--json", action="store_true")

    rebuild = sub.add_parser("rebuild-vectors", help="rebuild dense vector indexes from SQLite")
    rebuild.add_argument("--profile", default=None, help="index profile ID")
    rebuild.add_argument(
        "--all", dest="all_profiles", action="store_true", help="every dense profile"
    )
    rebuild.add_argument(
        "--keep", action="store_true", help="do not drop the collection first (repair in place)"
    )
    rebuild.add_argument("--json", action="store_true")

    status = sub.add_parser("vectors-status", help="compare vector points with SQLite chunks")
    status.add_argument("--profile", default=None, help="index profile ID")
    status.add_argument("--repair", action="store_true", help="repair missing/orphan points")
    status.add_argument("--json", action="store_true")

    serve = sub.add_parser("serve", help="run the FastAPI app with uvicorn")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")

    # ------------------------------------------------------------- datasets
    datasets = sub.add_parser("datasets", help="list, import and ingest evaluation datasets")
    datasets_sub = datasets.add_subparsers(dest="datasets_command", required=True)

    datasets_list = datasets_sub.add_parser("list", help="list available datasets")
    datasets_list.add_argument("--json", action="store_true")

    datasets_import = datasets_sub.add_parser(
        "import", help="normalise local BEIR-format files into a dataset"
    )
    datasets_import.add_argument("--name", required=True)
    datasets_import.add_argument("--corpus", type=Path, required=True)
    datasets_import.add_argument("--queries", type=Path, required=True)
    datasets_import.add_argument("--qrels", type=Path, required=True)
    datasets_import.add_argument("--evidence", type=Path, default=None)
    datasets_import.add_argument("--source", required=True)
    datasets_import.add_argument("--license", required=True)
    datasets_import.add_argument("--version", default="1")
    datasets_import.add_argument("--description", default="")
    datasets_import.add_argument(
        "--mapping", choices=["document", "evidence"], default="document"
    )
    datasets_import.add_argument("--json", action="store_true")

    datasets_download = datasets_sub.add_parser(
        "download-scifact", help="download and import the BEIR SciFact archive (explicit)"
    )
    datasets_download.add_argument("--name", default="scifact")
    datasets_download.add_argument("--version", default="beir")
    datasets_download.add_argument("--json", action="store_true")

    datasets_ingest = datasets_sub.add_parser(
        "ingest", help="ingest a dataset's corpus into an index profile"
    )
    datasets_ingest.add_argument("dataset")
    datasets_ingest.add_argument("--collection", default=None)
    datasets_ingest.add_argument(
        "--strategy", choices=["paragraph", "fixed_token", "clause_aware"], default=None
    )
    datasets_ingest.add_argument("--max-tokens", type=int, default=None)
    datasets_ingest.add_argument("--overlap-tokens", type=int, default=None)
    datasets_ingest.add_argument("--name", default=None, help="index profile label")
    datasets_ingest.add_argument("--embed", action="store_true")
    datasets_ingest.add_argument("--embedding-model", default=None)
    datasets_ingest.add_argument("--embedding-dimension", type=int, default=None)
    datasets_ingest.add_argument("--limit", type=int, default=None)
    datasets_ingest.add_argument("--json", action="store_true")

    # -------------------------------------------------------------- evaluation
    evaluation = sub.add_parser("eval", help="run and inspect retrieval experiments")
    eval_sub = evaluation.add_subparsers(dest="eval_command", required=True)

    eval_run = eval_sub.add_parser("run", help="run an evaluation configuration")
    eval_run.add_argument("--dataset", required=True)
    eval_run.add_argument("--profile", required=True, help="index profile ID")
    eval_run.add_argument("--collection", default=None)
    eval_run.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    eval_run.add_argument("--query-mode", choices=["and", "or"], default="or")
    eval_run.add_argument("--candidate-depth", type=int, default=None)
    eval_run.add_argument("--top-k", type=int, default=10)
    eval_run.add_argument("--k", default="1,3,5,10", help="comma-separated cut-offs")
    eval_run.add_argument("--rerank", action="store_true")
    eval_run.add_argument("--reranker-model", default=None)
    eval_run.add_argument("--rerank-candidates", type=int, default=None)
    eval_run.add_argument("--candidates", type=int, default=None)
    eval_run.add_argument("--max-queries", type=int, default=None)
    eval_run.add_argument("--label", default=None)
    eval_run.add_argument("--subset", action="store_true")
    eval_run.add_argument("--json", action="store_true")
    eval_run.add_argument("--out", type=Path, default=None, help="also write the JSON record here")

    eval_list = eval_sub.add_parser("list", help="list live and recorded experiments")
    eval_list.add_argument("--dataset", default=None)
    eval_list.add_argument("--json", action="store_true")

    eval_show = eval_sub.add_parser("show", help="show one experiment record")
    eval_show.add_argument("experiment_id")
    eval_show.add_argument("--json", action="store_true")

    eval_export = eval_sub.add_parser("export", help="export an experiment as JSON or CSV")
    eval_export.add_argument("experiment_id")
    eval_export.add_argument("--format", choices=["json", "csv"], default="json")
    eval_export.add_argument("--out", type=Path, default=None)

    eval_failures = eval_sub.add_parser(
        "failures", help="list queries with observed failure diagnoses"
    )
    eval_failures.add_argument("experiment_id")
    eval_failures.add_argument("--limit", type=int, default=50)
    eval_failures.add_argument("--json", action="store_true")

    eval_record = eval_sub.add_parser(
        "record", help="write an experiment as a committed read-only artifact"
    )
    eval_record.add_argument("experiment_id")
    eval_record.add_argument("--name", default=None, help="artifact filename")
    eval_record.add_argument("--directory", type=Path, default=None)
    eval_record.add_argument(
        "--full",
        action="store_true",
        help="write the complete record (large); default trims per-query traces",
    )

    # ---------------------------------------------------------------- answering
    answer = sub.add_parser(
        "answer", help="optional grounded answer generation, demos and quality"
    )
    answer_sub = answer.add_subparsers(dest="answer_command", required=True)

    answer_ask = answer_sub.add_parser("ask", help="generate a grounded answer locally")
    answer_ask.add_argument("question")
    answer_ask.add_argument("--profile", default=None, help="index profile ID")
    answer_ask.add_argument("--collection", default=None)
    answer_ask.add_argument("--mode", choices=["bm25", "dense", "hybrid"], default="bm25")
    answer_ask.add_argument("--query-mode", choices=["and", "or"], default="and")
    answer_ask.add_argument("--candidate-depth", type=int, default=None)
    answer_ask.add_argument("--candidates", type=int, default=None)
    answer_ask.add_argument("--rerank", action="store_true")
    answer_ask.add_argument("--reranker-model", default=None)
    answer_ask.add_argument("--rerank-candidates", type=int, default=None)
    answer_ask.add_argument("--max-evidence", type=int, default=None)
    answer_ask.add_argument("--context-tokens", type=int, default=None)
    answer_ask.add_argument("--temperature", type=float, default=None)
    answer_ask.add_argument("--num-predict", type=int, default=None)
    answer_ask.add_argument("--seed", type=int, default=None)
    answer_ask.add_argument("--json", action="store_true")

    answer_demos = answer_sub.add_parser("demos", help="list recorded answer demonstrations")
    answer_demos.add_argument("--json", action="store_true")

    answer_demo = answer_sub.add_parser("demo", help="show one recorded answer demonstration")
    answer_demo.add_argument("demo_id")

    answer_quality = answer_sub.add_parser(
        "quality", help="run the grounded-answer quality set"
    )
    answer_quality.add_argument(
        "--dataset", default="answer-quality/legal-contracts-answers"
    )
    answer_quality.add_argument("--profile", required=True)
    answer_quality.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid"], default="bm25"
    )
    answer_quality.add_argument("--query-mode", choices=["and", "or"], default="and")
    answer_quality.add_argument("--candidate-depth", type=int, default=None)
    answer_quality.add_argument("--rerank", action="store_true")
    answer_quality.add_argument("--rerank-candidates", type=int, default=None)
    answer_quality.add_argument("--max-evidence", type=int, default=None)
    answer_quality.add_argument("--context-tokens", type=int, default=None)
    answer_quality.add_argument("--seed", type=int, default=None)
    answer_quality.add_argument("--label", default=None)
    answer_quality.add_argument(
        "--recommended",
        action="store_true",
        help="mark this report as the recommended configuration for the demo",
    )
    answer_quality.add_argument("--out", type=Path, default=None)

    answer_record = answer_sub.add_parser(
        "record", help="record answer demos and a quality report as artifacts"
    )
    answer_record.add_argument(
        "--dataset", default="answer-quality/legal-contracts-answers"
    )
    answer_record.add_argument("--profile", required=True)
    answer_record.add_argument(
        "--mode", choices=["bm25", "dense", "hybrid"], default="bm25"
    )
    answer_record.add_argument("--query-mode", choices=["and", "or"], default="and")
    answer_record.add_argument("--candidate-depth", type=int, default=None)
    answer_record.add_argument("--rerank", action="store_true")
    answer_record.add_argument("--rerank-candidates", type=int, default=None)
    answer_record.add_argument("--max-evidence", type=int, default=None)
    answer_record.add_argument("--context-tokens", type=int, default=None)
    answer_record.add_argument("--seed", type=int, default=None)
    answer_record.add_argument("--label", default=None)
    answer_record.add_argument(
        "--recommended",
        action="store_true",
        help="mark this report as the recommended configuration for the demo",
    )
    answer_record.add_argument("--directory", type=Path, default=None)
    answer_record.add_argument(
        "--no-demos", action="store_true", help="write only the aggregate quality report"
    )

    # ---------------------------------------------------------------- scalability
    scaling = sub.add_parser(
        "scaling", help="performance baselines, capacity planning and load testing"
    )
    scaling_sub = scaling.add_subparsers(dest="scaling_command", required=True)

    scaling_corpus = scaling_sub.add_parser(
        "corpus", help="generate a reproducible synthetic corpus to a directory"
    )
    scaling_corpus.add_argument("--documents", type=int, required=True)
    scaling_corpus.add_argument("--out", type=Path, required=True)
    scaling_corpus.add_argument("--seed", type=int, default=0)
    scaling_corpus.add_argument("--paragraphs-per-section", type=int, default=2)
    scaling_corpus.add_argument("--words-per-paragraph", type=int, default=55)

    scaling_capacity = scaling_sub.add_parser(
        "capacity", help="estimate storage and ingestion capacity from assumptions"
    )
    scaling_capacity.add_argument("--documents", type=int, default=10_000_000)
    scaling_capacity.add_argument("--chunks-per-document", type=float, default=10.0)
    scaling_capacity.add_argument("--embedding-dimensions", type=int, default=384)
    scaling_capacity.add_argument("--bytes-per-dimension", type=int, default=4)
    scaling_capacity.add_argument("--vector-indexes", type=int, default=1)
    scaling_capacity.add_argument("--replication-factor", type=int, default=2)
    scaling_capacity.add_argument("--documents-per-day", type=int, default=None)
    scaling_capacity.add_argument(
        "--embedding-chunks-per-second-per-worker", type=float, default=None
    )
    scaling_capacity.add_argument("--workers", type=int, default=4)
    scaling_capacity.add_argument("--scenario", choices=["small", "typical", "large"], default=None)
    scaling_capacity.add_argument("--json", action="store_true")

    scaling_run = scaling_sub.add_parser(
        "run", help="run bounded local indexing and retrieval benchmarks"
    )
    scaling_run.add_argument(
        "--documents", default="100,1000,5000", help="comma-separated corpus sizes"
    )
    scaling_run.add_argument(
        "--strategy", choices=["paragraph", "fixed_token", "clause_aware"], default="paragraph"
    )
    scaling_run.add_argument("--max-tokens", type=int, default=220)
    scaling_run.add_argument("--overlap-tokens", type=int, default=40)
    scaling_run.add_argument("--seed", type=int, default=0)
    scaling_run.add_argument(
        "--depths", default="10,50,100", help="comma-separated candidate depths"
    )
    scaling_run.add_argument("--queries", type=int, default=20)
    scaling_run.add_argument("--modes", default="bm25", help="comma-separated retrieval modes")
    scaling_run.add_argument("--out", type=Path, default=None, help="write the record JSON here")
    scaling_run.add_argument(
        "--record", action="store_true", help="write the record to benchmarks/recorded_scaling"
    )
    scaling_run.add_argument("--name", default=None, help="artifact filename when --record")
    scaling_run.add_argument("--json", action="store_true")

    scaling_list = scaling_sub.add_parser("list", help="list recorded scaling benchmarks")
    scaling_list.add_argument("--json", action="store_true")

    scaling_show = scaling_sub.add_parser("show", help="show one recorded scaling benchmark")
    scaling_show.add_argument("benchmark_id")

    scaling_load = scaling_sub.add_parser(
        "load-test", help="bounded local load test against a running API"
    )
    scaling_load.add_argument("--url", default="http://127.0.0.1:8765")
    scaling_load.add_argument(
        "--workload",
        choices=["health", "bm25_search", "experiments", "recorded_artifacts", "mixed"],
        default="mixed",
    )
    scaling_load.add_argument("--users", type=int, default=4)
    scaling_load.add_argument("--duration", type=float, default=10.0)
    scaling_load.add_argument(
        "--warmup", type=float, default=2.0, help="warm-up seconds excluded from results"
    )
    scaling_load.add_argument("--json", action="store_true")

    return parser


def _profile_from_args(settings: Settings, args: argparse.Namespace) -> IndexProfileSpec:
    return build_profile(
        settings,
        strategy=args.strategy,
        max_tokens=args.max_tokens,
        overlap_tokens=args.overlap_tokens,
        name=args.name,
        embed=getattr(args, "embed", False),
        embedding_model=getattr(args, "embedding_model", None),
        embedding_dimension=getattr(args, "embedding_dimension", None),
    )


def _cmd_ingest(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    service = IngestionService(database, settings)
    profile = _profile_from_args(settings, args)
    result = service.ingest_path(
        args.path, collection=args.collection or settings.default_collection, profile=profile
    )
    if args.json:
        print(result.model_dump_json(indent=2))
    else:
        if result.is_duplicate:
            status = "duplicate (already indexed)"
        else:
            status = "ingested" if result.state == "completed" else result.state
        print(f"{status}: {result.source_name}")
        print(f"  document_id : {result.document_id}")
        print(f"  profile_id  : {result.profile_id}")
        print(f"  chunks      : {result.chunk_count}")
        print(f"  timings_ms  : {result.timings_ms}")
        for note in result.diagnostics:
            print(f"  note        : {note}")
    return 0


def _cmd_search(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    service = SearchService(database, settings)
    query = " ".join(args.query)
    response = service.search(
        query=query,
        profile_id=args.profile,
        mode=args.mode,
        top_k=args.top_k,
        collection=args.collection,
        candidates=args.candidates,
        rerank=args.rerank,
        reranker_model=args.reranker_model,
        rerank_candidates=args.rerank_candidates,
        query_mode=args.query_mode,
    )
    if args.json:
        print(response.model_dump_json(indent=2))
        return 0

    rerank_note = ""
    if response.rerank and response.reranker:
        rerank_note = (
            f" rerank={response.reranker.get('model_id')} "
            f"pool={response.rerank_candidates}"
        )
    print(
        f"query={response.query!r} mode={response.mode} query_mode={response.query_mode} "
        f"profile={response.profile_id} "
        f"({response.profile_name}) hits={len(response.hits)}{rerank_note} "
        f"total_ms={response.total_ms} timings_ms={response.timings_ms}"
    )
    if not response.hits:
        print("  no hits")
    for hit in response.hits:
        where = hit.source_name
        if hit.page_start is not None:
            where += f" p.{hit.page_start}" + (
                f"-{hit.page_end}" if hit.page_end and hit.page_end != hit.page_start else ""
            )
        if hit.section_path:
            where += f" | {hit.section_path}"
        excerpt = " ".join(hit.text.split())
        components = ""
        if hit.bm25_rank is not None or hit.dense_rank is not None:
            components = f" [bm25_rank={hit.bm25_rank} dense_rank={hit.dense_rank}]"
        rerank_info = ""
        if hit.retrieval_rank is not None:
            movement = hit.rank_movement or 0
            rerank_info = (
                f" [retrieval_rank={hit.retrieval_rank} rerank_score={hit.rerank_score:.4f} "
                f"movement={movement:+d}]"
            )
        print(f"  #{hit.rank} {hit.score_kind}={hit.score:.4f} {where}{components}{rerank_info}")
        print(f"     {excerpt[:200]}")
    return 0


def _cmd_chunks(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    service = IngestionService(database, settings)
    profile_id = args.profile
    if profile_id is None:
        profiles = service.list_profiles()
        if not profiles:
            print("no index profiles exist yet", file=sys.stderr)
            return 0
        profile_id = profiles[-1].profile_id
    chunks, total = service.list_chunks(
        profile_id, document_id=args.document, limit=args.limit
    )
    if args.json:
        print(
            json.dumps(
                {"profile_id": profile_id, "total": total,
                 "chunks": [c.model_dump(mode="json") for c in chunks]},
                indent=2,
            )
        )
        return 0
    print(f"profile={profile_id} chunks={total}")
    for chunk in chunks:
        where = chunk.section_path or "(no section)"
        number = f" [{chunk.section_number}]" if chunk.section_number else ""
        pages = f" p.{chunk.page_start}" if chunk.page_start is not None else ""
        print(f"  #{chunk.chunk_order} tokens={chunk.token_count}{pages} {where}{number}")
        print(f"     {' '.join(chunk.text.split())[:160]}")
        for note in chunk.diagnostics:
            print(f"     note: {note}")
    return 0


def _cmd_documents(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    service = IngestionService(database, settings)
    documents, total = service.list_documents(collection=args.collection, limit=args.limit)
    if args.json:
        print(
            json.dumps(
                {"total": total, "documents": [d.model_dump(mode="json") for d in documents]},
                indent=2,
            )
        )
        return 0
    print(f"{total} document(s)")
    for document in documents:
        print(
            f"  {document.document_id}  {document.source_name}  "
            f"[{document.collection}] chunks={document.chunk_count} status={document.status}"
        )
    return 0


def _cmd_profiles(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    service = IngestionService(database, settings)
    profiles = service.list_profiles()
    if args.json:
        print(json.dumps([p.model_dump(mode="json") for p in profiles], indent=2))
        return 0
    print(f"{len(profiles)} profile(s)")
    for profile in profiles:
        dense = (
            f"model={profile.embedding_model} dim={profile.embedding_dimension}"
            if profile.embedding_model
            else "lexical"
        )
        print(
            f"  {profile.profile_id}  {profile.name}  "
            f"strategy={profile.chunker_strategy} params={profile.chunker_params} "
            f"{dense} chunks={profile.chunk_count}"
        )
    return 0


def _cmd_rebuild_vectors(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    indexer = VectorIndexer(database, settings)
    targets = [args.profile] if args.profile else indexer.dense_profile_ids()
    if not targets:
        print("no dense profiles to rebuild", file=sys.stderr)
        return 0
    results = [indexer.rebuild_profile(pid, drop=not args.keep) for pid in targets]
    if args.json:
        print(json.dumps([r.model_dump(mode="json") for r in results], indent=2))
        return 0
    for result in results:
        print(
            f"rebuilt {result.profile_id} -> {result.collection}: "
            f"chunks={result.chunks} points={result.points_written} "
            f"duration_ms={result.duration_ms}"
        )
        for note in result.diagnostics:
            print(f"  note: {note}")
    return 0


def _cmd_vectors_status(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    indexer = VectorIndexer(database, settings)
    targets = [args.profile] if args.profile else indexer.dense_profile_ids()
    if not targets:
        print("no dense profiles", file=sys.stderr)
        return 0
    statuses = [indexer.reconcile_profile(pid, repair=args.repair) for pid in targets]
    if args.json:
        print(json.dumps([s.model_dump(mode="json") for s in statuses], indent=2))
        return 0
    for status in statuses:
        flag = "in-sync" if status.in_sync else "OUT-OF-SYNC"
        print(
            f"  {status.profile_id}  {status.collection}  [{flag}] "
            f"chunks={status.chunk_count} points={status.point_count} "
            f"missing={status.missing_points} orphans={status.orphan_points}"
        )
    return 0


def _evaluation_service(settings: Settings, database: Database) -> EvaluationService:
    ingestion = IngestionService(database, settings)
    search = SearchService(database, settings)
    return EvaluationService(database, settings, ingestion, search)


def _cmd_datasets(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    service = _evaluation_service(settings, database)
    command = args.datasets_command
    if command == "list":
        datasets = service.list_datasets()
        if args.json:
            print(json.dumps(datasets, indent=2))
            return 0
        if not datasets:
            print("no datasets available", file=sys.stderr)
            return 0
        print(f"{len(datasets)} dataset(s)")
        for item in datasets:
            print(
                f"  {item['slug']}  {item['name']}  v{item['version']}  "
                f"mapping={item['relevance_mapping']} docs={item['corpus_count']} "
                f"queries={item['query_count']} qrels={item['qrel_count']}"
            )
            print(f"     source: {item['source']}")
        return 0

    if command == "import":
        summary = service.import_beir(
            name=args.name,
            corpus_path=args.corpus,
            queries_path=args.queries,
            qrels_path=args.qrels,
            source=args.source,
            license=args.license,
            version=args.version,
            description=args.description,
            relevance_mapping=args.mapping,
            evidence_path=args.evidence,
        )
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"imported '{summary['name']}' -> {summary['slug']} "
                f"({summary['corpus_count']} docs, {summary['query_count']} queries, "
                f"{summary['qrel_count']} qrels, mapping={summary['relevance_mapping']})"
            )
        return 0

    if command == "download-scifact":
        summary = service.download_and_import_scifact(name=args.name, version=args.version)
        if args.json:
            print(json.dumps(summary, indent=2))
        else:
            print(
                f"downloaded and imported SciFact -> {summary['slug']} "
                f"({summary['corpus_count']} docs, {summary['query_count']} queries, "
                f"{summary['qrel_count']} qrels)"
            )
        return 0

    if command == "ingest":
        report = service.ingest_dataset(
            args.dataset,
            collection=args.collection,
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            profile_name=args.name,
            embed=args.embed,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
            limit=args.limit,
        )
        if args.json:
            print(report.model_dump_json(indent=2))
        else:
            print(
                f"ingested '{report.dataset}' into collection '{report.collection}' "
                f"profile={report.profile_id}: documents={report.documents} "
                f"new={report.new_documents} duplicates={report.duplicates} "
                f"chunks={report.chunks}"
            )
            for note in report.diagnostics:
                print(f"  note: {note}")
        return 0

    raise RagWorkbenchError(f"unknown datasets command: {command}", code="cli_error")


def _cmd_eval(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    service = _evaluation_service(settings, database)
    command = args.eval_command

    if command == "run":
        k_values = tuple(int(part) for part in args.k.split(",") if part.strip())
        candidate_depth = args.candidate_depth or settings.default_candidate_depth
        config = EvalRunConfig(
            dataset=args.dataset,
            profile_id=args.profile,
            collection=args.collection,
            mode=args.mode,
            query_mode=args.query_mode,
            candidate_depth=candidate_depth,
            top_k=args.top_k,
            k_values=k_values,
            rerank=args.rerank,
            rerank_candidates=args.rerank_candidates,
            reranker_model=args.reranker_model,
            candidates=args.candidates,
            max_queries=args.max_queries,
            label=args.label,
            subset=args.subset,
        )
        record = service.run(config)
        if args.out is not None:
            args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
        if args.json:
            print(json.dumps(record, indent=2))
            return 0
        print(f"experiment {record['experiment_id']} ({record.get('source', 'live')})")
        print(f"  dataset   : {record['dataset'].get('name')} v{record['dataset'].get('version')}")
        print(f"  config    : mode={record['config'].get('mode')} "
              f"query_mode={record['config'].get('query_mode')} "
              f"profile={record['config'].get('profile_id')} "
              f"rerank={record['config'].get('rerank')} "
              f"depth={record['config'].get('candidate_depth')}")
        print(f"  counts    : {record['counts']}")
        for name in sorted(record["aggregate"]):
            print(f"  {name:<12}: {record['aggregate'][name]:.4f}")
        for name in sorted(record.get("candidate_metrics", {})):
            print(f"  {name:<12}: {record['candidate_metrics'][name]:.4f}")
        rerank_stats = record.get("latency", {}).get("rerank_ms")
        if rerank_stats:
            print(f"  rerank_ms : {rerank_stats}")
            print(f"  model_init: {record['latency'].get('model_init_ms')} ms")
        for note in record.get("notes", []):
            print(f"  note      : {note}")
        return 0

    if command == "list":
        experiments = service.list_experiments(dataset=args.dataset)
        if args.json:
            print(json.dumps(experiments, indent=2))
            return 0
        if not experiments:
            print("no experiments recorded", file=sys.stderr)
            return 0
        for item in experiments:
            recall = item.get("aggregate", {}).get("recall@10")
            recall_text = f" recall@10={recall:.4f}" if isinstance(recall, float) else ""
            print(
                f"  {item['experiment_id']} [{item.get('source')}] "
                f"dataset={item.get('dataset')} mode={item.get('mode')} "
                f"rerank={item.get('rerank')}{recall_text}"
            )
        return 0

    if command == "show":
        record = service.get_experiment(args.experiment_id)
        print(json.dumps(record, indent=2))
        return 0

    if command == "export":
        if args.format == "csv":
            text = service.export_csv(args.experiment_id)
        else:
            text = json.dumps(service.export_json(args.experiment_id), indent=2)
        if args.out is not None:
            args.out.write_text(text, encoding="utf-8")
            print(f"wrote {args.out}")
        else:
            print(text)
        return 0

    if command == "failures":
        record = service.get_experiment(args.experiment_id)
        shown = 0
        for query in record.get("per_query", []):
            codes = [
                item["code"]
                for item in query.get("diagnostics", [])
                if item["code"] not in {"unresolved"}
            ]
            if not codes:
                continue
            if shown >= args.limit:
                break
            shown += 1
            metrics = query.get("metrics") or {}
            recall = metrics.get(f"recall@{10}")
            top_key = query["trace"][0]["key"] if query.get("trace") else "(none)"
            print(
                f"  {query['query_id']}  relevant={query['num_relevant']} "
                f"recall@10={recall if recall is None else round(recall, 3)} "
                f"top={top_key}  codes={','.join(codes)}"
            )
            print(f"     {query['query'][:120]}")
        print(f"{shown} failure(s) shown for {args.experiment_id}")
        return 0

    if command == "record":
        record = service.get_experiment(args.experiment_id)
        # A committed artifact is explicitly labelled as previously recorded so a
        # read-only deployment never implies the run happened in that process.
        record["source"] = "recorded"
        if not args.full:
            record = compact_record(record)
        directory = args.directory or settings.recorded_experiments_dir
        directory.mkdir(parents=True, exist_ok=True)
        filename = args.name or f"{args.experiment_id}.json"
        path = directory / filename
        # Compact JSON: these are machine artifacts, and the per-query records are
        # large; pretty-printing would only inflate the committed diff.
        path.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
        print(f"recorded {args.experiment_id} -> {path}")
        return 0

    raise RagWorkbenchError(f"unknown eval command: {command}", code="cli_error")


def _answer_service(settings: Settings, database: Database) -> AnswerService:
    return AnswerService(database, settings, SearchService(database, settings))


def _answer_quality_dataset_path(settings: Settings, name: str) -> Path:
    candidate = Path(name)
    if candidate.is_file():
        return candidate
    bundled = settings.bundled_datasets_dir / f"{name}.json"
    if bundled.is_file():
        return bundled
    raise RagWorkbenchError(
        f"answer-quality dataset not found: {name}", code="dataset_not_found"
    )


def _require_generation(settings: Settings) -> None:
    if not settings.generation_enabled:
        raise GeneratorUnavailableError(
            "live answer generation is disabled; set "
            "RAG_WORKBENCH_GENERATION_ENABLED=true and run Ollama to enable it"
        )


def _format_answer(response: AnswerResponse) -> str:
    lines = [f"status  : {response.status}", f"question: {response.question}"]
    if response.answer:
        lines.append("answer  :")
        lines.extend(f"  {line}" for line in response.answer.splitlines())
    if response.reason:
        lines.append(f"reason  : {response.reason}")
    if response.claims:
        lines.append("claims  :")
        for index, claim in enumerate(response.claims, 1):
            cites = " ".join(f"[{c}]" for c in claim.citations) or "(no citation)"
            lines.append(f"  {index}. {claim.text}  {cites}")
    if response.unanswered:
        lines.append("unanswered: " + "; ".join(response.unanswered))
    validation = response.citation_validation
    lines.append(
        f"citation: structural={validation.structural_valid} "
        f"valid={validation.citation_valid} "
        f"unknown={validation.unknown_source_ids} "
        f"semantic={validation.semantic_support}"
    )
    lines.append(
        f"cit-full: all_citations_resolved={validation.all_citations_resolved} "
        f"all_claims_cited={validation.all_claims_cited} "
        f"answer_claims_consistency={validation.answer_claims_consistency}"
    )
    if validation.malformed_citations:
        lines.append(f"malformed: {validation.malformed_citations}")
    context = response.context
    lines.append(
        f"context : retrieved={context.retrieved} selected={context.selected} "
        f"excluded={len(response.excluded_evidence)} "
        f"tokens~{context.token_estimate}/{context.token_budget} ({context.token_count_method})"
    )
    lines.append(
        "evidence: "
        + ", ".join(
            f"[{item.source_id}] {item.source_name}"
            + (f" #{item.section_number}" if item.section_number else "")
            for item in response.evidence
        )
    )
    lines.append(f"timings : {response.timings_ms}")
    if response.diagnostics:
        lines.append("diagnostics:")
        lines.extend(f"  {d['code']}: {d['detail']}" for d in response.diagnostics)
    return "\n".join(lines)


def _format_quality_report(report: dict) -> str:
    lines = [
        f"answer quality {report['report_id']} "
        f"({report['dataset']['name']} v{report['dataset']['version']})",
        f"model  : {report['config'].get('model_id')}",
        f"counts : {report['counts']}",
    ]
    for name in sorted(report["metrics"]):
        lines.append(f"  {name:<34}: {report['metrics'][name]}")
    loss = report.get("evidence_loss")
    if loss:
        lines.append(
            "evidence: "
            f"included={loss['included']} "
            f"retrieved_but_excluded={loss['retrieved_but_excluded']} "
            f"never_retrieved={loss['never_retrieved']} "
            f"(of {loss['questions_with_reference']} with reference)"
        )
        for case in loss["cases"]:
            if case["classification"] != "included":
                missing = case["missing_selected"] or case["missing_retrieved"]
                lines.append(
                    f"    {case['id']}: {case['classification']} "
                    f"missing={missing}"
                )
    lines.append(
        f"latency: retrieval={report['latency']['retrieval_ms']['p50']}ms "
        f"generation={report['latency']['generation_ms']['p50']}ms "
        f"total={report['latency']['total_ms']['p50']}ms (p50)"
    )
    return "\n".join(lines)


def _demo_record(
    *,
    item,
    response: AnswerResponse,
    dataset,
    run_config: dict,
    created_at,
) -> dict:
    record = response.model_dump()
    record.update(
        {
            "schema_version": DEMO_SCHEMA_VERSION,
            "kind": "answer_demo",
            "demo_id": new_demo_id(item.question, created_at=created_at),
            "created_at": created_at.isoformat(),
            "source": "recorded",
            "recorded": True,
            "dataset": {
                "name": dataset.name,
                "version": dataset.version,
                "collection": dataset.collection,
            },
            "run_config": run_config,
            "expected": {
                "status": item.expected_status,
                "reference_answer": item.reference_answer,
                "reference_keywords": item.reference_keywords,
                "reference_sections": item.reference_sections,
                "note": item.note,
            },
            "runtime": runtime_environment(),
            "git_commit": git_commit(find_repo_root()),
        }
    )
    return record


def _cmd_answer(settings: Settings, database: Database, args: argparse.Namespace) -> int:
    command = args.answer_command

    if command == "demos":
        demos = load_answer_demos(settings.recorded_answers_dir)
        if args.json:
            print(json.dumps([answer_demo_summary(d) for d in demos], indent=2))
            return 0
        if not demos:
            print(
                f"no recorded answer demos in {settings.recorded_answers_dir}",
                file=sys.stderr,
            )
            return 0
        for demo in demos:
            summary = answer_demo_summary(demo)
            print(
                f"  {summary['demo_id']}  [{summary['status']}]  "
                f"citation_valid={summary['citation_valid']}  "
                f"{str(summary['question'])[:80]}"
            )
        return 0

    if command == "demo":
        record = find_answer_demo(settings.recorded_answers_dir, args.demo_id)
        if record is None:
            raise AnswerNotFoundError(f"no recorded answer demo with id '{args.demo_id}'")
        print(json.dumps(record, indent=2))
        return 0

    if command == "ask":
        _require_generation(settings)
        response = _answer_service(settings, database).answer(
            question=args.question,
            profile_id=args.profile,
            mode=args.mode,
            collection=args.collection,
            query_mode=args.query_mode,
            candidate_depth=args.candidate_depth,
            candidates=args.candidates,
            rerank=args.rerank,
            reranker_model=args.reranker_model,
            rerank_candidates=args.rerank_candidates,
            max_evidence=args.max_evidence,
            context_tokens=args.context_tokens,
            temperature=args.temperature,
            num_predict=args.num_predict,
            seed=args.seed,
        )
        if args.json:
            print(response.model_dump_json(indent=2))
            return 0
        print(_format_answer(response))
        return 0

    if command in {"quality", "record"}:
        _require_generation(settings)
        dataset = load_answer_quality_dataset(
            _answer_quality_dataset_path(settings, args.dataset)
        )
        runner = AnswerQualityRunner(_answer_service(settings, database))
        report, responses = runner.run_with_responses(
            dataset,
            profile_id=args.profile,
            mode=args.mode,
            query_mode=args.query_mode,
            candidate_depth=args.candidate_depth,
            rerank=args.rerank,
            rerank_candidates=args.rerank_candidates,
            max_evidence=args.max_evidence,
            context_tokens=args.context_tokens,
            seed=args.seed,
            label=args.label,
            recommended=args.recommended,
        )
        if command == "quality":
            if args.out is not None:
                args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
                print(f"wrote {args.out}")
            print(_format_quality_report(report))
            return 0

        directory = args.directory or settings.recorded_answers_dir
        run_config = report["config"]
        created_at = datetime.now(UTC)
        if not args.no_demos:
            for item, response in responses:
                record = _demo_record(
                    item=item,
                    response=response,
                    dataset=dataset,
                    run_config=run_config,
                    created_at=created_at,
                )
                path = write_answer_demo(directory, record)
                print(f"recorded demo {record['demo_id']} -> {path.name}")
        path = write_quality_report(directory, report)
        print(f"recorded quality report {report['report_id']} -> {path.name}")
        return 0

    raise RagWorkbenchError(f"unknown answer command: {command}", code="cli_error")


def _cmd_scaling(settings: Settings, args: argparse.Namespace) -> int:
    service = ScalingService(settings)
    command = args.scaling_command
    # Benchmarking emits one INFO line per ingested document; that is useful in
    # normal use but drowns a benchmark's own output. Quiet only that logger.
    logging.getLogger("rag_workbench.ingestion.pipeline").setLevel(logging.WARNING)
    # The load-test client logs one INFO line per HTTP request via httpx/httpcore,
    # which would bury the one-line summary under thousands of lines.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    if command == "corpus":
        config = CorpusConfig(
            documents=args.documents,
            seed=args.seed,
            paragraphs_per_section=args.paragraphs_per_section,
            words_per_paragraph=args.words_per_paragraph,
        )
        documents = generate_corpus(config)
        manifest = write_corpus(documents, args.out)
        print(f"wrote {manifest['documents']} synthetic documents to {args.out}")
        print(f"fingerprint: {manifest['fingerprint']}")
        print(f"size classes: {manifest['size_classes']}")
        return 0

    if command == "capacity":
        base = StorageAssumptions(
            documents=args.documents,
            chunks_per_document=args.chunks_per_document,
            embedding_dimensions=args.embedding_dimensions,
            bytes_per_dimension=args.bytes_per_dimension,
            vector_indexes=args.vector_indexes,
            replication_factor=args.replication_factor,
        )
        if args.scenario:
            base = storage_scenarios(base)[args.scenario]
        estimate = estimate_storage(base)
        payload: dict[str, object] = {"storage": estimate.model_dump(mode="json")}
        if args.documents_per_day is not None:
            throughput = args.embedding_chunks_per_second_per_worker or 20.0
            ingestion = estimate_ingestion(
                IngestionAssumptions(
                    documents_per_day=args.documents_per_day,
                    chunks_per_document=base.chunks_per_document,
                    embedding_chunks_per_second_per_worker=throughput,
                    workers=args.workers,
                )
            )
            payload["ingestion"] = ingestion.model_dump(mode="json")
        if args.json:
            print(json.dumps(payload, indent=2))
            return 0
        print(
            f"storage estimate for {base.documents:,} documents "
            f"({base.chunks_per_document} chunks/doc)"
        )
        for component in estimate.components:
            print(
                f"  {component.label:<32} {human_bytes(component.bytes):>12}  [{component.basis}]"
            )
        print(f"  {'primary':<32} {human_bytes(estimate.primary_bytes):>12}")
        print(f"  {'replicas':<32} {human_bytes(estimate.replica_bytes):>12}")
        print(f"  {'backups':<32} {human_bytes(estimate.backup_bytes):>12}")
        print(f"  {'reindex working set':<32} {human_bytes(estimate.reindex_bytes):>12}")
        print(f"  {'operational headroom':<32} {human_bytes(estimate.headroom_bytes):>12}")
        print(f"  {'TOTAL':<32} {human_bytes(estimate.total_bytes):>12}")
        if "ingestion" in payload:
            ing = payload["ingestion"]
            assert isinstance(ing, dict)
            print(
                f"ingestion: {ing['chunks_per_day']:,.0f} chunks/day, bottleneck "
                f"{ing['bottleneck_stage']} {ing['bottleneck_per_day']:,.0f} chunks/day, "
                f"utilization {ing['utilization']:.2f}, "
                f"keeps_up={ing['keeps_up']} (conditional projection, basis "
                f"'{ing['basis']}')"
            )
        return 0

    if command == "run":
        sizes = [int(part) for part in str(args.documents).split(",") if part.strip()]
        depths = [int(part) for part in str(args.depths).split(",") if part.strip()]
        modes = [part.strip() for part in str(args.modes).split(",") if part.strip()]
        config = CorpusConfig(documents=max(sizes), seed=args.seed)
        record = run_scaling_benchmark(
            settings,
            config=config,
            workloads=sizes,
            strategy=args.strategy,
            max_tokens=args.max_tokens,
            overlap_tokens=args.overlap_tokens,
            candidate_depths=depths,
            modes=modes,
            retrieval_queries=args.queries,
            progress=lambda message: print(message),
        )
        if args.out is not None:
            args.out.write_text(json.dumps(record, indent=2), encoding="utf-8")
            print(f"wrote {args.out}")
        if args.record:
            path = service.write_artifact(record, filename=args.name)
            print(f"recorded artifact: {path}")
        if args.json:
            print(json.dumps(record, indent=2))
            return 0
        print(f"benchmark {record['benchmark_id']} (schema {record['schema_version']})")
        print(f"provenance: {record.get('provenance')}")
        for workload in record["workloads"]:
            ingestion = workload["ingestion"]
            index = workload["index"]
            print(
                f"  {workload['documents']:>6} docs  {workload['chunks']:>7} chunks  "
                f"{ingestion['documents_per_second']:.1f} docs/s  "
                f"db={human_bytes(index['db_bytes'])}  fts={human_bytes(index['fts_bytes'])}"
            )
            for retrieval in workload["retrieval"]:
                stats = retrieval["latency_ms"]
                print(
                    f"      {retrieval['mode']}@{retrieval['candidate_depth']}: "
                    f"p50={stats['p50']}ms p95={stats['p95']}ms errors={retrieval['errors']}"
                )
        for failure in record["failures"]:
            print(f"  ! {failure}")
        return 0

    if command == "list":
        benchmarks = service.list_benchmarks()
        if args.json:
            print(json.dumps(benchmarks, indent=2))
            return 0
        if not benchmarks:
            print("no recorded scaling benchmarks", file=sys.stderr)
            return 0
        for item in benchmarks:
            marker = " [INVALID]" if item.get("invalid") else ""
            print(
                f"  {item['benchmark_id']}{marker} workloads={item['workload_count']} "
                f"docs_max={item['documents_max']} "
                f"modes={','.join(item.get('modes') or [])} created={item.get('created_at')}"
            )
        return 0

    if command == "show":
        print(json.dumps(service.get_benchmark(args.benchmark_id), indent=2))
        return 0

    if command == "load-test":
        result = run_load_test_sync(
            args.url,
            workload=args.workload,
            users=args.users,
            duration_s=args.duration,
            warmup_s=args.warmup,
        )
        if args.json:
            print(json.dumps(result, indent=2))
            return 0
        print(f"load test '{result['workload']}' against {result['base_url']}")
        print(
            f"  users={result['concurrent_users']} warmup={result['warmup_s']}s "
            f"measured={result['duration_s']}s requests={result['requests']} "
            f"rps={result['requests_per_second']} errors={result['errors']} "
            f"error_rate={result['error_rate']}"
        )
        latency = result["latency_ms"]
        print(f"  latency ms: p50={latency['p50']} p95={latency['p95']} mean={latency['mean']}")
        for name, endpoint in (result.get("by_endpoint") or {}).items():
            endpoint_latency = endpoint["latency_ms"]
            print(
                f"  endpoint {name:<20} requests={endpoint['requests']:>6} "
                f"errors={endpoint['errors']:>3} p50={endpoint_latency['p50']} "
                f"p95={endpoint_latency['p95']}"
            )
        print("  (smoke-level local observation; HTTP throughput, not generation throughput)")
        return 0

    raise RagWorkbenchError(f"unknown scaling command: {command}", code="cli_error")


def _cmd_serve(settings: Settings, args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run(
        "rag_workbench.api.app:create_app",
        factory=True,
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    try:
        if args.command == "serve":
            return _cmd_serve(settings, args)

        if args.command == "scaling":
            # Scalability commands use their own isolated databases (or none at
            # all) and must not create the default database as a side effect.
            return _cmd_scaling(settings, args)

        database = Database(settings.resolved_database_url)
        database.create_schema()

        if args.command == "init-db":
            print(f"initialized database at {settings.resolved_database_url}")
            return 0
        if args.command == "ingest":
            return _cmd_ingest(settings, database, args)
        if args.command == "search":
            return _cmd_search(settings, database, args)
        if args.command == "documents":
            return _cmd_documents(settings, database, args)
        if args.command == "profiles":
            return _cmd_profiles(settings, database, args)
        if args.command == "chunks":
            return _cmd_chunks(settings, database, args)
        if args.command == "rebuild-vectors":
            return _cmd_rebuild_vectors(settings, database, args)
        if args.command == "vectors-status":
            return _cmd_vectors_status(settings, database, args)
        if args.command == "datasets":
            return _cmd_datasets(settings, database, args)
        if args.command == "eval":
            return _cmd_eval(settings, database, args)
        if args.command == "answer":
            return _cmd_answer(settings, database, args)
        parser.error(f"unknown command: {args.command}")
        return 2
    except RagWorkbenchError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    finally:
        close_vector_stores()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
