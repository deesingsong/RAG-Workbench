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
    rag-workbench serve

The CLI, the API and the tests all call the same core services; the CLI adds no
business logic of its own.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings, get_settings
from rag_workbench.domain.errors import RagWorkbenchError
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.logging_config import configure_logging
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.vector_store import close_vector_stores
from rag_workbench.retrieval.service import SearchService


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

    return parser


def _profile_from_args(settings: Settings, args: argparse.Namespace) -> IndexProfileSpec:
    chunker = ChunkerConfig(
        strategy=args.strategy or settings.default_chunk_strategy,
        max_tokens=args.max_tokens or settings.default_max_tokens,
        overlap_tokens=(
            args.overlap_tokens
            if args.overlap_tokens is not None
            else settings.default_overlap_tokens
        ),
    )
    if getattr(args, "embed", False):
        return settings.dense_profile(
            name=args.name or f"dense-{chunker.strategy}-{chunker.max_tokens}",
            chunker=chunker,
            embedding_model=args.embedding_model,
            embedding_dimension=args.embedding_dimension,
        )
    name = args.name or f"{chunker.strategy}-{chunker.max_tokens}"
    return IndexProfileSpec(name=name, chunker=chunker)


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
        f"query={response.query!r} mode={response.mode} profile={response.profile_id} "
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
        parser.error(f"unknown command: {args.command}")
        return 2
    except RagWorkbenchError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1
    finally:
        close_vector_stores()


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
