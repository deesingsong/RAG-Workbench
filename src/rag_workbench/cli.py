"""Command-line interface.

    rag-workbench init-db
    rag-workbench ingest path/to/file.pdf --collection contracts
    rag-workbench search "termination for convenience" --top-k 5
    rag-workbench documents
    rag-workbench profiles
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
from rag_workbench.logging_config import configure_logging
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-workbench",
        description="Local-first RAG retrieval laboratory (Phase 0-1: ingestion + BM25).",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("init-db", help="create the SQLite schema and FTS5 index")

    ingest = sub.add_parser("ingest", help="ingest a TXT/MD/PDF document")
    ingest.add_argument("path", type=Path, help="path to the document")
    ingest.add_argument("--collection", default=None, help="collection (dataset) name")
    ingest.add_argument(
        "--strategy", choices=["paragraph", "fixed_token"], default=None, help="chunking strategy"
    )
    ingest.add_argument("--max-tokens", type=int, default=None)
    ingest.add_argument("--overlap-tokens", type=int, default=None)
    ingest.add_argument("--name", default=None, help="human label for the index profile")
    ingest.add_argument("--json", action="store_true", help="emit machine-readable JSON")

    search = sub.add_parser("search", help="run a BM25 search")
    search.add_argument("query", nargs="+", help="query text")
    search.add_argument("--top-k", type=int, default=5)
    search.add_argument("--profile", default=None, help="index profile ID")
    search.add_argument("--collection", default=None)
    search.add_argument("--json", action="store_true")

    documents = sub.add_parser("documents", help="list indexed documents")
    documents.add_argument("--collection", default=None)
    documents.add_argument("--limit", type=int, default=50)
    documents.add_argument("--json", action="store_true")

    profiles = sub.add_parser("profiles", help="list index profiles")
    profiles.add_argument("--json", action="store_true")

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
        status = "duplicate (already indexed)" if result.is_duplicate else "ingested"
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
        query=query, profile_id=args.profile, top_k=args.top_k, collection=args.collection
    )
    if args.json:
        print(response.model_dump_json(indent=2))
        return 0

    print(
        f"query={response.query!r} mode={response.mode} profile={response.profile_id} "
        f"({response.profile_name}) hits={len(response.hits)} "
        f"retrieval_ms={response.timings_ms.get('retrieval')}"
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
        print(f"  #{hit.rank} score={hit.score:.4f} {where}")
        print(f"     {excerpt[:200]}")
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
        print(
            f"  {profile.profile_id}  {profile.name}  "
            f"strategy={profile.chunker_strategy} params={profile.chunker_params} "
            f"chunks={profile.chunk_count}"
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
        parser.error(f"unknown command: {args.command}")
        return 2
    except RagWorkbenchError as exc:
        print(f"error [{exc.code}]: {exc.message}", file=sys.stderr)
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
