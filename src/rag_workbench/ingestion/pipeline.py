"""Ingestion pipeline and corpus catalog.

:class:`IngestionService` is the single write path: it parses bytes, chunks them
under a profile, persists documents/chunks (which populates FTS5 via triggers),
and records an :class:`IngestionJob`. Re-ingesting the same bytes into the same
profile is idempotent and reported as a duplicate; the same bytes under a
different profile produce a separate, isolated set of chunks.

For a *dense* profile the same call also embeds and upserts each chunk into the
profile's vector collection. Vector indexing runs **after** the chunk write
transaction commits and outside it, so a long embedding job never holds the
single SQLite writer lock. If embedding or the vector store fails, the chunks stay
durable and the job is recorded as ``partial`` (BM25-searchable) rather than
silently pretending the document is fully indexed; ``rebuild-vectors`` repairs it.

The service is single-writer by design (one SQLite writer); concurrent ingestion
is a documented non-goal for the prototype.
"""

from __future__ import annotations

import hashlib
import logging
import time
import uuid
from pathlib import Path

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from rag_workbench.config import IndexProfileSpec, Settings
from rag_workbench.domain.errors import (
    FileTooLargeError,
    IngestionError,
    NotFoundError,
    RagWorkbenchError,
)
from rag_workbench.domain.schemas import (
    ChunkRecord,
    DocumentDeleteResult,
    DocumentRecord,
    IngestionResult,
    ProfileRecord,
)
from rag_workbench.ingestion.chunkers import get_chunker
from rag_workbench.ingestion.parsers import get_parser, mime_type_for
from rag_workbench.ingestion.vector_index import VectorChunk, VectorIndexer
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Chunk, Document, IndexProfile, IngestionJob, utcnow

logger = logging.getLogger(__name__)

DOCUMENT_NAMESPACE = uuid.UUID("b7d1f3a4-5c6e-4a2b-9d3f-1e8c0a7b6d52")


def make_document_id(collection: str, checksum: str) -> str:
    """Deterministic document ID from its collection and content checksum."""

    return str(uuid.uuid5(DOCUMENT_NAMESPACE, f"rag-workbench:document:{collection}:{checksum}"))


class IngestionService:
    """Parse, chunk and persist documents; answer catalog read queries."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        *,
        vector_indexer: VectorIndexer | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._vector_indexer = vector_indexer

    @property
    def _indexer(self) -> VectorIndexer:
        if self._vector_indexer is None:
            self._vector_indexer = VectorIndexer(self._db, self._settings)
        return self._vector_indexer

    # ------------------------------------------------------------------ ingest

    def ingest_path(
        self, path: Path, *, collection: str, profile: IndexProfileSpec
    ) -> IngestionResult:
        if not path.is_file():
            raise NotFoundError(f"file not found: {path}")
        data = path.read_bytes()
        return self.ingest_bytes(
            data, source_name=path.name, collection=collection, profile=profile
        )

    def ingest_bytes(
        self,
        data: bytes,
        *,
        source_name: str,
        collection: str,
        profile: IndexProfileSpec,
    ) -> IngestionResult:
        if len(data) > self._settings.max_upload_bytes:
            raise FileTooLargeError(
                f"file exceeds the {self._settings.max_upload_mb} MB upload limit"
            )
        if not data:
            # Empty files are handled as valid-but-empty documents below.
            logger.info("ingesting empty document", extra={"source_name": source_name})

        mime_type = mime_type_for(source_name)  # raises for unsupported extensions
        parser = get_parser(source_name, mime_type)

        checksum = hashlib.sha256(data).hexdigest()
        document_id = make_document_id(collection, checksum)
        job_id = str(uuid.uuid4())

        # Transaction 1: make sure the profile exists and the job is durable.
        with self._db.session() as session:
            self._ensure_profile(session, profile)
            session.add(
                IngestionJob(
                    id=job_id,
                    document_id=None,
                    profile_id=profile.key,
                    collection=collection,
                    state="running",
                    progress=0.0,
                )
            )

        started = time.perf_counter()
        try:
            with self._db.session() as session:
                # Fast path: document + profile already indexed -> duplicate.
                if self._already_indexed(session, document_id, profile.key):
                    duration_ms = (time.perf_counter() - started) * 1000.0
                    chunk_count = self._count_chunks(session, document_id, profile.key)
                    job = session.get(IngestionJob, job_id)
                    assert job is not None
                    job.state = "duplicate"
                    job.progress = 1.0
                    job.chunk_count = chunk_count
                    job.finished_at = utcnow()
                    job.duration_ms = duration_ms
                    session.flush()
                    return IngestionResult(
                        job_id=job_id,
                        document_id=document_id,
                        profile_id=profile.key,
                        state="duplicate",
                        is_duplicate=True,
                        chunk_count=chunk_count,
                        source_name=source_name,
                        checksum=checksum,
                        collection=collection,
                        diagnostics=["document already indexed for this profile"],
                        timings_ms={"total": round(duration_ms, 3)},
                    )

                parse_started = time.perf_counter()
                parsed = parser.parse(data, source_name=source_name, mime_type=mime_type)
                parse_ms = (time.perf_counter() - parse_started) * 1000.0

                document = session.get(Document, document_id)
                if document is None:
                    document = Document(
                        id=document_id,
                        collection=collection,
                        source_name=source_name,
                        source_checksum=checksum,
                        mime_type=mime_type,
                        byte_size=len(data),
                        status="empty" if not parsed.blocks else "indexed",
                        diagnostics=list(parsed.diagnostics),
                    )
                    session.add(document)
                    session.flush()

                chunk_started = time.perf_counter()
                chunker = get_chunker(profile.chunker)
                chunk_data = chunker.chunk(
                    parsed, profile.chunker, profile_key=profile.key, document_id=document_id
                )
                chunk_ms = (time.perf_counter() - chunk_started) * 1000.0

                index_started = time.perf_counter()
                session.add_all(
                    [
                        Chunk(
                            chunk_id=chunk.chunk_id,
                            document_id=document_id,
                            profile_id=profile.key,
                            chunk_order=chunk.order,
                            text=chunk.text,
                            token_count=chunk.token_count,
                            section_path=chunk.section_path,
                            page_start=chunk.page_start,
                            page_end=chunk.page_end,
                            char_start=chunk.char_start,
                            char_end=chunk.char_end,
                            section_number=chunk.section_number,
                            section_title=chunk.section_title,
                            parent_clause=chunk.parent_clause,
                            section_level=chunk.section_level,
                            diagnostics=list(chunk.diagnostics),
                        )
                        for chunk in chunk_data
                    ]
                )
                session.flush()
                index_ms = (time.perf_counter() - index_started) * 1000.0

                # Snapshot the inputs needed for embedding; the ORM rows expire
                # when this session closes.
                vector_chunks = (
                    [
                        VectorChunk(
                            chunk_id=chunk.chunk_id,
                            text=chunk.text,
                            document_id=document_id,
                            collection=collection,
                        )
                        for chunk in chunk_data
                    ]
                    if profile.is_dense
                    else []
                )

            # Chunks are now durable. Index vectors outside the write transaction.
            vector_index_ms: float | None = None
            vector_error: tuple[str, str] | None = None
            vector_diagnostics: list[str] = []
            if profile.is_dense and vector_chunks:
                vector_started = time.perf_counter()
                try:
                    report = self._indexer.index_chunks(profile.key, vector_chunks)
                    vector_index_ms = (time.perf_counter() - vector_started) * 1000.0
                    vector_diagnostics = list(report.diagnostics)
                except RagWorkbenchError as exc:
                    vector_error = (exc.code, exc.message)
                except Exception as exc:  # noqa: BLE001 - recorded, not swallowed
                    vector_error = ("vector_indexing_failed", str(exc))

            duration_ms = (time.perf_counter() - started) * 1000.0
            timings = {
                "parse": round(parse_ms, 3),
                "chunk": round(chunk_ms, 3),
                "index": round(index_ms, 3),
            }
            if vector_index_ms is not None:
                timings["vector_index"] = round(vector_index_ms, 3)
            timings["total"] = round(duration_ms, 3)

            diagnostics = list(parsed.diagnostics) + vector_diagnostics
            state = "completed"
            if vector_error is not None:
                state = "partial"
                diagnostics.append(
                    f"vector indexing failed [{vector_error[0]}]: {vector_error[1]}. "
                    "Chunks are searchable via BM25; run `rag-workbench rebuild-vectors` "
                    "to complete the dense index."
                )

            with self._db.session() as session:
                job = session.get(IngestionJob, job_id)
                assert job is not None
                job.document_id = document_id
                job.state = state
                job.progress = 1.0
                job.chunk_count = len(chunk_data)
                job.finished_at = utcnow()
                job.duration_ms = duration_ms
                job.diagnostics = diagnostics
                job.timings_ms = timings

            logger.info(
                "document ingested",
                extra={
                    "document_id": document_id,
                    "profile_id": profile.key,
                    "collection": collection,
                    "chunk_count": len(chunk_data),
                    "state": state,
                    "duration_ms": round(duration_ms, 3),
                },
            )
            return IngestionResult(
                job_id=job_id,
                document_id=document_id,
                profile_id=profile.key,
                state=state,
                is_duplicate=False,
                chunk_count=len(chunk_data),
                source_name=source_name,
                checksum=checksum,
                collection=collection,
                diagnostics=diagnostics,
                timings_ms=timings,
            )
        except RagWorkbenchError as exc:
            self._mark_failed(job_id, exc.code, exc.message, started)
            raise
        except Exception as exc:  # noqa: BLE001 - converted to an actionable error
            self._mark_failed(job_id, "ingestion_error", str(exc), started)
            raise IngestionError(f"failed to ingest '{source_name}': {exc}") from exc

    # ------------------------------------------------------------------ delete

    def delete_document(
        self, document_id: str, *, profile_id: str | None = None
    ) -> DocumentDeleteResult:
        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise NotFoundError(f"document not found: {document_id}")

            query = session.query(Chunk).filter(Chunk.document_id == document_id)
            if profile_id is not None:
                query = query.filter(Chunk.profile_id == profile_id)

            # Remember which (profile, chunk) pairs to remove from the vector index
            # before the rows disappear.
            removed = session.execute(
                select(Chunk.profile_id, Chunk.chunk_id).where(
                    Chunk.document_id == document_id,
                    *([Chunk.profile_id == profile_id] if profile_id is not None else []),
                )
            ).all()
            by_profile: dict[str, list[str]] = {}
            for row_profile_id, chunk_id in removed:
                by_profile.setdefault(row_profile_id, []).append(chunk_id)

            chunks_deleted = query.count()
            query.delete(synchronize_session=False)
            session.flush()

            remaining = (
                session.query(func.count(Chunk.id))
                .filter(Chunk.document_id == document_id)
                .scalar()
                or 0
            )
            document_deleted = False
            if remaining == 0:
                session.query(IngestionJob).filter(IngestionJob.document_id == document_id).update(
                    {"document_id": None}
                )
                session.delete(document)
                document_deleted = True

        # Propagate deletion to any dense profile's vector index. A failure here
        # leaves an orphan point, which `reconcile` detects and repairs; it must
        # not fail the relational deletion that already committed.
        for removed_profile_id, chunk_ids in by_profile.items():
            try:
                self._indexer.delete_points(removed_profile_id, chunk_ids)
            except Exception:  # noqa: BLE001 - best effort, reconcilable
                logger.warning(
                    "could not remove vector points for deleted chunks",
                    extra={"profile_id": removed_profile_id, "document_id": document_id},
                )

        logger.info(
            "document deletion",
            extra={
                "document_id": document_id,
                "profile_id": profile_id,
                "chunks_deleted": chunks_deleted,
                "document_deleted": document_deleted,
            },
        )
        return DocumentDeleteResult(
            document_id=document_id,
            profile_id=profile_id,
            chunks_deleted=chunks_deleted,
            document_deleted=document_deleted,
        )

    # ----------------------------------------------------------------- catalog

    def list_documents(
        self, *, collection: str | None = None, limit: int = 50, offset: int = 0
    ) -> tuple[list[DocumentRecord], int]:
        with self._db.session() as session:
            base = select(Document)
            count_stmt = select(func.count(Document.id))
            if collection is not None:
                base = base.where(Document.collection == collection)
                count_stmt = count_stmt.where(Document.collection == collection)

            total = session.execute(count_stmt).scalar() or 0
            rows = (
                session.execute(
                    base.order_by(Document.imported_at.desc(), Document.id)
                    .limit(limit)
                    .offset(offset)
                )
                .scalars()
                .all()
            )
            records = [self._document_record(session, row) for row in rows]
            return records, total

    def get_document(self, document_id: str) -> DocumentRecord:
        with self._db.session() as session:
            document = session.get(Document, document_id)
            if document is None:
                raise NotFoundError(f"document not found: {document_id}")
            return self._document_record(session, document)

    def list_profiles(self) -> list[ProfileRecord]:
        with self._db.session() as session:
            rows = (
                session.execute(select(IndexProfile).order_by(IndexProfile.created_at))
                .scalars()
                .all()
            )
            return [self._profile_record(session, row) for row in rows]

    def list_chunks(
        self,
        profile_id: str,
        *,
        document_id: str | None = None,
        limit: int = 200,
        offset: int = 0,
    ) -> tuple[list[ChunkRecord], int]:
        """Return stored chunks for a profile, with full provenance."""

        limit = max(1, min(limit, 1000))
        with self._db.session() as session:
            if session.get(IndexProfile, profile_id) is None:
                raise NotFoundError(f"index profile not found: {profile_id}")
            filters = [Chunk.profile_id == profile_id]
            if document_id is not None:
                filters.append(Chunk.document_id == document_id)
            total = int(
                session.query(func.count(Chunk.id)).filter(*filters).scalar() or 0
            )
            rows = session.execute(
                select(Chunk, Document)
                .join(Document, Document.id == Chunk.document_id)
                .where(*filters)
                .order_by(Chunk.document_id, Chunk.chunk_order)
                .limit(limit)
                .offset(offset)
            ).all()
            records = [
                ChunkRecord(
                    chunk_id=chunk.chunk_id,
                    document_id=chunk.document_id,
                    profile_id=chunk.profile_id,
                    source_name=document.source_name,
                    collection=document.collection,
                    chunk_order=chunk.chunk_order,
                    text=chunk.text,
                    token_count=chunk.token_count,
                    section_path=chunk.section_path,
                    section_number=chunk.section_number,
                    section_title=chunk.section_title,
                    parent_clause=chunk.parent_clause,
                    section_level=chunk.section_level,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    char_start=chunk.char_start,
                    char_end=chunk.char_end,
                    diagnostics=list(chunk.diagnostics or []),
                )
                for chunk, document in rows
            ]
            return records, total

    def get_job(self, job_id: str) -> IngestionJob:
        with self._db.session() as session:
            job = session.get(IngestionJob, job_id)
            if job is None:
                raise NotFoundError(f"ingestion job not found: {job_id}")
            session.expunge(job)
            return job

    # ------------------------------------------------------------------ helpers

    def _ensure_profile(self, session: Session, profile: IndexProfileSpec) -> IndexProfile:
        existing = session.get(IndexProfile, profile.key)
        if existing is not None:
            return existing
        row = IndexProfile(
            id=profile.key,
            name=profile.name,
            config_hash=profile.config_hash,
            parser_version=profile.parser_version,
            chunker_strategy=profile.chunker.strategy,
            chunker_params=profile.chunker.model_dump(),
            embedding_model=profile.embedding_model,
            embedding_dimension=profile.embedding_dimension,
            embedding_params={
                "revision": profile.embedding_revision,
                "metric": profile.similarity_metric,
                "normalize": profile.normalize_embeddings,
                "document_prefix": profile.embedding_document_prefix,
                "query_prefix": profile.embedding_query_prefix,
            },
            index_version=profile.index_version,
        )
        session.add(row)
        session.flush()
        return row

    @staticmethod
    def _already_indexed(session: Session, document_id: str, profile_id: str) -> bool:
        exists = session.get(Document, document_id)
        if exists is None:
            return False
        return (
            session.query(func.count(Chunk.id))
            .filter(Chunk.document_id == document_id, Chunk.profile_id == profile_id)
            .scalar()
            or 0
        ) > 0

    @staticmethod
    def _count_chunks(session: Session, document_id: str, profile_id: str) -> int:
        return int(
            session.query(func.count(Chunk.id))
            .filter(Chunk.document_id == document_id, Chunk.profile_id == profile_id)
            .scalar()
            or 0
        )

    def _mark_failed(self, job_id: str, code: str, message: str, started: float) -> None:
        duration_ms = (time.perf_counter() - started) * 1000.0
        try:
            with self._db.session() as session:
                job = session.get(IngestionJob, job_id)
                if job is None:
                    return
                job.state = "failed"
                job.error_code = code
                job.error_message = message
                job.finished_at = utcnow()
                job.duration_ms = duration_ms
        except Exception:  # noqa: BLE001 - never mask the original failure
            logger.exception("could not record ingestion failure", extra={"job_id": job_id})

    @staticmethod
    def _document_record(session: Session, document: Document) -> DocumentRecord:
        chunk_count = int(
            session.query(func.count(Chunk.id))
            .filter(Chunk.document_id == document.id)
            .scalar()
            or 0
        )
        return DocumentRecord(
            document_id=document.id,
            collection=document.collection,
            source_name=document.source_name,
            source_checksum=document.source_checksum,
            mime_type=document.mime_type,
            byte_size=document.byte_size,
            imported_at=document.imported_at,
            status=document.status,  # type: ignore[arg-type]
            diagnostics=list(document.diagnostics or []),
            chunk_count=chunk_count,
        )

    @staticmethod
    def _profile_record(session: Session, profile: IndexProfile) -> ProfileRecord:
        chunk_count = int(
            session.query(func.count(Chunk.id))
            .filter(Chunk.profile_id == profile.id)
            .scalar()
            or 0
        )
        params = dict(profile.embedding_params or {})
        is_dense = profile.embedding_model is not None
        return ProfileRecord(
            profile_id=profile.id,
            name=profile.name,
            config_hash=profile.config_hash,
            parser_version=profile.parser_version,
            chunker_strategy=profile.chunker_strategy,
            chunker_params=dict(profile.chunker_params or {}),
            embedding_model=profile.embedding_model,
            embedding_dimension=profile.embedding_dimension,
            embedding_revision=params.get("revision"),
            similarity_metric=params.get("metric", "cosine"),
            normalize_embeddings=params.get("normalize", True),
            vector_collection=f"rw_{profile.id}" if is_dense else None,
            index_version=profile.index_version,
            created_at=profile.created_at,
            chunk_count=chunk_count,
        )
