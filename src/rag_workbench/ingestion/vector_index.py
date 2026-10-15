"""Vector indexing, rebuild and reconciliation.

This module owns everything about keeping the dense index in sync with SQLite:

* :meth:`VectorIndexer.index_chunks` embeds and upserts the chunks of one document.
* :meth:`VectorIndexer.rebuild_profile` regenerates a whole profile's index purely
  from chunks already stored in SQLite — no re-upload or re-parse required.
* :meth:`VectorIndexer.reconcile_profile` detects missing points (chunks without
  vectors) and orphans (vectors without chunks), optionally repairing them.
* :meth:`VectorIndexer.delete_points` propagates document deletion.

Provider and store construction are injected as factories so tests can supply a
deterministic fake embedding provider while still exercising the real Qdrant
local store.
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from sqlalchemy import select

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.domain.errors import NotFoundError, ProfileNotDenseError
from rag_workbench.domain.interfaces import EmbeddingProvider, VectorStore
from rag_workbench.domain.schemas import VectorIndexStatus, VectorPoint, VectorRebuildResult
from rag_workbench.embeddings import build_embedding_provider, validate_dimension
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Chunk, Document, IndexProfile
from rag_workbench.persistence.vector_store import collection_name, get_vector_store

logger = logging.getLogger(__name__)

StoreFactory = Callable[[], VectorStore]
ProviderFactory = Callable[[IndexProfileSpec], EmbeddingProvider]


@dataclass(frozen=True)
class VectorChunk:
    """The minimum needed to embed and locate one chunk."""

    chunk_id: str
    text: str
    document_id: str
    collection: str


@dataclass(frozen=True)
class IndexReport:
    """Outcome of indexing a set of chunks."""

    points_written: int
    embedding_ms: float
    upsert_ms: float
    diagnostics: list[str] = field(default_factory=list)


def spec_from_profile_row(row: IndexProfile) -> IndexProfileSpec:
    """Reconstruct the immutable profile spec from its persisted row."""

    params = dict(row.embedding_params or {})
    return IndexProfileSpec(
        name=row.name,
        parser_version=row.parser_version,
        chunker=ChunkerConfig(**dict(row.chunker_params or {})),
        embedding_model=row.embedding_model,
        embedding_dimension=row.embedding_dimension,
        embedding_revision=params.get("revision"),
        similarity_metric=params.get("metric", "cosine"),
        normalize_embeddings=params.get("normalize", True),
        embedding_document_prefix=params.get("document_prefix", ""),
        embedding_query_prefix=params.get("query_prefix", ""),
    )


class VectorIndexer:
    """Keeps the dense index synchronized with the relational chunk store."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        *,
        vector_store_factory: StoreFactory | None = None,
        embedding_provider_factory: ProviderFactory | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._store_factory: StoreFactory = vector_store_factory or (
            lambda: get_vector_store(settings.vector_store_path)
        )
        self._provider_factory: ProviderFactory = embedding_provider_factory or (
            lambda spec: build_embedding_provider(spec, settings)
        )
        self._store: VectorStore | None = None

    def _get_store(self) -> VectorStore:
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    # ------------------------------------------------------------------ indexing

    def _embed_slices(
        self, spec: IndexProfileSpec, chunks: Sequence[VectorChunk], collection: str
    ) -> IndexReport:
        store = self._get_store()
        provider = self._provider_factory(spec)
        assert spec.embedding_dimension is not None
        store.ensure_collection(
            collection, dimension=spec.embedding_dimension, metric=spec.similarity_metric
        )

        slice_size = max(self._settings.embedding_batch_size * 4, 64)
        written = 0
        embedding_ms = 0.0
        upsert_ms = 0.0
        diagnostics: list[str] = []

        for start in range(0, len(chunks), slice_size):
            batch = chunks[start : start + slice_size]
            embed_started = time.perf_counter()
            result = provider.embed_documents([item.text for item in batch])
            embedding_ms += (time.perf_counter() - embed_started) * 1000.0
            validate_dimension(spec.embedding_dimension, result.dimension)
            for note in result.diagnostics:
                if note not in diagnostics:
                    diagnostics.append(note)

            points = [
                VectorPoint(
                    id=item.chunk_id,
                    vector=vector,
                    payload={
                        "document_id": item.document_id,
                        "profile_id": spec.key,
                        "collection": item.collection,
                    },
                )
                for item, vector in zip(batch, result.vectors, strict=True)
            ]
            upsert_started = time.perf_counter()
            written += store.upsert(collection, points)
            upsert_ms += (time.perf_counter() - upsert_started) * 1000.0

        return IndexReport(
            points_written=written,
            embedding_ms=round(embedding_ms, 3),
            upsert_ms=round(upsert_ms, 3),
            diagnostics=diagnostics,
        )

    # ------------------------------------------------------------------ public

    def dense_profile_ids(self) -> list[str]:
        """IDs of every profile that stores dense vectors, oldest first."""

        with self._db.session() as session:
            rows = (
                session.execute(
                    select(IndexProfile.id)
                    .where(IndexProfile.embedding_model.is_not(None))
                    .order_by(IndexProfile.created_at)
                )
                .scalars()
                .all()
            )
        return list(rows)

    def index_chunks(self, profile_id: str, chunks: Sequence[VectorChunk]) -> IndexReport:
        """Embed and upsert the chunks of a single document (idempotent per ID)."""

        with self._db.session() as session:
            profile_row = session.get(IndexProfile, profile_id)
            if profile_row is None:
                raise NotFoundError(f"index profile not found: {profile_id}")
            spec = spec_from_profile_row(profile_row)
        if not spec.is_dense:
            raise ProfileNotDenseError(f"profile '{spec.key}' is lexical-only")
        return self._embed_slices(spec, chunks, spec.vector_collection)

    # ------------------------------------------------------------------ rebuild

    def rebuild_profile(self, profile_id: str, *, drop: bool = True) -> VectorRebuildResult:
        """Rebuild a profile's vectors from the chunks already stored in SQLite."""

        started = time.perf_counter()
        spec, chunks = self._load_profile_chunks(profile_id, with_text=True)
        collection = spec.vector_collection
        store = self._get_store()
        if drop:
            store.drop(collection)

        report = self._embed_slices(spec, chunks, collection)
        duration_ms = (time.perf_counter() - started) * 1000.0
        logger.info(
            "vector index rebuilt",
            extra={
                "profile_id": profile_id,
                "collection": collection,
                "chunks": len(chunks),
                "points_written": report.points_written,
                "duration_ms": round(duration_ms, 3),
            },
        )
        return VectorRebuildResult(
            profile_id=profile_id,
            collection=collection,
            chunks=len(chunks),
            points_written=report.points_written,
            duration_ms=round(duration_ms, 3),
            diagnostics=report.diagnostics,
        )

    # ------------------------------------------------------------------ reconcile

    def reconcile_profile(self, profile_id: str, *, repair: bool = False) -> VectorIndexStatus:
        """Compare SQLite chunks with vector points; optionally repair the delta."""

        spec, chunks = self._load_profile_chunks(profile_id, with_text=repair)
        collection = spec.vector_collection
        chunk_ids = {item.chunk_id for item in chunks}
        store = self._get_store()

        point_ids = store.list_ids(collection)
        missing = chunk_ids - point_ids
        orphans = point_ids - chunk_ids

        if repair and missing:
            to_index = sorted(
                (item for item in chunks if item.chunk_id in missing),
                key=lambda item: item.chunk_id,
            )
            by_collection: dict[str, list[VectorChunk]] = {}
            for item in to_index:
                by_collection.setdefault(item.collection, []).append(item)
            for label, items in by_collection.items():
                self._embed_slices(spec, items, collection)
                logger.info(
                    "reconcile repaired missing vectors",
                    extra={"profile_id": profile_id, "doc_collection": label, "count": len(items)},
                )
        if repair and orphans:
            store.delete_ids(collection, sorted(orphans))
            logger.info(
                "reconcile removed orphan vectors",
                extra={"profile_id": profile_id, "count": len(orphans)},
            )

        missing_after = 0 if repair else len(missing)
        orphans_after = 0 if repair else len(orphans)
        return VectorIndexStatus(
            profile_id=profile_id,
            collection=collection,
            embedding_model=spec.embedding_model,
            dimension=spec.embedding_dimension,
            chunk_count=len(chunk_ids),
            point_count=store.count(collection),
            missing_points=missing_after,
            orphan_points=orphans_after,
            in_sync=(missing_after == 0 and orphans_after == 0),
        )

    # ------------------------------------------------------------------ deletion

    def delete_points(self, profile_id: str, chunk_ids: Sequence[str]) -> int:
        """Remove vector points for deleted chunks (no-op for lexical profiles)."""

        if not chunk_ids:
            return 0
        with self._db.session() as session:
            profile_row = session.get(IndexProfile, profile_id)
            if profile_row is None or not profile_row.embedding_model:
                return 0
        return self._get_store().delete_ids(collection_name(profile_id), list(chunk_ids))

    # ------------------------------------------------------------------ helpers

    def _load_profile_chunks(
        self, profile_id: str, *, with_text: bool
    ) -> tuple[IndexProfileSpec, list[VectorChunk]]:
        with self._db.session() as session:
            profile_row = session.get(IndexProfile, profile_id)
            if profile_row is None:
                raise NotFoundError(f"index profile not found: {profile_id}")
            spec = spec_from_profile_row(profile_row)
            if not spec.is_dense:
                raise ProfileNotDenseError(
                    f"profile '{profile_id}' is lexical-only; it has no vector index to manage"
                )
            columns = [Chunk.chunk_id, Chunk.document_id, Document.collection]
            if with_text:
                columns.insert(1, Chunk.text)
            rows = session.execute(
                select(*columns)
                .join(Document, Document.id == Chunk.document_id)
                .where(Chunk.profile_id == profile_id)
            ).all()

        chunks: list[VectorChunk] = []
        for row in rows:
            if with_text:
                chunks.append(
                    VectorChunk(
                        chunk_id=row[0], text=row[1], document_id=row[2], collection=row[3]
                    )
                )
            else:
                chunks.append(
                    VectorChunk(chunk_id=row[0], text="", document_id=row[1], collection=row[2])
                )
        return spec, chunks


__all__ = ["IndexReport", "VectorChunk", "VectorIndexer", "spec_from_profile_row"]
