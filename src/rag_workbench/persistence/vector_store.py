"""Persistent vector storage in Qdrant local mode.

SQLite is the authoritative store for documents, chunks and provenance; this
module stores only vectors and the minimal payload needed for filtering
(``document_id``, ``collection``, ``profile_id``). Point IDs are the stable
chunk UUIDs, so an upsert is idempotent.

Ownership model (documented, not engineered away): Qdrant local mode takes an
exclusive lock on its storage directory. This process keeps exactly one client
per resolved path via :func:`get_vector_store`, and :func:`close_vector_stores`
releases it. Do **not** point two processes at the same vector directory.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client import models as qm

from rag_workbench.domain.errors import DimensionMismatchError, VectorStoreError
from rag_workbench.domain.schemas import VectorMatch, VectorPoint

logger = logging.getLogger(__name__)

DEFAULT_METRIC = "cosine"
_UPSERT_BATCH = 128

_METRIC_TO_QDRANT = {
    "cosine": qm.Distance.COSINE,
    "dot": qm.Distance.DOT,
    "euclid": qm.Distance.EUCLID,
}

_STORES: dict[str, QdrantLocalStore] = {}
_STORES_LOCK = threading.Lock()


def collection_name(profile_id: str) -> str:
    """Deterministic collection name for an index profile."""

    return f"rw_{profile_id}"


class QdrantLocalStore:
    """Thin, explicit wrapper over ``QdrantClient`` in local persistence mode."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        try:
            self._path.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise VectorStoreError(
                f"could not create vector store directory {self._path}: {exc}"
            ) from exc
        try:
            self._client = QdrantClient(path=str(self._path))
        except Exception as exc:  # noqa: BLE001 - lock contention, corruption, permissions
            raise VectorStoreError(
                f"could not open Qdrant local storage at {self._path}: {exc}. If another "
                "process owns this directory, close it first (local mode is single-owner)."
            ) from exc

    # ---------------------------------------------------------------- lifecycle

    @property
    def path(self) -> Path:
        return self._path

    def close(self) -> None:
        try:
            self._client.close()
        except Exception:  # noqa: BLE001 - best effort on shutdown
            logger.warning("failed to close Qdrant client cleanly", extra={"path": str(self._path)})

    # ---------------------------------------------------------------- schema

    def has_collection(self, name: str) -> bool:
        try:
            return bool(self._client.collection_exists(name))
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not check collection '{name}': {exc}") from exc

    def collection_dimension(self, name: str) -> int | None:
        if not self.has_collection(name):
            return None
        try:
            info = self._client.get_collection(name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not read collection '{name}': {exc}") from exc
        vectors = info.config.params.vectors
        if isinstance(vectors, dict):
            if not vectors:
                return None
            vectors = next(iter(vectors.values()))
        size = getattr(vectors, "size", None)
        return int(size) if size is not None else None

    def ensure_collection(self, name: str, *, dimension: int, metric: str = DEFAULT_METRIC) -> None:
        if metric not in _METRIC_TO_QDRANT:
            raise VectorStoreError(f"unsupported similarity metric '{metric}'")
        if self.has_collection(name):
            existing = self.collection_dimension(name)
            if existing is not None and existing != dimension:
                raise DimensionMismatchError(
                    f"collection '{name}' stores {existing}-dimensional vectors but this profile "
                    f"expects {dimension}; rebuild the vector index"
                )
            return
        try:
            self._client.create_collection(
                name,
                vectors_config=qm.VectorParams(
                    size=dimension, distance=_METRIC_TO_QDRANT[metric]
                ),
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not create collection '{name}': {exc}") from exc

    def drop(self, name: str) -> None:
        if not self.has_collection(name):
            return
        try:
            self._client.delete_collection(name)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not delete collection '{name}': {exc}") from exc

    # ---------------------------------------------------------------- writes

    def upsert(self, name: str, points: Sequence[VectorPoint]) -> int:
        if not points:
            return 0
        written = 0
        for start in range(0, len(points), _UPSERT_BATCH):
            batch = points[start : start + _UPSERT_BATCH]
            try:
                self._client.upsert(
                    name,
                    points=[
                        qm.PointStruct(
                            id=point.id, vector=point.vector, payload=dict(point.payload)
                        )
                        for point in batch
                    ],
                    wait=True,
                )
            except Exception as exc:  # noqa: BLE001
                raise VectorStoreError(
                    f"failed to upsert {len(batch)} point(s) into '{name}': {exc}"
                ) from exc
            written += len(batch)
        return written

    def delete_ids(self, name: str, ids: Sequence[str]) -> int:
        if not ids or not self.has_collection(name):
            return 0
        try:
            self._client.delete(
                name, points_selector=qm.PointIdsList(points=list(ids)), wait=True
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"failed to delete points from '{name}': {exc}") from exc
        return len(ids)

    # ---------------------------------------------------------------- reads

    def search(
        self,
        name: str,
        vector: Sequence[float],
        *,
        limit: int,
        collection: str | None = None,
    ) -> list[VectorMatch]:
        if not self.has_collection(name):
            return []
        query_filter = None
        if collection is not None:
            query_filter = qm.Filter(
                must=[
                    qm.FieldCondition(key="collection", match=qm.MatchValue(value=collection))
                ]
            )
        try:
            response = self._client.query_points(
                name,
                query=list(vector),
                limit=limit,
                query_filter=query_filter,
                with_payload=True,
            )
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"vector search failed on '{name}': {exc}") from exc
        return [
            VectorMatch(
                id=str(point.id), score=float(point.score), payload=dict(point.payload or {})
            )
            for point in response.points
        ]

    def list_ids(self, name: str) -> set[str]:
        if not self.has_collection(name):
            return set()
        found: set[str] = set()
        offset = None
        try:
            while True:
                points, offset = self._client.scroll(
                    name, limit=1024, offset=offset, with_payload=False, with_vectors=False
                )
                found.update(str(point.id) for point in points)
                if offset is None:
                    break
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not enumerate points in '{name}': {exc}") from exc
        return found

    def count(self, name: str) -> int:
        if not self.has_collection(name):
            return 0
        try:
            return int(self._client.count(name).count)
        except Exception as exc:  # noqa: BLE001
            raise VectorStoreError(f"could not count points in '{name}': {exc}") from exc


def get_vector_store(path: Path | str) -> QdrantLocalStore:
    """Return the process-wide store for ``path`` (one client per path)."""

    key = str(Path(path).resolve())
    with _STORES_LOCK:
        store = _STORES.get(key)
        if store is None:
            store = QdrantLocalStore(key)
            _STORES[key] = store
        return store


def close_vector_stores() -> None:
    """Close and forget every cached store (used on shutdown and in tests)."""

    with _STORES_LOCK:
        for store in _STORES.values():
            store.close()
        _STORES.clear()


__all__ = [
    "QdrantLocalStore",
    "close_vector_stores",
    "collection_name",
    "get_vector_store",
]
