"""Retrieval service: profile resolution, mode dispatch and timing capture.

Phase 1 exposes only the ``bm25`` mode. The service is the single place that
resolves which index profile a query runs against, records a configuration
fingerprint, and measures retrieval latency; the API and the Streamlit UI both
go through it (the UI over HTTP).
"""

from __future__ import annotations

import time

from sqlalchemy import select

from rag_workbench.config import Settings
from rag_workbench.domain.errors import ProfileNotFoundError, ValidationError
from rag_workbench.domain.schemas import SearchResponse
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Chunk, IndexProfile
from rag_workbench.retrieval.bm25 import Bm25Retriever

SUPPORTED_MODES = ("bm25",)
MAX_TOP_K = 50


class SearchService:
    """Resolve a profile, run the requested retrieval mode and report timings."""

    def __init__(self, database: Database, settings: Settings) -> None:
        self._db = database
        self._settings = settings
        self._bm25 = Bm25Retriever()

    def search(
        self,
        *,
        query: str,
        profile_id: str | None = None,
        mode: str = "bm25",
        top_k: int = 5,
        collection: str | None = None,
    ) -> SearchResponse:
        if mode not in SUPPORTED_MODES:
            raise ValidationError(
                f"mode '{mode}' is not implemented in Phase 1; supported modes: "
                f"{', '.join(SUPPORTED_MODES)} (dense/hybrid arrive in Phase 2)",
                code="unsupported_mode",
            )
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValidationError(f"top_k must be between 1 and {MAX_TOP_K}", code="invalid_top_k")

        with self._db.session() as session:
            profile = self._resolve_profile(session, profile_id)
            retrieval_started = time.perf_counter()
            hits = self._bm25.search(
                session,
                query=query,
                profile_id=profile.id,
                top_k=top_k,
                collection=collection,
            )
            retrieval_ms = (time.perf_counter() - retrieval_started) * 1000.0

            return SearchResponse(
                query=query,
                mode=mode,
                profile_id=profile.id,
                profile_name=profile.name,
                config_hash=profile.config_hash,
                collection=collection,
                top_k=top_k,
                timings_ms={"retrieval": round(retrieval_ms, 3)},
                hits=hits,
            )

    @staticmethod
    def _resolve_profile(session, profile_id: str | None) -> IndexProfile:  # noqa: ANN001
        if profile_id is not None:
            profile = session.get(IndexProfile, profile_id)
            if profile is None:
                raise ProfileNotFoundError(f"index profile not found: {profile_id}")
            return profile

        newest = session.execute(
            select(IndexProfile).order_by(IndexProfile.created_at.desc()).limit(1)
        ).scalar_one_or_none()
        if newest is None:
            raise ProfileNotFoundError(
                "no index profiles exist yet; ingest a document before searching"
            )
        return newest

    def profile_chunk_count(self, profile_id: str) -> int:
        with self._db.session() as session:
            return int(
                session.query(Chunk)
                .filter(Chunk.profile_id == profile_id)
                .count()
            )


__all__ = ["SUPPORTED_MODES", "SearchService"]
