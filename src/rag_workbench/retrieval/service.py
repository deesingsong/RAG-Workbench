"""Retrieval service: profile resolution, mode dispatch and timing capture.

Phase 2 exposes three modes over the same profile resolution and timing
contract:

* ``bm25``   — lexical retrieval over SQLite FTS5.
* ``dense``  — cosine similarity over the profile's Qdrant vector index.
* ``hybrid`` — Reciprocal Rank Fusion of the two candidate lists.

Dense and hybrid modes require a *dense* profile (one with an ``embedding_model``)
and only load an embedding model when actually invoked. The provider and vector
store are injected as factories so tests can run the real pipeline with a
deterministic fake embedding provider and no model download.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.config import IndexProfileSpec, Settings
from rag_workbench.domain.errors import (
    ProfileNotDenseError,
    ProfileNotFoundError,
    ValidationError,
    VectorIndexNotBuiltError,
)
from rag_workbench.domain.interfaces import EmbeddingProvider, VectorStore
from rag_workbench.domain.schemas import SearchHit, SearchResponse
from rag_workbench.embeddings import build_embedding_provider
from rag_workbench.ingestion.vector_index import spec_from_profile_row
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Chunk, Document, IndexProfile
from rag_workbench.persistence.vector_store import get_vector_store
from rag_workbench.retrieval.bm25 import Bm25Retriever
from rag_workbench.retrieval.dense import DenseRetriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion

SUPPORTED_MODES = ("bm25", "dense", "hybrid")
MAX_TOP_K = 50
MAX_CANDIDATES = 500

StoreFactory = Callable[[], VectorStore]
ProviderFactory = Callable[[IndexProfileSpec], EmbeddingProvider]


class SearchService:
    """Resolve a profile, run the requested retrieval mode and report timings."""

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
        self._bm25 = Bm25Retriever()
        self._store_factory: StoreFactory = vector_store_factory or (
            lambda: get_vector_store(settings.vector_store_path)
        )
        self._provider_factory: ProviderFactory = embedding_provider_factory or (
            lambda spec: build_embedding_provider(spec, settings)
        )
        self._store: VectorStore | None = None

    # ------------------------------------------------------------------ public

    def search(
        self,
        *,
        query: str,
        profile_id: str | None = None,
        mode: str = "bm25",
        top_k: int = 5,
        collection: str | None = None,
        candidates: int | None = None,
    ) -> SearchResponse:
        if mode not in SUPPORTED_MODES:
            raise ValidationError(
                f"mode '{mode}' is not supported; choose one of: {', '.join(SUPPORTED_MODES)}",
                code="unsupported_mode",
            )
        if top_k < 1 or top_k > MAX_TOP_K:
            raise ValidationError(f"top_k must be between 1 and {MAX_TOP_K}", code="invalid_top_k")
        candidate_k = self._settings.dense_candidate_k if candidates is None else candidates
        if candidate_k < 1 or candidate_k > MAX_CANDIDATES:
            raise ValidationError(
                f"candidates must be between 1 and {MAX_CANDIDATES}", code="invalid_candidates"
            )
        candidate_k = max(candidate_k, top_k)

        started = time.perf_counter()
        with self._db.session() as session:
            profile = self._resolve_profile(session, profile_id)

            if mode == "bm25":
                hits, timings = self._run_bm25(
                    session, query=query, profile=profile, top_k=top_k, collection=collection
                )
            elif mode == "dense":
                hits, timings = self._run_dense(
                    session,
                    query=query,
                    profile=profile,
                    top_k=top_k,
                    candidates=candidate_k,
                    collection=collection,
                )
            else:
                hits, timings = self._run_hybrid(
                    session,
                    query=query,
                    profile=profile,
                    top_k=top_k,
                    candidates=candidate_k,
                    collection=collection,
                )

        total_ms = (time.perf_counter() - started) * 1000.0
        timings["total"] = round(total_ms, 3)
        return SearchResponse(
            query=query,
            mode=mode,
            profile_id=profile.id,
            profile_name=profile.name,
            config_hash=profile.config_hash,
            collection=collection,
            top_k=top_k,
            candidates=candidate_k if mode != "bm25" else None,
            timings_ms=timings,
            total_ms=round(total_ms, 3),
            hits=hits,
        )

    # ------------------------------------------------------------------ modes

    def _run_bm25(
        self,
        session: Session,
        *,
        query: str,
        profile: IndexProfile,
        top_k: int,
        collection: str | None,
    ) -> tuple[list[SearchHit], dict[str, float]]:
        started = time.perf_counter()
        hits = self._bm25.search(
            session, query=query, profile_id=profile.id, top_k=top_k, collection=collection
        )
        return hits, {"retrieval": self._ms(started)}

    def _run_dense(
        self,
        session: Session,
        *,
        query: str,
        profile: IndexProfile,
        top_k: int,
        candidates: int,
        collection: str | None,
    ) -> tuple[list[SearchHit], dict[str, float]]:
        retriever = self._dense_retriever(profile)
        hits, timings = retriever.search(
            session, query=query, top_k=top_k, candidates=candidates, collection=collection
        )
        return hits, timings

    def _run_hybrid(
        self,
        session: Session,
        *,
        query: str,
        profile: IndexProfile,
        top_k: int,
        candidates: int,
        collection: str | None,
    ) -> tuple[list[SearchHit], dict[str, float]]:
        lexical_started = time.perf_counter()
        bm25_hits = self._bm25.search(
            session, query=query, profile_id=profile.id, top_k=candidates, collection=collection
        )
        timings = {"lexical": self._ms(lexical_started)}

        retriever = self._dense_retriever(profile)
        dense_hits, dense_timings = retriever.search(
            session, query=query, top_k=candidates, candidates=candidates, collection=collection
        )
        timings.update({f"dense_{key}": value for key, value in dense_timings.items()})

        fusion_started = time.perf_counter()
        fused = reciprocal_rank_fusion(
            {
                "bm25": [hit.chunk_id for hit in bm25_hits],
                "dense": [hit.chunk_id for hit in dense_hits],
            },
            k=self._settings.rrf_k,
        )
        timings["fusion"] = self._ms(fusion_started)

        bm25_by_id = {hit.chunk_id: hit for hit in bm25_hits}
        dense_by_id = {hit.chunk_id: hit for hit in dense_hits}
        ordered = fused[:top_k]
        metadata = self._chunk_metadata(session, [item.chunk_id for item in ordered], profile.id)

        hits: list[SearchHit] = []
        for item in ordered:
            pair = metadata.get(item.chunk_id)
            if pair is None:
                continue  # index drift: never invent a source
            chunk, document = pair
            if collection is not None and document.collection != collection:
                continue
            lexical = bm25_by_id.get(item.chunk_id)
            semantic = dense_by_id.get(item.chunk_id)
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    chunk_id=chunk.chunk_id,
                    document_id=document.id,
                    source_name=document.source_name,
                    collection=document.collection,
                    chunk_order=chunk.chunk_order,
                    text=chunk.text,
                    score=item.score,
                    score_kind="rrf",
                    section_path=chunk.section_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    bm25_rank=lexical.rank if lexical else None,
                    bm25_score=lexical.score if lexical else None,
                    dense_rank=semantic.rank if semantic else None,
                    dense_score=semantic.score if semantic else None,
                    fused_score=item.score,
                )
            )
        return hits, timings

    # ------------------------------------------------------------------ helpers

    def _dense_retriever(self, profile: IndexProfile) -> DenseRetriever:
        spec = spec_from_profile_row(profile)
        if not spec.is_dense:
            raise ProfileNotDenseError(
                f"profile '{profile.id}' is lexical-only; dense and hybrid retrieval "
                "require an embedding model (create a dense profile with --embed)"
            )
        assert spec.embedding_dimension is not None
        store = self._get_store()
        collection = spec.vector_collection
        if not store.has_collection(collection):
            raise VectorIndexNotBuiltError(
                f"vector index for profile '{profile.id}' has not been built; "
                f"run `rag-workbench rebuild-vectors --profile {profile.id}`"
            )
        observed = store.collection_dimension(collection)
        if observed is not None and observed != spec.embedding_dimension:
            raise VectorIndexNotBuiltError(
                f"vector collection '{collection}' has dimension {observed} but profile "
                f"'{profile.id}' expects {spec.embedding_dimension}; rebuild the vector index"
            )
        provider = self._provider_factory(spec)
        return DenseRetriever(
            store,
            provider,
            profile_id=profile.id,
            collection=collection,
            dimension=spec.embedding_dimension,
        )

    def _get_store(self) -> VectorStore:
        if self._store is None:
            self._store = self._store_factory()
        return self._store

    @staticmethod
    def _chunk_metadata(
        session: Session, chunk_ids: Sequence[str], profile_id: str
    ) -> dict[str, tuple[Chunk, Document]]:
        if not chunk_ids:
            return {}
        rows = session.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.chunk_id.in_(list(chunk_ids)), Chunk.profile_id == profile_id)
        ).all()
        return {chunk.chunk_id: (chunk, document) for chunk, document in rows}

    @staticmethod
    def _resolve_profile(session: Session, profile_id: str | None) -> IndexProfile:
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

    @staticmethod
    def _ms(started: float) -> float:
        return round((time.perf_counter() - started) * 1000.0, 3)

    def profile_chunk_count(self, profile_id: str) -> int:
        with self._db.session() as session:
            return int(session.query(Chunk).filter(Chunk.profile_id == profile_id).count())


__all__ = ["SUPPORTED_MODES", "SearchService"]
