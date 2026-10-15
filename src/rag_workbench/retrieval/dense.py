"""Dense semantic retrieval over the Qdrant-backed vector index.

Retrieval is a two-step, explicitly separated process:

1. encode the query with the configured :class:`EmbeddingProvider`;
2. search the profile's vector collection, then resolve authoritative text and
   provenance from SQLite by stable chunk ID.

Similarity scores are cosine similarity where **larger is better**; results are
returned best-first with a 1-based ``rank``. The vector store never supplies
document text, so provenance cannot drift from the relational store.
"""

from __future__ import annotations

import time
from collections.abc import Sequence

from sqlalchemy import select
from sqlalchemy.orm import Session

from rag_workbench.domain.interfaces import EmbeddingProvider, VectorStore
from rag_workbench.domain.schemas import SearchHit
from rag_workbench.embeddings import validate_dimension
from rag_workbench.persistence.models import Chunk, Document


class DenseRetriever:
    """Profile-scoped semantic retriever."""

    def __init__(
        self,
        store: VectorStore,
        provider: EmbeddingProvider,
        *,
        profile_id: str,
        collection: str,
        dimension: int,
    ) -> None:
        self._store = store
        self._provider = provider
        self._profile_id = profile_id
        self._collection = collection
        self._dimension = dimension

    def search(
        self,
        session: Session,
        *,
        query: str,
        top_k: int,
        candidates: int,
        collection: str | None = None,
    ) -> tuple[list[SearchHit], dict[str, float]]:
        timings: dict[str, float] = {}

        embed_started = time.perf_counter()
        query_vector = self._provider.embed_query(query)
        timings["embed_query"] = round((time.perf_counter() - embed_started) * 1000.0, 3)
        validate_dimension(self._dimension, len(query_vector), context="query embedding")

        search_started = time.perf_counter()
        matches = self._store.search(
            self._collection, query_vector, limit=candidates, collection=collection
        )
        timings["vector_search"] = round((time.perf_counter() - search_started) * 1000.0, 3)

        resolve_started = time.perf_counter()
        hits = self._resolve(session, matches, top_k=top_k, collection=collection)
        timings["resolve"] = round((time.perf_counter() - resolve_started) * 1000.0, 3)

        return hits, timings

    def _resolve(
        self,
        session: Session,
        matches: Sequence[object],
        *,
        top_k: int,
        collection: str | None,
    ) -> list[SearchHit]:
        if not matches:
            return []

        order = [match.id for match in matches]  # type: ignore[attr-defined]
        score_by_id = {match.id: match.score for match in matches}  # type: ignore[attr-defined]

        rows = session.execute(
            select(Chunk, Document)
            .join(Document, Document.id == Chunk.document_id)
            .where(Chunk.chunk_id.in_(order), Chunk.profile_id == self._profile_id)
        ).all()
        by_id = {chunk.chunk_id: (chunk, document) for chunk, document in rows}

        hits: list[SearchHit] = []
        for chunk_id in order:
            pair = by_id.get(chunk_id)
            if pair is None:
                continue  # vector point without a matching chunk: ignore, not invent
            chunk, document = pair
            if collection is not None and document.collection != collection:
                continue
            hits.append(
                SearchHit(
                    rank=len(hits) + 1,
                    chunk_id=chunk.chunk_id,
                    document_id=document.id,
                    source_name=document.source_name,
                    collection=document.collection,
                    chunk_order=chunk.chunk_order,
                    text=chunk.text,
                    score=float(score_by_id[chunk_id]),
                    score_kind="cosine",
                    section_path=chunk.section_path,
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                )
            )
            if len(hits) >= top_k:
                break
        return hits


__all__ = ["DenseRetriever"]
