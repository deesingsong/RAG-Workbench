"""Real cross-encoder reranker smoke test (opt-in via ``-m reranker``).

This test loads a real Sentence Transformers cross-encoder. It is deselected by
default and needs the optional dependency and a one-time model download:

    uv sync --extra dense
    uv run pytest -m reranker -s

It is an integration smoke test, not a benchmark: it asserts that the real
cross-encoder reorders a real candidate list, keeps retrieval and rerank scores
separate, and reports its own initialization/inference timings.
"""

from __future__ import annotations

import pytest

from rag_workbench.config import (
    DEFAULT_RERANKER_MODEL,
    ChunkerConfig,
    IndexProfileSpec,
    Settings,
)
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService

pytestmark = pytest.mark.reranker

pytest.importorskip("sentence_transformers")


def test_cross_encoder_reranks_sample_contract(
    settings: Settings, database: Database, legal_contract_bytes: bytes
) -> None:
    profile = IndexProfileSpec(
        name="legal-clause",
        chunker=ChunkerConfig(strategy="clause_aware", max_tokens=120),
    )
    service = IngestionService(database, settings)
    result = service.ingest_bytes(
        legal_contract_bytes,
        source_name="legal_contract.txt",
        collection="contracts",
        profile=profile,
    )
    assert result.state == "completed"
    assert result.chunk_count > 0

    search = SearchService(database, settings)
    query = "confidential information"

    baseline = search.search(query=query, profile_id=profile.key, mode="bm25", top_k=5)
    reranked = search.search(
        query=query,
        profile_id=profile.key,
        mode="bm25",
        top_k=5,
        rerank=True,
        rerank_candidates=20,
    )

    assert baseline.hits, "BM25 returned no candidates to rerank"
    assert reranked.rerank is True
    assert reranked.reranker is not None
    assert reranked.reranker["model_id"] == DEFAULT_RERANKER_MODEL
    assert reranked.timings_ms["rerank_init"] >= 0
    assert reranked.timings_ms["rerank_inference"] >= 0
    assert [hit.rank for hit in reranked.hits] == list(range(1, len(reranked.hits) + 1))
    assert all(hit.rerank_score is not None for hit in reranked.hits)
    assert all(hit.retrieval_rank is not None for hit in reranked.hits)
    assert all(hit.retrieval_score_kind == "bm25" for hit in reranked.hits)
    # Every reranked hit was an original candidate (the reranker adds none).
    original_ids = {hit.chunk_id for hit in baseline.hits}
    candidate_ids = {hit.chunk_id for hit in reranked.hits}
    assert candidate_ids <= original_ids

    # Informational comparison, printed with -s.
    bm25_view = [(h.rank, h.section_number, round(h.score, 3)) for h in baseline.hits]
    print(f"\nBM25     top: {bm25_view}")
    print("RERANKED top:")
    for hit in reranked.hits:
        print(
            f"  #{hit.rank} num={hit.section_number} rerank={hit.rerank_score:.3f} "
            f"retrieval_rank={hit.retrieval_rank} movement={hit.rank_movement}"
        )
    print(f"timings bm25={baseline.timings_ms} reranked={reranked.timings_ms}")
