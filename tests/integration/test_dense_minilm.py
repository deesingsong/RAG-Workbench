"""Real Sentence Transformers smoke test (opt-in via ``-m dense``).

This is the *only* test that loads a real model. It is deselected by default and
needs the optional dependency and a one-time model download:

    uv sync --extra dense
    uv run pytest -m dense -s

It is an integration smoke test, not a benchmark: it asserts that the real model
produces a ranked, provenance-carrying result for a query whose answer is known
to be in the fixture.
"""

from __future__ import annotations

import pytest

from rag_workbench.config import Settings
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService

pytestmark = pytest.mark.dense

pytest.importorskip("sentence_transformers")


def test_minilm_dense_and_hybrid_on_sample_contract(
    settings: Settings, database: Database, contract_bytes: bytes
) -> None:
    profile = settings.dense_profile()
    service = IngestionService(database, settings)
    result = service.ingest_bytes(
        contract_bytes,
        source_name="sample_contract.txt",
        collection="contracts",
        profile=profile,
    )
    assert result.state == "completed"
    assert result.chunk_count > 0

    search = SearchService(database, settings)
    query = "termination for convenience"

    bm25 = search.search(query=query, profile_id=profile.key, mode="bm25", top_k=5)
    dense = search.search(query=query, profile_id=profile.key, mode="dense", top_k=5)
    hybrid = search.search(query=query, profile_id=profile.key, mode="hybrid", top_k=5)

    assert dense.hits, "dense retrieval returned no hits"
    assert all(hit.source_name == "sample_contract.txt" for hit in dense.hits)
    assert all(hit.score_kind == "cosine" for hit in dense.hits)
    assert dense.timings_ms["embed_query"] >= 0

    assert any(
        "termination" in (hit.section_path or "").lower() or "termination" in hit.text.lower()
        for hit in dense.hits[:3]
    ), "MiniLM did not rank a termination-related chunk in the top 3"

    assert hybrid.hits
    assert any(hit.bm25_rank is not None for hit in hybrid.hits)
    assert any(hit.dense_rank is not None for hit in hybrid.hits)

    # Informational comparison, printed with -s.
    print(f"\nBM25  top ids: {[(h.rank, h.chunk_id[:8]) for h in bm25.hits]}")
    print(f"DENSE top ids: {[(h.rank, h.chunk_id[:8], round(h.score, 4)) for h in dense.hits]}")
    print(
        f"HYBRID top   : "
        f"{[(h.rank, h.chunk_id[:8], h.bm25_rank, h.dense_rank) for h in hybrid.hits]}"
    )
    overlap = {h.chunk_id for h in bm25.hits} & {h.chunk_id for h in dense.hits}
    print(f"BM25 and dense overlap in top-5: {len(overlap)}")
    print(f"timings dense={dense.timings_ms} hybrid={hybrid.timings_ms}")
