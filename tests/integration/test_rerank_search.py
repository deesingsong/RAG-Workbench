"""End-to-end reranking across bm25/dense/hybrid with a deterministic fake reranker.

No model is downloaded: a fake cross-encoder is injected while SQLite, the real
Qdrant local store and RRF fusion are the real implementations. The reranker is
never allowed to add candidates, only reorder them.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.domain.errors import RerankerUnavailableError, ValidationError
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService
from tests.fakes import FakeEmbeddingProvider, FakeReranker

LEGAL_PROFILE = IndexProfileSpec(
    name="legal-clause", chunker=ChunkerConfig(strategy="clause_aware", max_tokens=120)
)


def _ingest(service: IngestionService, data: bytes, profile: IndexProfileSpec):
    return service.ingest_bytes(
        data, source_name="legal_contract.txt", collection="contracts", profile=profile
    )


def _service(
    settings: Settings,
    database: Database,
    reranker: FakeReranker,
    *,
    provider: FakeEmbeddingProvider | None = None,
) -> SearchService:
    kwargs = {"reranker_factory": lambda _model: reranker}
    if provider is not None:
        kwargs["embedding_provider_factory"] = lambda _spec: provider
    return SearchService(database, settings, **kwargs)


def test_bm25_rerank_reports_diagnostics_and_scores(
    settings: Settings,
    database: Database,
    legal_contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings)
    result = _ingest(service, legal_contract_bytes, LEGAL_PROFILE)
    assert result.chunk_count > 0

    reranker = FakeReranker()
    response = _service(settings, database, reranker).search(
        query="termination for convenience",
        profile_id=LEGAL_PROFILE.key,
        mode="bm25",
        top_k=3,
        rerank=True,
        rerank_candidates=10,
    )

    assert response.rerank is True
    assert response.rerank_candidates == 10
    assert response.reranker is not None
    assert response.reranker["model_id"] == "fake/cross-encoder"
    assert response.timings_ms["rerank_init"] >= 0
    assert response.timings_ms["rerank_inference"] >= 0
    assert [hit.rank for hit in response.hits] == list(range(1, len(response.hits) + 1))
    assert response.hits
    assert all(hit.rerank_score is not None for hit in response.hits)
    assert all(hit.retrieval_rank is not None for hit in response.hits)
    assert all(hit.retrieval_score_kind == "bm25" for hit in response.hits)
    # The reranker may only reorder the candidates it was given.
    assert all(hit.retrieval_rank <= 10 for hit in response.hits)
    assert reranker.ensure_calls == 1


def test_reranking_reorders_candidates_and_records_movement(
    settings: Settings,
    database: Database,
    legal_contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings)
    _ingest(service, legal_contract_bytes, LEGAL_PROFILE)

    baseline = SearchService(database, settings).search(
        query="confidential information",
        profile_id=LEGAL_PROFILE.key,
        mode="bm25",
        top_k=5,
    )
    assert baseline.hits
    assert baseline.hits[0].section_number == "1.1"

    # Boost a clause that baseline retrieval ranks below the top.
    reranker = FakeReranker(boosts={"return or destroy": 100.0})
    response = _service(settings, database, reranker).search(
        query="confidential information",
        profile_id=LEGAL_PROFILE.key,
        mode="bm25",
        top_k=5,
        rerank=True,
        rerank_candidates=50,
    )

    top = response.hits[0]
    assert top.section_number == "3.3(b)"
    assert top.rank == 1
    assert top.retrieval_rank is not None and top.retrieval_rank > 1
    assert top.rank_movement == top.retrieval_rank - 1
    assert top.rank_movement > 0


def test_rerank_ties_preserve_original_order(
    settings: Settings,
    database: Database,
) -> None:
    from rag_workbench.domain.schemas import SearchHit

    def hit(order: int) -> SearchHit:
        return SearchHit(
            rank=order,
            chunk_id=f"chunk-{order}",
            document_id="doc-1",
            source_name="legal_contract.txt",
            collection="contracts",
            chunk_order=order,
            text=f"passage number {order}",
            score=-1.0 * order,
            score_kind="bm25",
        )

    hits = [hit(1), hit(2), hit(3)]
    service = _service(settings, database, FakeReranker())
    results, timings, meta = service._apply_rerank(
        query="zzzznomatch", hits=hits, top_k=3, reranker_model=None
    )
    # Equal (zero) rerank scores fall back to the original retrieval order.
    assert [h.chunk_id for h in results] == ["chunk-1", "chunk-2", "chunk-3"]
    assert [h.rank for h in results] == [1, 2, 3]
    assert all(h.rank_movement == 0 for h in results)
    assert timings["rerank_inference"] >= 0
    assert meta["model_id"] == "fake/cross-encoder"


def test_rerank_candidates_must_cover_top_k(
    settings: Settings,
    database: Database,
    legal_contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings)
    _ingest(service, legal_contract_bytes, LEGAL_PROFILE)
    with pytest.raises(ValidationError):
        _service(settings, database, FakeReranker()).search(
            query="termination",
            profile_id=LEGAL_PROFILE.key,
            top_k=5,
            rerank=True,
            rerank_candidates=2,
        )


def test_missing_reranker_dependency_surfaces_clearly(
    settings: Settings,
    database: Database,
    legal_contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings)
    _ingest(service, legal_contract_bytes, LEGAL_PROFILE)

    def _boom(_model: str | None):
        raise RerankerUnavailableError("simulated missing model")

    search = SearchService(database, settings, reranker_factory=_boom)
    with pytest.raises(RerankerUnavailableError):
        search.search(
            query="termination",
            profile_id=LEGAL_PROFILE.key,
            top_k=3,
            rerank=True,
            rerank_candidates=8,
        )


def test_truncation_diagnostics_are_reported(
    settings: Settings,
    database: Database,
    legal_contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings)
    _ingest(service, legal_contract_bytes, LEGAL_PROFILE)

    reranker = FakeReranker(max_length=5)
    response = _service(settings, database, reranker).search(
        query="termination",
        profile_id=LEGAL_PROFILE.key,
        top_k=3,
        rerank=True,
        rerank_candidates=8,
    )
    assert response.reranker is not None
    assert response.reranker["truncated_pairs"] > 0
    assert response.reranker["diagnostics"]


def test_rerank_with_no_hits_is_safe(
    settings: Settings,
    database: Database,
    legal_contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings)
    _ingest(service, legal_contract_bytes, LEGAL_PROFILE)
    response = _service(settings, database, FakeReranker()).search(
        query="zzzznomatch",
        profile_id=LEGAL_PROFILE.key,
        top_k=3,
        rerank=True,
        rerank_candidates=8,
    )
    assert response.hits == []
    assert response.rerank is True


def test_dense_and_hybrid_rerank_preserve_component_scores(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
    fake_provider: FakeEmbeddingProvider,
) -> None:
    service = IngestionService(database, settings, vector_indexer=vector_indexer)
    _ingest(service, contract_bytes, dense_profile)

    for mode, kind in (("dense", "cosine"), ("hybrid", "rrf")):
        response = _service(
            settings, database, FakeReranker(), provider=fake_provider
        ).search(
            query="termination for convenience",
            profile_id=dense_profile.key,
            mode=mode,
            top_k=3,
            rerank=True,
            rerank_candidates=10,
        )
        assert response.hits
        assert all(hit.rerank_score is not None for hit in response.hits)
        assert all(hit.retrieval_score_kind == kind for hit in response.hits)
        # The reranker score is distinct from the retrieval score.
        assert all(hit.score_kind == kind for hit in response.hits)
        assert [hit.rank for hit in response.hits] == [1, 2, 3]


def test_api_search_exposes_reranking(
    client: TestClient,
    contract_bytes: bytes,
) -> None:
    app = client.app
    app.state.search_service = SearchService(
        app.state.database,
        app.state.settings,
        reranker_factory=lambda _model: FakeReranker(),
    )
    upload = client.post(
        "/v1/documents",
        files={"file": ("sample_contract.txt", contract_bytes, "text/plain")},
        data={"collection": "contracts"},
    )
    assert upload.status_code == 201, upload.text

    response = client.post(
        "/v1/search",
        json={
            "query": "termination for convenience",
            "mode": "bm25",
            "top_k": 3,
            "rerank": True,
            "rerank_candidates": 8,
        },
    )
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["rerank"] is True
    assert body["rerank_candidates"] == 8
    assert body["reranker"]["model_id"] == "fake/cross-encoder"
    assert body["hits"]
    assert all(hit["retrieval_rank"] is not None for hit in body["hits"])
    assert all(hit["rerank_score"] is not None for hit in body["hits"])


def test_api_rejects_rerank_candidates_below_top_k(
    client: TestClient,
    contract_bytes: bytes,
) -> None:
    client.post(
        "/v1/documents",
        files={"file": ("sample_contract.txt", contract_bytes, "text/plain")},
        data={"collection": "contracts"},
    )
    response = client.post(
        "/v1/search",
        json={"query": "termination", "top_k": 5, "rerank": True, "rerank_candidates": 2},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_candidates"
