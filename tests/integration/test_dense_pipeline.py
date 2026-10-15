"""End-to-end dense and hybrid retrieval using the fake provider + real Qdrant.

No model is downloaded: a deterministic fake embedding provider is injected while
SQLite, the Qdrant local store and RRF fusion are the real implementations.
"""

from __future__ import annotations

import uuid

import pytest

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.domain.errors import (
    DenseUnavailableError,
    ProfileNotDenseError,
    VectorIndexNotBuiltError,
)
from rag_workbench.domain.schemas import VectorPoint
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.vector_store import get_vector_store
from rag_workbench.retrieval.service import SearchService
from tests.fakes import FakeEmbeddingProvider


def _boom(_spec: IndexProfileSpec) -> FakeEmbeddingProvider:
    raise DenseUnavailableError("simulated missing model")


def _ingest(service: IngestionService, data: bytes, profile: IndexProfileSpec):
    return service.ingest_bytes(
        data, source_name="sample_contract.txt", collection="contracts", profile=profile
    )


def _ingest_with(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    data: bytes,
    profile: IndexProfileSpec,
):
    service = IngestionService(database, settings, vector_indexer=vector_indexer)
    return _ingest(service, data, profile)


def _search(
    settings: Settings, database: Database, provider: FakeEmbeddingProvider
) -> SearchService:
    return SearchService(
        database, settings, embedding_provider_factory=lambda _spec: provider
    )


def test_dense_ingest_indexes_vectors_and_is_idempotent(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings, vector_indexer=vector_indexer)
    result = _ingest(service, contract_bytes, dense_profile)

    assert result.state == "completed"
    assert result.chunk_count > 0
    status = vector_indexer.reconcile_profile(dense_profile.key)
    assert status.in_sync
    assert status.point_count == result.chunk_count

    again = _ingest(service, contract_bytes, dense_profile)
    assert again.is_duplicate
    assert vector_indexer.reconcile_profile(dense_profile.key).point_count == result.chunk_count


def test_dense_search_returns_ranked_provenance(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
    fake_provider: FakeEmbeddingProvider,
) -> None:
    _ingest_with(settings, database, vector_indexer, contract_bytes, dense_profile)
    response = _search(settings, database, fake_provider).search(
        query="termination for convenience",
        profile_id=dense_profile.key,
        mode="dense",
        top_k=3,
    )

    assert response.hits
    assert [hit.rank for hit in response.hits] == list(range(1, len(response.hits) + 1))
    assert all(hit.score_kind == "cosine" for hit in response.hits)
    assert all(hit.source_name == "sample_contract.txt" for hit in response.hits)
    assert all(hit.collection == "contracts" for hit in response.hits)
    scores = [hit.score for hit in response.hits]
    assert scores == sorted(scores, reverse=True)  # cosine: larger is better
    assert response.mode == "dense"
    assert response.candidates is not None


def test_hybrid_preserves_component_ranks_and_is_deterministic(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
    fake_provider: FakeEmbeddingProvider,
) -> None:
    _ingest_with(settings, database, vector_indexer, contract_bytes, dense_profile)
    service = _search(settings, database, fake_provider)
    first = service.search(
        query="termination for convenience", profile_id=dense_profile.key, mode="hybrid", top_k=5
    )
    second = service.search(
        query="termination for convenience", profile_id=dense_profile.key, mode="hybrid", top_k=5
    )

    assert first.hits
    assert all(hit.score_kind == "rrf" for hit in first.hits)
    assert all(hit.fused_score == hit.score for hit in first.hits)
    assert any(hit.bm25_rank is not None for hit in first.hits)
    assert [h.chunk_id for h in first.hits] == [h.chunk_id for h in second.hits]


def test_dense_mode_rejects_a_lexical_profile(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    default_profile: IndexProfileSpec,
    contract_bytes: bytes,
    fake_provider: FakeEmbeddingProvider,
) -> None:
    _ingest_with(settings, database, vector_indexer, contract_bytes, default_profile)
    with pytest.raises(ProfileNotDenseError):
        _search(settings, database, fake_provider).search(
            query="termination", profile_id=default_profile.key, mode="dense"
        )


def test_reconcile_detects_and_repairs_missing_and_orphans(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
) -> None:
    _ingest_with(settings, database, vector_indexer, contract_bytes, dense_profile)
    collection = dense_profile.vector_collection
    store = get_vector_store(settings.vector_store_path)
    ids = sorted(store.list_ids(collection))
    store.delete_ids(collection, ids[:1])
    store.upsert(
        collection,
        [
            VectorPoint(
                id=str(uuid.uuid4()),
                vector=[0.0] * 64,
                payload={"collection": "contracts", "profile_id": dense_profile.key},
            )
        ],
    )

    broken = vector_indexer.reconcile_profile(dense_profile.key)
    assert broken.missing_points == 1
    assert broken.orphan_points == 1
    assert broken.in_sync is False

    repaired = vector_indexer.reconcile_profile(dense_profile.key, repair=True)
    assert repaired.in_sync is True
    assert repaired.point_count == repaired.chunk_count


def test_rebuild_restores_a_dropped_index(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
    fake_provider: FakeEmbeddingProvider,
) -> None:
    result = _ingest_with(settings, database, vector_indexer, contract_bytes, dense_profile)
    store = get_vector_store(settings.vector_store_path)
    store.drop(dense_profile.vector_collection)

    with pytest.raises(VectorIndexNotBuiltError):
        _search(settings, database, fake_provider).search(
            query="termination", profile_id=dense_profile.key, mode="dense"
        )

    rebuilt = vector_indexer.rebuild_profile(dense_profile.key)
    assert rebuilt.points_written == result.chunk_count
    assert rebuilt.chunks == result.chunk_count

    response = _search(settings, database, fake_provider).search(
        query="termination", profile_id=dense_profile.key, mode="dense", top_k=3
    )
    assert response.hits


def test_deletion_propagates_to_the_vector_index(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings, vector_indexer=vector_indexer)
    _ingest(service, contract_bytes, dense_profile)

    documents, _ = service.list_documents()
    document_id = documents[0].document_id
    assert vector_indexer.reconcile_profile(dense_profile.key).point_count > 0

    service.delete_document(document_id, profile_id=dense_profile.key)

    status = vector_indexer.reconcile_profile(dense_profile.key)
    assert status.point_count == 0
    assert status.chunk_count == 0
    assert status.in_sync


def test_vector_failure_marks_partial_and_rebuild_recovers(
    settings: Settings,
    database: Database,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
    fake_provider: FakeEmbeddingProvider,
) -> None:
    failing = VectorIndexer(database, settings, embedding_provider_factory=_boom)
    service = IngestionService(database, settings, vector_indexer=failing)
    result = _ingest(service, contract_bytes, dense_profile)
    assert result.state == "partial"
    assert any("vector indexing failed" in note for note in result.diagnostics)

    # Chunks are durable, so BM25 still answers ...
    bm25 = SearchService(database, settings).search(
        query="termination", profile_id=dense_profile.key, mode="bm25"
    )
    assert bm25.hits
    # ... but the dense index was never built.
    with pytest.raises(VectorIndexNotBuiltError):
        _search(settings, database, fake_provider).search(
            query="termination", profile_id=dense_profile.key, mode="dense"
        )

    good = VectorIndexer(database, settings, embedding_provider_factory=lambda _s: fake_provider)
    rebuilt = good.rebuild_profile(dense_profile.key)
    assert rebuilt.points_written == result.chunk_count
    assert _search(settings, database, fake_provider).search(
        query="termination", profile_id=dense_profile.key, mode="dense"
    ).hits


def test_partial_ingest_does_not_duplicate_chunks(
    settings: Settings,
    database: Database,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
) -> None:
    service = IngestionService(
        database,
        settings,
        vector_indexer=VectorIndexer(database, settings, embedding_provider_factory=_boom),
    )
    first = _ingest(service, contract_bytes, dense_profile)
    second = _ingest(service, contract_bytes, dense_profile)
    assert first.chunk_count == second.chunk_count
    assert second.is_duplicate


def test_dense_and_lexical_profiles_isolate_chunks(
    settings: Settings,
    database: Database,
    vector_indexer: VectorIndexer,
    dense_profile: IndexProfileSpec,
    contract_bytes: bytes,
) -> None:
    service = IngestionService(database, settings, vector_indexer=vector_indexer)
    lexical = IndexProfileSpec(name="lex", chunker=ChunkerConfig())
    _ingest(service, contract_bytes, dense_profile)
    _ingest(service, contract_bytes, lexical)

    profiles = service.list_profiles()
    counts = {profile.profile_id: profile.chunk_count for profile in profiles}
    assert counts[dense_profile.key] == counts[lexical.key]
    assert dense_profile.key != lexical.key
