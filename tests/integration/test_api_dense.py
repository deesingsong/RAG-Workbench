"""API-level dense/hybrid behavior and error mapping (fake provider, real Qdrant)."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_workbench.api.app import create_app
from rag_workbench.domain.errors import DenseUnavailableError
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.persistence.vector_store import get_vector_store
from rag_workbench.retrieval.service import SearchService


def _upload(client: TestClient, data: bytes, **form: str) -> dict:
    response = client.post(
        "/v1/documents",
        files={"file": ("sample_contract.txt", data, "text/plain")},
        data={"collection": "contracts", **form},
    )
    assert response.status_code == 201, response.text
    return response.json()


def test_bm25_api_is_backward_compatible(client: TestClient, contract_bytes: bytes) -> None:
    body = _upload(client, contract_bytes)
    assert body["state"] == "completed"

    response = client.post(
        "/v1/search",
        json={"query": "termination for convenience", "profile_id": body["profile_id"]},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "bm25"
    assert payload["candidates"] is None
    assert payload["hits"]
    assert all(hit["score_kind"] == "bm25" for hit in payload["hits"])
    assert payload["timings_ms"]["lexical"] >= 0


def test_dense_ingest_and_search_via_api(dense_client: TestClient, contract_bytes: bytes) -> None:
    body = _upload(dense_client, contract_bytes, embed="true")
    assert body["state"] == "completed"
    profile_id = body["profile_id"]

    profiles = dense_client.get("/v1/profiles").json()["profiles"]
    profile = next(item for item in profiles if item["profile_id"] == profile_id)
    assert profile["embedding_model"] == "fake/hash-bow"
    assert profile["embedding_dimension"] == 64
    assert profile["vector_collection"] == f"rw_{profile_id}"

    response = dense_client.post(
        "/v1/search",
        json={"query": "termination for convenience", "profile_id": profile_id, "mode": "dense"},
    )
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["mode"] == "dense"
    assert payload["hits"]
    assert all(hit["score_kind"] == "cosine" for hit in payload["hits"])
    assert all(hit["source_name"] == "sample_contract.txt" for hit in payload["hits"])


def test_hybrid_via_api_preserves_component_ranks(
    dense_client: TestClient, contract_bytes: bytes
) -> None:
    body = _upload(dense_client, contract_bytes, embed="true")
    response = dense_client.post(
        "/v1/search",
        json={
            "query": "termination for convenience",
            "profile_id": body["profile_id"],
            "mode": "hybrid",
        },
    )
    assert response.status_code == 200, response.text
    hits = response.json()["hits"]
    assert hits
    assert all(hit["score_kind"] == "rrf" for hit in hits)
    assert all(hit["fused_score"] is not None for hit in hits)
    assert any(hit["dense_rank"] is not None for hit in hits)


def test_dense_on_lexical_profile_is_422(dense_client: TestClient, contract_bytes: bytes) -> None:
    body = _upload(dense_client, contract_bytes)  # no embed
    response = dense_client.post(
        "/v1/search",
        json={"query": "termination", "profile_id": body["profile_id"], "mode": "dense"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "profile_not_dense"


def test_unsupported_mode_is_422(client: TestClient, contract_bytes: bytes) -> None:
    body = _upload(client, contract_bytes)
    response = client.post(
        "/v1/search",
        json={"query": "termination", "profile_id": body["profile_id"], "mode": "bm42"},
    )
    assert response.status_code == 422
    assert response.json()["error"] == "unsupported_mode"


def test_missing_vector_index_reports_not_built(
    contract_bytes: bytes, settings, fake_provider
) -> None:
    def boom(_spec):
        raise DenseUnavailableError("no model")

    app = create_app(settings)
    indexer = VectorIndexer(app.state.database, settings, embedding_provider_factory=boom)
    app.state.vector_indexer = indexer
    app.state.ingestion_service = IngestionService(
        app.state.database, settings, vector_indexer=indexer
    )
    app.state.search_service = SearchService(
        app.state.database, settings, embedding_provider_factory=lambda _s: fake_provider
    )
    with TestClient(app) as client:
        body = _upload(client, contract_bytes, embed="true")
        assert body["state"] == "partial"
        assert any("vector indexing failed" in note for note in body["diagnostics"])
        response = client.post(
            "/v1/search",
            json={"query": "termination", "profile_id": body["profile_id"], "mode": "dense"},
        )
        assert response.status_code == 503
        assert response.json()["error"] == "vector_index_not_built"


def test_dense_unavailable_maps_to_503(dense_client: TestClient, contract_bytes: bytes) -> None:
    body = _upload(dense_client, contract_bytes, embed="true")  # vectors built with fake provider
    dense_client.app.state.search_service = SearchService(
        dense_client.app.state.database,
        dense_client.app.state.settings,
        embedding_provider_factory=lambda _spec: _raise(),
    )
    response = dense_client.post(
        "/v1/search",
        json={"query": "termination", "profile_id": body["profile_id"], "mode": "dense"},
    )
    assert response.status_code == 503
    assert response.json()["error"] == "dense_unavailable"


def test_vector_status_endpoint(dense_client: TestClient, contract_bytes: bytes) -> None:
    body = _upload(dense_client, contract_bytes, embed="true")
    response = dense_client.get("/v1/vectors/status", params={"profile_id": body["profile_id"]})
    assert response.status_code == 200
    statuses = response.json()["statuses"]
    assert len(statuses) == 1
    assert statuses[0]["in_sync"] is True
    assert statuses[0]["point_count"] == body["chunk_count"]


def test_vector_rebuild_endpoint(dense_client: TestClient, contract_bytes: bytes) -> None:
    body = _upload(dense_client, contract_bytes, embed="true")
    profile_id = body["profile_id"]

    # Wipe the collection, then rebuild purely from SQLite.
    store = get_vector_store(dense_client.app.state.settings.vector_store_path)
    store.drop(f"rw_{profile_id}")

    response = dense_client.post("/v1/vectors/rebuild", json={"profile_id": profile_id})
    assert response.status_code == 200, response.text
    result = response.json()["results"][0]
    assert result["points_written"] == body["chunk_count"]

    search = dense_client.post(
        "/v1/search",
        json={"query": "termination", "profile_id": profile_id, "mode": "dense"},
    )
    assert search.status_code == 200
    assert search.json()["hits"]


def _raise():
    raise DenseUnavailableError("no model")
