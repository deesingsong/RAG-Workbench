"""Integration tests for the HTTP API."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_workbench.api.app import create_app
from rag_workbench.config import Settings


def _upload(client: TestClient, name: str, data: bytes, **form):
    return client.post(
        "/v1/documents",
        files={"file": (name, data)},
        data={k: str(v) for k, v in form.items()},
    )


def test_health_reports_ok(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["document_count"] == 0


def test_upload_then_search_returns_provenance(client, contract_bytes):
    upload = _upload(client, "sample_contract.txt", contract_bytes, collection="contracts")
    assert upload.status_code == 201
    body = upload.json()
    assert body["chunk_count"] > 0
    assert body["state"] == "completed"

    search = client.post(
        "/v1/search",
        json={"query": "termination for convenience", "top_k": 5},
    )
    assert search.status_code == 200
    payload = search.json()
    assert payload["hits"]
    hit = payload["hits"][0]
    assert hit["source_name"] == "sample_contract.txt"
    assert hit["rank"] == 1
    assert payload["config_hash"]


def test_search_without_index_returns_404(client):
    response = client.post("/v1/search", json={"query": "anything"})
    assert response.status_code == 404
    assert response.json()["error"] == "profile_not_found"


def test_search_with_unsearchable_terms_returns_422(client, contract_bytes):
    _upload(client, "sample_contract.txt", contract_bytes)
    response = client.post("/v1/search", json={"query": "***"})
    assert response.status_code == 422
    assert response.json()["error"] == "invalid_query"


def test_unsupported_upload_type_returns_415(client):
    response = _upload(client, "notes.docx", b"not supported")
    assert response.status_code == 415
    assert response.json()["error"] == "unsupported_file_type"


def test_malformed_pdf_returns_400(client):
    response = _upload(client, "broken.pdf", b"definitely not a pdf")
    assert response.status_code == 400
    assert response.json()["error"] == "parsing_error"


def test_empty_upload_is_accepted(client):
    response = _upload(client, "empty.txt", b"")
    assert response.status_code == 201
    assert response.json()["chunk_count"] == 0


def test_document_listing_and_deletion(client, contract_bytes):
    upload = _upload(client, "sample_contract.txt", contract_bytes).json()
    document_id = upload["document_id"]

    listing = client.get("/v1/documents")
    assert listing.status_code == 200
    assert listing.json()["total"] == 1

    detail = client.get(f"/v1/documents/{document_id}")
    assert detail.status_code == 200
    assert detail.json()["chunk_count"] == upload["chunk_count"]

    delete = client.delete(f"/v1/documents/{document_id}")
    assert delete.status_code == 200
    assert delete.json()["document_deleted"] is True

    assert client.get(f"/v1/documents/{document_id}").status_code == 404
    # The profile row remains, so search resolves it and returns no hits.
    no_hits = client.post("/v1/search", json={"query": "confidential"})
    assert no_hits.status_code == 200
    assert no_hits.json()["hits"] == []


def test_profiles_endpoint_lists_index_profiles(client, contract_bytes):
    _upload(client, "sample_contract.txt", contract_bytes, strategy="fixed_token", max_tokens=50)
    response = client.get("/v1/profiles")
    assert response.status_code == 200
    profiles = response.json()["profiles"]
    assert len(profiles) == 1
    assert profiles[0]["chunker_strategy"] == "fixed_token"


def test_invalid_chunking_configuration_returns_422(client, contract_bytes):
    response = _upload(
        client, "sample_contract.txt", contract_bytes, max_tokens=64, overlap_tokens=64
    )
    assert response.status_code == 422
    assert response.json()["error"] == "validation_error"


def test_oversize_upload_returns_413(tmp_path):
    settings = Settings(data_dir=tmp_path, max_upload_mb=1, log_level="WARNING", _env_file=None)
    app = create_app(settings)
    with TestClient(app) as client:
        response = _upload(client, "big.txt", b"x" * (2 * 1024 * 1024))
    assert response.status_code == 413


def test_ingestion_job_is_retrievable(client, contract_bytes):
    upload = _upload(client, "sample_contract.txt", contract_bytes).json()
    response = client.get(f"/v1/ingestion-jobs/{upload['job_id']}")
    assert response.status_code == 200
    body = response.json()
    assert body["state"] == "completed"
    assert body["timings_ms"]["total"] >= 0
