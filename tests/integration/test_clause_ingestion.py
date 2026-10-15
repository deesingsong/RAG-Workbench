"""Clause-aware ingestion: persistence, provenance, idempotency and isolation."""

from __future__ import annotations

from fastapi.testclient import TestClient

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService

CLAUSE_PROFILE = IndexProfileSpec(
    name="legal-clause", chunker=ChunkerConfig(strategy="clause_aware", max_tokens=120)
)
PARAGRAPH_PROFILE = IndexProfileSpec(name="paragraph", chunker=ChunkerConfig(strategy="paragraph"))


def _ingest(service: IngestionService, data: bytes, profile: IndexProfileSpec):
    return service.ingest_bytes(
        data,
        source_name="legal_contract.txt",
        collection="contracts",
        profile=profile,
    )


def test_clause_ingestion_persists_provenance(
    settings: Settings, legal_contract_bytes: bytes, database: Database
) -> None:
    service = IngestionService(database, settings)
    result = _ingest(service, legal_contract_bytes, CLAUSE_PROFILE)
    assert result.state == "completed"
    assert result.chunk_count > 0

    chunks, total = service.list_chunks(CLAUSE_PROFILE.key, limit=1000)
    assert total == result.chunk_count

    by_number = {c.section_number: c for c in chunks if c.section_number}
    assert by_number["3.1"].section_title == "Termination for Convenience"
    assert by_number["3.1"].parent_clause == "ARTICLE III"
    assert by_number["3.3(a)"].section_level == 3
    assert any(c.section_title == '"Confidential Information"' for c in chunks)
    # Diagnostics travel with the chunk they describe.
    assert all(isinstance(c.diagnostics, list) for c in chunks)


def test_clause_ingestion_is_idempotent(
    settings: Settings, legal_contract_bytes: bytes, database: Database
) -> None:
    service = IngestionService(database, settings)
    first = _ingest(service, legal_contract_bytes, CLAUSE_PROFILE)
    second = _ingest(service, legal_contract_bytes, CLAUSE_PROFILE)
    assert second.is_duplicate
    assert second.chunk_count == first.chunk_count

    chunks, total = service.list_chunks(CLAUSE_PROFILE.key, limit=1000)
    assert total == first.chunk_count
    assert len({c.chunk_id for c in chunks}) == len(chunks)


def test_clause_and_paragraph_profiles_are_isolated(
    settings: Settings, legal_contract_bytes: bytes, database: Database
) -> None:
    service = IngestionService(database, settings)
    clause = _ingest(service, legal_contract_bytes, CLAUSE_PROFILE)
    paragraph = _ingest(service, legal_contract_bytes, PARAGRAPH_PROFILE)
    assert CLAUSE_PROFILE.key != PARAGRAPH_PROFILE.key

    clause_chunks, _ = service.list_chunks(CLAUSE_PROFILE.key, limit=1000)
    paragraph_chunks, _ = service.list_chunks(PARAGRAPH_PROFILE.key, limit=1000)
    # Clause-aware chunks carry section numbers; the paragraph profile does not.
    assert any(c.section_number for c in clause_chunks)
    assert not any(c.section_number for c in paragraph_chunks)
    assert clause.chunk_count > 0 and paragraph.chunk_count > 0


def test_section_metadata_is_searchable_as_provenance(
    settings: Settings, legal_contract_bytes: bytes, database: Database
) -> None:
    service = IngestionService(database, settings)
    _ingest(service, legal_contract_bytes, CLAUSE_PROFILE)

    response = SearchService(database, settings).search(
        query="confidential information",
        profile_id=CLAUSE_PROFILE.key,
        mode="bm25",
        top_k=3,
    )
    assert response.hits
    assert any(hit.section_number for hit in response.hits)
    assert all(hit.section_path for hit in response.hits)


def test_deleting_a_document_removes_clause_chunks(
    settings: Settings, legal_contract_bytes: bytes, database: Database
) -> None:
    service = IngestionService(database, settings)
    _ingest(service, legal_contract_bytes, CLAUSE_PROFILE)
    documents, _ = service.list_documents()
    assert documents

    service.delete_document(documents[0].document_id, profile_id=CLAUSE_PROFILE.key)
    _, total = service.list_chunks(CLAUSE_PROFILE.key, limit=1000)
    assert total == 0


def test_api_chunks_endpoint_returns_clause_metadata(
    client: TestClient, legal_contract_bytes: bytes
) -> None:
    upload = client.post(
        "/v1/documents",
        files={"file": ("legal_contract.txt", legal_contract_bytes, "text/plain")},
        data={"collection": "contracts", "strategy": "clause_aware", "max_tokens": "120"},
    )
    assert upload.status_code == 201, upload.text
    body = upload.json()
    assert body["chunk_count"] > 0
    profile_id = body["profile_id"]

    response = client.get(f"/v1/profiles/{profile_id}/chunks", params={"limit": 500})
    assert response.status_code == 200, response.text
    payload = response.json()
    assert payload["total"] == body["chunk_count"]
    numbers = {chunk["section_number"] for chunk in payload["chunks"]}
    assert "3.1" in numbers
    sample = next(c for c in payload["chunks"] if c["section_number"] == "3.1")
    assert sample["section_title"] == "Termination for Convenience"
    assert sample["parent_clause"] == "ARTICLE III"

    missing = client.get("/v1/profiles/does-not-exist/chunks")
    assert missing.status_code == 404
