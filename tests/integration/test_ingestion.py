"""Integration tests: ingestion, idempotency, provenance, deletion, isolation."""

from __future__ import annotations

import pytest
from sqlalchemy import func, select, text

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.domain.errors import FileTooLargeError, UnsupportedFileTypeError
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Chunk
from rag_workbench.retrieval.service import SearchService


@pytest.fixture
def service(database: Database, settings: Settings) -> IngestionService:
    return IngestionService(database, settings)


def _paragraph_profile() -> IndexProfileSpec:
    return IndexProfileSpec(name="paragraph-60", chunker=ChunkerConfig(max_tokens=60))


def test_ingest_contract_and_search_exact_clause(service, database, contract_path):
    result = service.ingest_path(
        contract_path, collection="contracts", profile=_paragraph_profile()
    )

    assert result.state == "completed"
    assert not result.is_duplicate
    assert result.chunk_count > 0

    search = SearchService(database, service._settings)
    response = search.search(query="termination for convenience", top_k=5)

    assert response.hits, "expected at least one hit"
    top = response.hits[0]
    assert top.rank == 1
    assert top.source_name == "sample_contract.txt"
    assert top.collection == "contracts"
    # The exact phrase "termination for convenience" appears in the indexed
    # section heading, which is why this chunk ranks first.
    assert top.section_path is not None
    assert "termination for convenience" in top.section_path.lower()
    assert "convenience" in top.text.lower()


def test_reingest_same_profile_is_idempotent(service, database, contract_bytes):
    profile = _paragraph_profile()
    first = service.ingest_bytes(
        contract_bytes, source_name="c.txt", collection="default", profile=profile
    )
    second = service.ingest_bytes(
        contract_bytes, source_name="c.txt", collection="default", profile=profile
    )

    assert not first.is_duplicate
    assert second.is_duplicate
    assert second.chunk_count == first.chunk_count

    with database.session() as session:
        count = session.execute(select(func.count(Chunk.id))).scalar()
    assert count == first.chunk_count


def test_two_profiles_are_isolated(service, database, contract_bytes):
    small = IndexProfileSpec(
        name="p-small",
        chunker=ChunkerConfig(strategy="fixed_token", max_tokens=30, overlap_tokens=5),
    )
    large = IndexProfileSpec(
        name="p-large",
        chunker=ChunkerConfig(strategy="fixed_token", max_tokens=200, overlap_tokens=20),
    )

    service.ingest_bytes(
        contract_bytes, source_name="c.txt", collection="default", profile=small
    )
    service.ingest_bytes(
        contract_bytes, source_name="c.txt", collection="default", profile=large
    )

    with database.session() as session:
        small_ids = set(
            session.execute(
                select(Chunk.chunk_id).where(Chunk.profile_id == small.key)
            ).scalars()
        )
        large_ids = set(
            session.execute(
                select(Chunk.chunk_id).where(Chunk.profile_id == large.key)
            ).scalars()
        )

    assert small_ids and large_ids
    # Chunk IDs are profile-scoped, so the two indexes share no rows.
    assert small_ids.isdisjoint(large_ids)
    assert len(small_ids) != len(large_ids)


def test_deletion_removes_chunks_and_fts_entries(service, database, contract_path):
    result = service.ingest_path(
        contract_path, collection="default", profile=_paragraph_profile()
    )
    search = SearchService(database, service._settings)
    assert search.search(query="confidential", top_k=5).hits

    deleted = service.delete_document(result.document_id)
    assert deleted.chunks_deleted == result.chunk_count
    assert deleted.document_deleted

    # FTS must be synchronized by the delete trigger, not just the relational rows.
    assert search.search(query="confidential", top_k=5).hits == []
    with database.session() as session:
        relational = session.execute(select(func.count(Chunk.id))).scalar()
        indexed = session.execute(text("SELECT count(*) FROM chunk_fts")).scalar()
    assert relational == 0
    assert indexed == 0


def test_same_bytes_different_collections_are_distinct_documents(service, database, contract_bytes):
    profile = _paragraph_profile()
    a = service.ingest_bytes(
        contract_bytes, source_name="c.txt", collection="alpha", profile=profile
    )
    b = service.ingest_bytes(
        contract_bytes, source_name="c.txt", collection="beta", profile=profile
    )
    assert a.document_id != b.document_id
    assert not b.is_duplicate


def test_oversize_upload_is_rejected(database, settings, contract_bytes):
    tiny = Settings(data_dir=settings.data_dir, max_upload_mb=1, _env_file=None)
    service = IngestionService(database, tiny)
    with pytest.raises(FileTooLargeError):
        service.ingest_bytes(
            b"x" * (2 * 1024 * 1024),
            source_name="big.txt",
            collection="default",
            profile=_paragraph_profile(),
        )


def test_unsupported_type_is_rejected(service):
    with pytest.raises(UnsupportedFileTypeError):
        service.ingest_bytes(
            b"data", source_name="notes.docx", collection="default", profile=_paragraph_profile()
        )


def test_empty_file_is_indexed_with_no_chunks(service, database):
    result = service.ingest_bytes(
        b"", source_name="empty.txt", collection="default", profile=_paragraph_profile()
    )
    assert result.chunk_count == 0
    assert result.state == "completed"
    assert any("no extractable text" in note for note in result.diagnostics)


def test_failed_job_is_recorded(service, database):
    from rag_workbench.domain.errors import ParsingError

    with pytest.raises(ParsingError):
        service.ingest_bytes(
            b"not a real pdf",
            source_name="broken.pdf",
            collection="default",
            profile=_paragraph_profile(),
        )

    with database.session() as session:
        job = session.execute(
            text("SELECT state, error_code FROM ingestion_jobs ORDER BY started_at DESC LIMIT 1")
        ).one()
    assert job.state == "failed"
    assert job.error_code == "parsing_error"
