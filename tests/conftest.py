"""Shared pytest fixtures: isolated settings, database and API client."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_workbench.api.app import create_app
from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.vector_store import close_vector_stores
from rag_workbench.retrieval.service import SearchService
from tests.fakes import FakeEmbeddingProvider

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture(autouse=True)
def _vector_store_guard() -> Iterator[None]:
    """Release Qdrant local-mode locks after every test."""

    yield
    close_vector_stores()


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Settings pointing at an isolated SQLite file, ignoring any local .env."""

    return Settings(data_dir=tmp_path, log_level="WARNING", _env_file=None)


@pytest.fixture
def database(settings: Settings):
    db = Database(settings.resolved_database_url)
    db.create_schema()
    try:
        yield db
    finally:
        db.dispose()


@pytest.fixture
def client(settings: Settings):
    app = create_app(settings)
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def fake_provider() -> FakeEmbeddingProvider:
    return FakeEmbeddingProvider(dimension=64)


@pytest.fixture
def dense_profile() -> IndexProfileSpec:
    """A dense profile whose declared dimension matches the fake provider."""

    return IndexProfileSpec(
        name="dense-test",
        chunker=ChunkerConfig(),
        embedding_model="fake/hash-bow",
        embedding_dimension=64,
        embedding_revision="v1",
    )


@pytest.fixture
def vector_indexer(settings: Settings, database: Database, fake_provider: FakeEmbeddingProvider):
    return VectorIndexer(
        database,
        settings,
        embedding_provider_factory=lambda _spec: fake_provider,
    )


@pytest.fixture
def dense_client(settings: Settings, fake_provider: FakeEmbeddingProvider):
    """API client whose dense services use the deterministic fake provider.

    Settings are tuned so that profiles created through the API declare the same
    dimension as the fake provider (otherwise the real default model's 384 would
    mismatch the fake's vectors).
    """

    tuned = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        embedding_model=fake_provider.model_id,
        embedding_dimension=fake_provider.dimension,
        _env_file=None,
    )
    app = create_app(tuned)
    indexer = VectorIndexer(
        app.state.database,
        tuned,
        embedding_provider_factory=lambda _spec: fake_provider,
    )
    app.state.vector_indexer = indexer
    app.state.ingestion_service = IngestionService(
        app.state.database, tuned, vector_indexer=indexer
    )
    app.state.search_service = SearchService(
        app.state.database,
        tuned,
        embedding_provider_factory=lambda _spec: fake_provider,
    )
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture
def contract_path() -> Path:
    return FIXTURES_DIR / "sample_contract.txt"


@pytest.fixture
def contract_bytes(contract_path: Path) -> bytes:
    return contract_path.read_bytes()


@pytest.fixture
def default_profile() -> IndexProfileSpec:
    return IndexProfileSpec(name="paragraph-220", chunker=ChunkerConfig())
