"""Shared pytest fixtures: isolated settings, database and API client."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_workbench.api.app import create_app
from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.persistence.database import Database

FIXTURES_DIR = Path(__file__).parent / "fixtures"


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
def contract_path() -> Path:
    return FIXTURES_DIR / "sample_contract.txt"


@pytest.fixture
def contract_bytes(contract_path: Path) -> bytes:
    return contract_path.read_bytes()


@pytest.fixture
def default_profile() -> IndexProfileSpec:
    return IndexProfileSpec(name="paragraph-220", chunker=ChunkerConfig())
