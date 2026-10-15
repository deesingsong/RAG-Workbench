"""Schema migration: a Phase 1 database must gain new columns without data loss."""

from __future__ import annotations

import sqlite3
from pathlib import Path

from sqlalchemy import text

from rag_workbench.config import Settings
from rag_workbench.persistence.database import Database

_LEGACY_INDEX_PROFILES = """
CREATE TABLE index_profiles (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    parser_version TEXT NOT NULL,
    chunker_strategy TEXT NOT NULL,
    chunker_params JSON NOT NULL,
    embedding_model TEXT,
    embedding_dimension INTEGER,
    index_version TEXT NOT NULL,
    created_at DATETIME NOT NULL
)
"""


def _make_legacy_database(path: Path) -> None:
    connection = sqlite3.connect(path)
    try:
        connection.execute(_LEGACY_INDEX_PROFILES)
        connection.execute(
            "INSERT INTO index_profiles VALUES "
            "('p-old', 'legacy', 'hash', '1.0.0', 'paragraph', '{}', NULL, NULL, '1', "
            "'2026-01-01 00:00:00')"
        )
        connection.commit()
    finally:
        connection.close()


def test_phase1_database_is_migrated_additively(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_database(db_path)

    settings = Settings(
        data_dir=tmp_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        _env_file=None,
    )
    database = Database(settings.resolved_database_url)
    database.create_schema()
    try:
        with database.engine.connect() as connection:
            columns = {
                row[1]
                for row in connection.execute(text("PRAGMA table_info(index_profiles)"))
            }
            assert "embedding_params" in columns

            row = connection.execute(
                text("SELECT id, name FROM index_profiles WHERE id = 'p-old'")
            ).one()
            assert row[0] == "p-old"
            assert row[1] == "legacy"  # existing row survived the migration
    finally:
        database.dispose()


def test_migration_is_idempotent(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_database(db_path)
    settings = Settings(
        data_dir=tmp_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        _env_file=None,
    )
    database = Database(settings.resolved_database_url)
    try:
        database.create_schema()
        database.create_schema()  # second run must not fail on the added column
    finally:
        database.dispose()
