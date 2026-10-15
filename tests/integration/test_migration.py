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

_LEGACY_CHUNKS = """
CREATE TABLE chunks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id TEXT NOT NULL UNIQUE,
    document_id TEXT NOT NULL,
    profile_id TEXT NOT NULL,
    chunk_order INTEGER NOT NULL,
    text TEXT NOT NULL,
    token_count INTEGER NOT NULL,
    section_path TEXT,
    page_start INTEGER,
    page_end INTEGER,
    char_start INTEGER,
    char_end INTEGER
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


def test_phase2_chunks_table_gains_clause_columns(tmp_path: Path) -> None:
    db_path = tmp_path / "legacy.db"
    _make_legacy_database(db_path)
    connection = sqlite3.connect(db_path)
    try:
        connection.execute(_LEGACY_CHUNKS)
        connection.execute(
            "INSERT INTO chunks (chunk_id, document_id, profile_id, chunk_order, text, "
            "token_count, section_path) VALUES ('c1', 'd1', 'p-old', 0, 'hello', 3, "
            "'1. Alpha')"
        )
        connection.commit()
    finally:
        connection.close()

    settings = Settings(
        data_dir=tmp_path,
        database_url=f"sqlite:///{db_path.as_posix()}",
        _env_file=None,
    )
    database = Database(settings.resolved_database_url)
    database.create_schema()
    try:
        with database.engine.connect() as conn:
            columns = {row[1] for row in conn.execute(text("PRAGMA table_info(chunks)"))}
            for name in (
                "section_number",
                "section_title",
                "parent_clause",
                "section_level",
                "diagnostics",
            ):
                assert name in columns
            row = conn.execute(
                text("SELECT section_path, section_number FROM chunks WHERE chunk_id = 'c1'")
            ).one()
            assert row[0] == "1. Alpha"
            assert row[1] is None  # legacy rows migrate with the new columns NULL
    finally:
        database.dispose()
