"""Unit tests for configuration and index-profile identity."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings


def test_overlap_must_be_smaller_than_max():
    with pytest.raises(ValidationError):
        ChunkerConfig(max_tokens=10, overlap_tokens=10)


def test_profile_key_ignores_name_but_tracks_chunker():
    a = IndexProfileSpec(name="one", chunker=ChunkerConfig(max_tokens=100))
    b = IndexProfileSpec(name="two", chunker=ChunkerConfig(max_tokens=100))
    c = IndexProfileSpec(name="one", chunker=ChunkerConfig(max_tokens=200))

    assert a.key == b.key
    assert a.key != c.key
    assert a.config_hash != c.config_hash


def test_profile_key_is_stable():
    spec = IndexProfileSpec(name="x", chunker=ChunkerConfig(strategy="fixed_token", max_tokens=64))
    assert spec.key == IndexProfileSpec(
        name="y", chunker=ChunkerConfig(strategy="fixed_token", max_tokens=64)
    ).key


def test_settings_resolve_default_sqlite_path(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    assert settings.resolved_database_url.startswith("sqlite:///")
    assert "rag_workbench.db" in settings.resolved_database_url


def test_settings_env_prefix(tmp_path, monkeypatch):
    monkeypatch.setenv("RAG_WORKBENCH_MAX_UPLOAD_MB", "7")
    monkeypatch.setenv("RAG_WORKBENCH_DATA_DIR", str(tmp_path))
    settings = Settings(_env_file=None)
    assert settings.max_upload_mb == 7
    assert settings.max_upload_bytes == 7 * 1024 * 1024


def test_default_profile_matches_settings():
    settings = Settings(
        default_chunk_strategy="fixed_token", default_max_tokens=128, _env_file=None
    )
    profile = settings.default_profile()
    assert profile.chunker.strategy == "fixed_token"
    assert profile.chunker.max_tokens == 128
