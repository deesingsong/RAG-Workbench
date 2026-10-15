"""Dense index-profile identity and Phase-1 backward compatibility."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from rag_workbench.config import (
    DEFAULT_PARSER_VERSION,
    INDEX_VERSION,
    ChunkerConfig,
    IndexProfileSpec,
    Settings,
    resolve_embedding_dimension,
)


def test_lexical_canonical_payload_is_unchanged_from_phase1():
    """A lexical profile must hash exactly as it did in Phase 1."""

    spec = IndexProfileSpec(name="paragraph-220", chunker=ChunkerConfig())
    assert spec.canonical() == {
        "parser_version": DEFAULT_PARSER_VERSION,
        "chunker": spec.chunker.model_dump(),
        "embedding_model": None,
        "embedding_dimension": None,
        "index_version": INDEX_VERSION,
    }
    assert spec.is_dense is False


def test_dense_profile_requires_a_dimension():
    with pytest.raises(ValidationError):
        IndexProfileSpec(name="d", chunker=ChunkerConfig(), embedding_model="m")


def test_dimension_without_model_is_rejected():
    with pytest.raises(ValidationError):
        IndexProfileSpec(name="d", chunker=ChunkerConfig(), embedding_dimension=384)


def test_dense_identity_tracks_every_behaviour_affecting_field():
    base = {
        "name": "d",
        "chunker": ChunkerConfig(),
        "embedding_model": "m",
        "embedding_dimension": 384,
        "embedding_revision": "r1",
    }
    reference = IndexProfileSpec(**base).key

    assert IndexProfileSpec(**{**base, "embedding_revision": "r2"}).key != reference
    assert IndexProfileSpec(**{**base, "normalize_embeddings": False}).key != reference
    assert IndexProfileSpec(**{**base, "embedding_document_prefix": "passage: "}).key != reference
    assert IndexProfileSpec(**{**base, "embedding_query_prefix": "query: "}).key != reference


def test_dense_and_lexical_profiles_never_share_a_key():
    chunker = ChunkerConfig()
    lexical = IndexProfileSpec(name="x", chunker=chunker)
    dense = IndexProfileSpec(
        name="x", chunker=chunker, embedding_model="m", embedding_dimension=384
    )
    assert lexical.key != dense.key


def test_vector_collection_derives_from_profile_key():
    spec = IndexProfileSpec(
        name="d", chunker=ChunkerConfig(), embedding_model="m", embedding_dimension=8
    )
    assert spec.vector_collection == f"rw_{spec.key}"


def test_resolve_embedding_dimension_refuses_to_guess():
    assert (
        resolve_embedding_dimension("sentence-transformers/all-MiniLM-L6-v2", None) == 384
    )
    assert resolve_embedding_dimension("whatever", 128) == 128
    with pytest.raises(ValueError):
        resolve_embedding_dimension("unknown/model", None)


def test_settings_dense_profile_uses_configured_defaults(tmp_path):
    settings = Settings(data_dir=tmp_path, _env_file=None)
    spec = settings.dense_profile()
    assert spec.is_dense
    assert spec.embedding_model == settings.embedding_model
    assert spec.embedding_dimension == 384
