"""Embedding provider: laziness, dependency handling and dimension checks."""

from __future__ import annotations

import sys

import pytest

from rag_workbench.config import ChunkerConfig, IndexProfileSpec
from rag_workbench.domain.errors import (
    DenseUnavailableError,
    DimensionMismatchError,
    ProfileNotDenseError,
)
from rag_workbench.embeddings import (
    SentenceTransformerEmbeddingProvider,
    _load_sentence_transformer,
    build_embedding_provider,
    clear_model_cache,
    validate_dimension,
)


def test_missing_sentence_transformers_raises_actionable_error(monkeypatch):
    clear_model_cache()
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    with pytest.raises(DenseUnavailableError) as excinfo:
        _load_sentence_transformer("some/model", None)
    assert "uv sync --extra dense" in str(excinfo.value)


def test_provider_construction_never_loads_a_model():
    provider = SentenceTransformerEmbeddingProvider("some/model")
    assert provider.model_id == "some/model"
    assert provider._model is None  # lazy: nothing imported or downloaded yet


def test_build_provider_rejects_lexical_profile():
    spec = IndexProfileSpec(name="x", chunker=ChunkerConfig())
    with pytest.raises(ProfileNotDenseError):
        build_embedding_provider(spec)


def test_validate_dimension_raises_on_mismatch():
    validate_dimension(384, 384)
    with pytest.raises(DimensionMismatchError):
        validate_dimension(384, 64)
