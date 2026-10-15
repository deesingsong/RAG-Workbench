"""Unit tests for the local cross-encoder reranker wiring (no model download)."""

from __future__ import annotations

from rag_workbench.config import DEFAULT_RERANKER_MODEL, Settings
from rag_workbench.reranking import (
    CrossEncoderReranker,
    _as_flat_list,
    build_reranker,
)


def test_build_reranker_defaults():
    reranker = build_reranker()
    assert isinstance(reranker, CrossEncoderReranker)
    assert reranker.model_id == DEFAULT_RERANKER_MODEL
    assert reranker.revision is None
    assert reranker.max_length is None


def test_build_reranker_uses_settings(tmp_path):
    settings = Settings(
        data_dir=tmp_path,
        reranker_model="cross-encoder/test-model",
        reranker_revision="v2",
        reranker_batch_size=8,
        reranker_max_length=128,
        _env_file=None,
    )
    reranker = build_reranker(settings=settings)
    assert reranker.model_id == "cross-encoder/test-model"
    assert reranker.revision == "v2"
    assert reranker.batch_size == 8
    assert reranker.max_length == 128


def test_explicit_model_id_overrides_settings(tmp_path):
    settings = Settings(data_dir=tmp_path, reranker_model="from-settings", _env_file=None)
    reranker = build_reranker("explicit-model", settings)
    assert reranker.model_id == "explicit-model"


def test_reranker_is_lazy_and_does_not_load_for_empty_input():
    reranker = CrossEncoderReranker("cross-encoder/never-loaded", max_length=64)
    assert reranker.is_loaded is False
    batch = reranker.score("query", [])
    assert batch.scores == []
    assert reranker.is_loaded is False


def test_as_flat_list_normalizes_shapes():
    assert _as_flat_list([1.0, 2.0]) == [1.0, 2.0]
    assert _as_flat_list([[0.5], [0.25]]) == [0.5, 0.25]


class _Array:
    """Minimal numpy-like object exposing ``reshape(-1).tolist()``."""

    def __init__(self, values: list[float]) -> None:
        self._values = values

    def reshape(self, *_shape):  # noqa: ANN002, ANN201
        return self

    def tolist(self) -> list[float]:
        return self._values


def test_as_flat_list_handles_array_like():
    assert _as_flat_list(_Array([3.0, 4.0])) == [3.0, 4.0]
