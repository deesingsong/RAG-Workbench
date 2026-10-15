"""Vector store behavior against real Qdrant local-mode storage."""

from __future__ import annotations

import uuid

import pytest

from rag_workbench.domain.errors import DimensionMismatchError, VectorStoreError
from rag_workbench.domain.schemas import VectorPoint
from rag_workbench.persistence.vector_store import (
    QdrantLocalStore,
    close_vector_stores,
    collection_name,
    get_vector_store,
)


def _point(vector: list[float], *, collection: str) -> VectorPoint:
    return VectorPoint(
        id=str(uuid.uuid4()), vector=vector, payload={"collection": collection}
    )


def test_ensure_collection_is_idempotent_and_checks_dimension(tmp_path):
    store = QdrantLocalStore(tmp_path / "q")
    try:
        store.ensure_collection("c", dimension=3)
        store.ensure_collection("c", dimension=3)  # no-op
        assert store.collection_dimension("c") == 3
        with pytest.raises(DimensionMismatchError):
            store.ensure_collection("c", dimension=4)
    finally:
        store.close()


def test_upsert_is_idempotent_per_point_id(tmp_path):
    store = QdrantLocalStore(tmp_path / "q")
    try:
        store.ensure_collection("c", dimension=2)
        point = _point([1.0, 0.0], collection="docs")
        assert store.upsert("c", [point]) == 1
        assert store.upsert("c", [point]) == 1  # same ID -> overwrite, not duplicate
        assert store.count("c") == 1
    finally:
        store.close()


def test_search_orders_by_similarity_and_filters_by_collection(tmp_path):
    store = QdrantLocalStore(tmp_path / "q")
    try:
        store.ensure_collection("c", dimension=2)
        near = _point([1.0, 0.0], collection="a")
        far = _point([0.0, 1.0], collection="b")
        store.upsert("c", [near, far])

        hits = store.search("c", [1.0, 0.0], limit=10)
        assert [hit.id for hit in hits] == [near.id, far.id]
        assert hits[0].score > hits[1].score

        filtered = store.search("c", [1.0, 0.0], limit=10, collection="b")
        assert [hit.id for hit in filtered] == [far.id]
    finally:
        store.close()


def test_delete_list_and_drop(tmp_path):
    store = QdrantLocalStore(tmp_path / "q")
    try:
        store.ensure_collection("c", dimension=2)
        points = [_point([1.0, 0.0], collection="a"), _point([0.0, 1.0], collection="a")]
        store.upsert("c", points)

        assert store.list_ids("c") == {p.id for p in points}
        assert store.delete_ids("c", [points[0].id]) == 1
        assert store.list_ids("c") == {points[1].id}

        store.drop("c")
        assert store.has_collection("c") is False
        assert store.count("c") == 0
        assert store.list_ids("c") == set()
        assert store.search("c", [1.0, 0.0], limit=5) == []
    finally:
        store.close()


def test_search_on_missing_collection_returns_empty(tmp_path):
    store = QdrantLocalStore(tmp_path / "q")
    try:
        assert store.search("nope", [1.0, 0.0], limit=5) == []
    finally:
        store.close()


def test_data_persists_across_reopen(tmp_path):
    path = tmp_path / "q"
    store = get_vector_store(path)
    store.ensure_collection("c", dimension=2)
    point = _point([1.0, 0.0], collection="a")
    store.upsert("c", [point])
    close_vector_stores()

    reopened = get_vector_store(path)
    try:
        assert reopened.count("c") == 1
        assert reopened.list_ids("c") == {point.id}
    finally:
        close_vector_stores()


def test_get_vector_store_returns_same_instance(tmp_path):
    path = tmp_path / "q"
    assert get_vector_store(path) is get_vector_store(path)


def test_collection_name_matches_profile_convention():
    assert collection_name("p-abc") == "rw_p-abc"


def test_unsupported_metric_is_rejected(tmp_path):
    store = QdrantLocalStore(tmp_path / "q")
    try:
        with pytest.raises(VectorStoreError):
            store.ensure_collection("c", dimension=2, metric="manhattan")
    finally:
        store.close()
