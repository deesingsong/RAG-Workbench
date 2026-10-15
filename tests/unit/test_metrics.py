"""Hand-computed tests for the retrieval metrics.

Every expectation below is derived by hand from the standard IR definitions, so
the suite validates the implementation rather than restating it.
"""

from __future__ import annotations

import math

import pytest

from rag_workbench.evaluation.metrics import (
    aggregate,
    average_precision_at_k,
    candidate_recall_at_n,
    dedupe_first,
    mrr_at_k,
    ndcg_at_k,
    precision_at_k,
    query_metrics,
    recall_at_k,
    relevant_keys,
)


def test_recall_and_precision_hand_computed() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    assert recall_at_k(ranked, relevant, 1) == pytest.approx(0.5)
    assert recall_at_k(ranked, relevant, 3) == pytest.approx(1.0)
    assert precision_at_k(ranked, relevant, 1) == pytest.approx(1.0)
    assert precision_at_k(ranked, relevant, 3) == pytest.approx(2 / 3)
    assert precision_at_k(ranked, relevant, 5) == pytest.approx(2 / 5)


def test_mrr_hand_computed() -> None:
    ranked = ["x", "a", "b"]
    relevant = {"a", "b"}
    assert mrr_at_k(ranked, relevant, 3) == pytest.approx(0.5)
    assert mrr_at_k(ranked, relevant, 1) == pytest.approx(0.0)
    assert mrr_at_k(["z", "y"], {"a"}, 2) == 0.0


def test_average_precision_hand_computed() -> None:
    ranked = ["a", "b", "c", "d", "e"]
    relevant = {"a", "c"}
    # (1/1 + 2/3) / 2
    assert average_precision_at_k(ranked, relevant, 5) == pytest.approx((1 + 2 / 3) / 2)
    # With the cut-off before the second relevant result only the first counts.
    assert average_precision_at_k(ranked, relevant, 1) == pytest.approx(1.0)


def test_ndcg_binary_hand_computed() -> None:
    ranked = ["a", "b", "c"]
    relevance = {"a": 1, "c": 1}
    # DCG = 1/log2(2) + 0 + 1/log2(4) = 1 + 0.5
    dcg = 1.0 + 1.0 / math.log2(4)
    idcg = 1.0 / math.log2(2) + 1.0 / math.log2(3)
    assert ndcg_at_k(ranked, relevance, 3) == pytest.approx(dcg / idcg)


def test_ndcg_graded_hand_computed() -> None:
    ranked = ["b", "a", "c"]
    relevance = {"a": 2, "b": 1, "c": 0}
    dcg = (2**1 - 1) / math.log2(2) + (2**2 - 1) / math.log2(3)
    idcg = (2**2 - 1) / math.log2(2) + (2**1 - 1) / math.log2(3)
    assert ndcg_at_k(ranked, relevance, 3) == pytest.approx(dcg / idcg)
    # At k=1 the ideal ranking is just the highest-graded document (gain 2).
    assert ndcg_at_k(ranked, relevance, 1) == pytest.approx(1.0 / ((2**2 - 1) / math.log2(2)))


def test_query_without_relevant_keys_is_none() -> None:
    assert query_metrics(["a", "b"], {}) is None
    assert query_metrics(["a"], {"x": 0}) is None
    assert relevant_keys({"x": 0, "y": 2}) == {"y"}


def test_recall_requires_relevant_keys() -> None:
    with pytest.raises(ValueError):
        recall_at_k(["a"], set(), 1)
    with pytest.raises(ValueError):
        average_precision_at_k(["a"], set(), 1)


def test_candidate_recall_measures_pool_coverage() -> None:
    # Two of three relevant keys are in the pool, regardless of their ordering.
    ranked = ["x", "a", "y", "b", "z"]
    relevant = {"a", "b", "c"}
    assert candidate_recall_at_n(ranked, relevant, 5) == pytest.approx(2 / 3)
    # A deeper pool does not add anything when the missing key is absent.
    assert candidate_recall_at_n(ranked, relevant, 10) == pytest.approx(2 / 3)


def test_dedupe_preserves_first_occurrence() -> None:
    assert dedupe_first(["a", "b", "a", "c", "b"]) == ["a", "b", "c"]


def test_multiple_relevant_documents_and_incomplete_pool() -> None:
    # Only one of four relevant documents is in the pool at all.
    ranked = ["x", "a", "y", "z"]
    relevant = {"a", "b", "c", "d"}
    assert recall_at_k(ranked, relevant, 10) == pytest.approx(0.25)
    assert precision_at_k(ranked, relevant, 4) == pytest.approx(0.25)
    assert candidate_recall_at_n(ranked, relevant, 4) == pytest.approx(0.25)


def test_query_metrics_shape() -> None:
    metrics = query_metrics(["a", "b"], {"a": 1}, k_values=(1, 3))
    assert metrics is not None
    assert set(metrics) == {
        "recall@1",
        "precision@1",
        "mrr@1",
        "ndcg@1",
        "map@1",
        "recall@3",
        "precision@3",
        "mrr@3",
        "ndcg@3",
        "map@3",
    }


def test_aggregate_excludes_unevaluated_queries() -> None:
    q1 = query_metrics(["a"], {"a": 1}, k_values=(1,))
    q2 = query_metrics(["b"], {}, k_values=(1,))
    q3 = query_metrics(["b", "a"], {"a": 1}, k_values=(1,))
    assert q1 is not None and q2 is None and q3 is not None
    agg = aggregate([q1, q2, q3], k_values=(1,))
    # recall: (1.0 + 0.0) / 2 ; the excluded query is not counted.
    assert agg["recall@1"] == pytest.approx(0.5)
    # precision: (1.0 + 0.0) / 2 -- q3's top-1 result is not relevant.
    assert agg["precision@1"] == pytest.approx(0.5)


def test_aggregate_empty_when_nothing_evaluated() -> None:
    assert aggregate([None, None]) == {}
