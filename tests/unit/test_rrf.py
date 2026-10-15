"""Unit tests for Reciprocal Rank Fusion."""

from __future__ import annotations

import pytest

from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion


def test_rrf_rewards_presence_in_both_lists():
    fused = reciprocal_rank_fusion({"bm25": ["p", "q"], "dense": ["p", "q"]}, k=60)
    assert [item.chunk_id for item in fused] == ["p", "q"]
    assert fused[0].score > fused[1].score
    assert fused[0].ranks == {"bm25": 1, "dense": 1}


def test_rrf_keeps_single_list_candidates():
    fused = reciprocal_rank_fusion({"bm25": ["a"], "dense": ["b"]}, k=60)
    ids = {item.chunk_id for item in fused}
    assert ids == {"a", "b"}
    # Both rank 1 in a single list -> equal score -> deterministic ID tie-break.
    assert [item.chunk_id for item in fused] == ["a", "b"]


def test_rrf_combines_partial_agreement():
    # "shared" is #2 in both lists; "top" is #1 in one list only.
    fused = reciprocal_rank_fusion(
        {"bm25": ["top", "shared", "x"], "dense": ["other", "shared"]}, k=60
    )
    scores = {item.chunk_id: item.score for item in fused}
    assert scores["shared"] == pytest.approx(1 / 62 + 1 / 62)
    assert scores["top"] == pytest.approx(1 / 61)
    assert fused[0].chunk_id == "shared"


def test_rrf_is_deterministic_under_ties():
    first = reciprocal_rank_fusion({"bm25": ["b", "a"], "dense": ["a", "b"]}, k=60)
    second = reciprocal_rank_fusion({"bm25": ["b", "a"], "dense": ["a", "b"]}, k=60)
    assert [item.chunk_id for item in first] == [item.chunk_id for item in second]


def test_rrf_rejects_invalid_k():
    with pytest.raises(ValueError):
        reciprocal_rank_fusion({"bm25": ["a"]}, k=0)
