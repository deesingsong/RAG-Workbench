"""Unit tests for the evaluation runner's pool-vs-ranking semantics.

These drive :meth:`EvaluationRunner._evaluate_query` directly with a fixed,
hand-built search stand-in, so the exact ordering contract can be asserted
without a database, a model or a corpus. They are the regression lock for the
rule that **reranking reorders the first-stage pool but never changes it**, and
for the four explicit quantities the runner keeps distinct: first-stage pool
depth, reranker input, final result count and candidate-recall depth.
"""

from __future__ import annotations

import pytest

from rag_workbench.domain.schemas import SearchHit, SearchResponse
from rag_workbench.evaluation.datasets import DOCUMENT_MAPPING
from rag_workbench.evaluation.runner import EvalRunConfig, EvaluationRunner


class _FixedSearch:
    """A search stand-in that models the two-call pool/rerank contract.

    A rerank-free call returns the *first-stage pool* (retrieval order). A
    reranking call returns the pre-supplied reranked order truncated to ``top_k``
    — exactly what the real service does. Every call's kwargs are recorded so a
    test can assert which depth and reranker input were requested.
    """

    def __init__(
        self,
        pool: list[SearchHit],
        reranked: list[SearchHit] | None = None,
    ) -> None:
        self._pool = pool
        self._reranked = reranked if reranked is not None else pool
        self.calls = 0
        self.kwargs: list[dict] = []

    def search(self, **kwargs):  # noqa: ANN003, ANN201 - duck-typed stand-in
        self.calls += 1
        self.kwargs.append(kwargs)
        rerank = bool(kwargs.get("rerank", False))
        ordered = self._reranked if rerank else self._pool
        top_k = int(kwargs.get("top_k", len(ordered)))
        return SearchResponse(
            query=kwargs["query"],
            mode=kwargs.get("mode", "bm25"),
            profile_id=kwargs.get("profile_id", "p-test"),
            profile_name="fixed",
            config_hash="0" * 64,
            top_k=top_k,
            rerank=rerank,
            rerank_candidates=kwargs.get("rerank_candidates"),
            timings_ms={"lexical": 1.0, "total": 2.0},
            total_ms=2.0,
            hits=list(ordered[:top_k]),
        )


def _hit(chunk_id: str, document_id: str, *, rank: int, retrieval_rank: int) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=chunk_id,
        document_id=document_id,
        source_name="doc.txt",
        collection="c",
        chunk_order=1,
        text=f"passage {chunk_id}",
        score=-1.0,
        score_kind="bm25",
        rerank_score=1.0,
        retrieval_rank=retrieval_rank,
        retrieval_score=-1.0,
        retrieval_score_kind="bm25",
        rank_movement=retrieval_rank - rank,
    )


def _runner(pool: list[SearchHit], reranked: list[SearchHit] | None = None) -> EvaluationRunner:
    runner = EvaluationRunner.__new__(EvaluationRunner)  # no database needed here
    runner._search = _FixedSearch(pool, reranked)  # type: ignore[attr-defined]
    return runner


def _evaluate(runner: EvaluationRunner, config: EvalRunConfig, relevance: dict[str, int]):
    external = {f"doc-{i}": f"D{i}" for i in range(1, 21)}
    return runner._evaluate_query(
        query_id="Q1",
        query_text="q",
        relevance=relevance,
        mapping=DOCUMENT_MAPPING,
        external_by_document=external,
        evidence_by_contract={},
        indexed_domain=set(external.values()),
        judged_domain=set(external.values()),
        collection="c",
        config=config,
    )


def test_candidate_recall_uses_first_stage_pool_not_rerank_order() -> None:
    # First-stage pool order is D1..D11 (retrieval ranks 1..11); reranking
    # promotes the relevant D11 (retrieval rank 11) to the top. candidate@10 must
    # reflect the *pool* (D11 is outside its first 10), while recall@1 reflects
    # the reranked order (D11 first).
    pool = [_hit(f"c{i}", f"doc-{i}", rank=i, retrieval_rank=i) for i in range(1, 12)]
    reranked = [pool[-1]] + pool[:-1]  # c11 first, then c1..c10
    runner = _runner(pool, reranked)
    config = EvalRunConfig(
        dataset="d",
        profile_id="p",
        top_k=1,
        candidate_depth=11,
        k_values=(1,),
        rerank=True,
        rerank_candidates=11,
    )
    outcome = _evaluate(runner, config, {"D11": 1})

    # Ranking metric follows the reranked order: D11 is now rank 1.
    assert outcome.metrics is not None
    assert outcome.metrics["recall@1"] == 1.0
    # Candidate recall follows the original pool: D11 is at retrieval rank 11.
    assert outcome.candidate_recall["candidate_recall@10"] == 0.0
    assert outcome.candidate_recall["candidate_recall@100"] == 1.0
    # The trace preserves the original retrieval rank of the promoted result.
    assert outcome.trace[0].key == "D11"
    assert outcome.trace[0].retrieval_rank == 11


def test_reranker_smaller_than_pool_still_measures_full_pool() -> None:
    # candidate_depth=11 (pool) but rerank_candidates=2: the reranker only sees
    # D1, D2 and returns D1, yet candidate recall is measured over the whole
    # 11-candidate pool, so the relevant D11 (rank 11) is *not* silently
    # discarded. The two calls must request the right depths.
    pool = [_hit(f"c{i}", f"doc-{i}", rank=i, retrieval_rank=i) for i in range(1, 12)]
    runner = _runner(pool, reranked=pool[:2])
    config = EvalRunConfig(
        dataset="d",
        profile_id="p",
        top_k=1,
        candidate_depth=11,
        k_values=(1,),
        rerank=True,
        rerank_candidates=2,
    )
    outcome = _evaluate(runner, config, {"D11": 1})

    # The final ranking only contains the reranker's input (D1, D2), so recall@1
    # is 0 (D11 was never scored).
    assert outcome.metrics is not None
    assert outcome.metrics["recall@1"] == 0.0
    # Candidate recall still covers the full pool: D11 sits at retrieval rank 11.
    assert outcome.candidate_recall["candidate_recall@10"] == 0.0
    assert outcome.candidate_recall["candidate_recall@100"] == 1.0
    # Two retrievals: the rerank-free pool at candidate_depth, then the rerank
    # over exactly rerank_candidates (requesting the whole reranked pool so
    # document-level de-duplication can fill top_k).
    search = runner._search  # type: ignore[attr-defined]
    assert search.calls == 2
    assert search.kwargs[0]["rerank"] is False
    assert search.kwargs[0]["top_k"] == 11
    assert search.kwargs[1]["rerank"] is True
    assert search.kwargs[1]["rerank_candidates"] == 2
    assert search.kwargs[1]["top_k"] == 2


def test_rerank_requests_full_pool_so_dedup_can_fill_top_k() -> None:
    # Two chunks from the same document occupy the top of the reranked order.
    # If the runner asked the service for only ``top_k`` hits, de-duplication
    # would collapse to a single document and the relevant second document would
    # be lost. The runner therefore requests the whole reranked pool.
    pool = [
        _hit("c1", "doc-1", rank=1, retrieval_rank=1),
        _hit("c2", "doc-1", rank=2, retrieval_rank=2),
        _hit("c3", "doc-2", rank=3, retrieval_rank=3),
    ]
    runner = _runner(pool)
    config = EvalRunConfig(
        dataset="d",
        profile_id="p",
        top_k=2,
        candidate_depth=3,
        k_values=(1, 2),
        rerank=True,
        rerank_candidates=3,
    )
    outcome = _evaluate(runner, config, {"D2": 1})

    # The service was asked for the full reranked pool (3), not just top_k (2).
    search = runner._search  # type: ignore[attr-defined]
    assert search.kwargs[1]["rerank"] is True
    assert search.kwargs[1]["top_k"] == 3
    # Document ranking after de-duplication is [D1, D2]; D2 is recovered at k=2.
    assert [item.key for item in outcome.trace] == ["D1", "D2"]
    assert outcome.metrics is not None
    assert outcome.metrics["recall@2"] == 1.0
    assert outcome.metrics["recall@1"] == 0.0


def test_without_rerank_candidate_recall_equals_ranking_pool() -> None:
    pool = [
        _hit("c1", "doc-1", rank=1, retrieval_rank=1),
        _hit("c2", "doc-2", rank=2, retrieval_rank=2),
    ]
    runner = _runner(pool)
    config = EvalRunConfig(
        dataset="d",
        profile_id="p",
        top_k=2,
        candidate_depth=2,
        k_values=(1,),
        rerank=False,
    )
    outcome = _evaluate(runner, config, {"D2": 1})
    assert outcome.candidate_recall["candidate_recall@10"] == 1.0
    assert outcome.candidate_recall["candidate_recall@100"] == 1.0
    # Without a reranker a single retrieval serves both roles.
    assert runner._search.calls == 1  # type: ignore[attr-defined]


@pytest.mark.parametrize(
    ("candidate_depth", "rerank_candidates", "top_k"),
    [
        (10, 20, 5),  # reranker input exceeds the retrieved pool
        (20, 5, 10),  # reranker input smaller than the final cut
    ],
)
def test_unsupported_rerank_geometry_is_rejected(
    candidate_depth: int, rerank_candidates: int, top_k: int
) -> None:
    with pytest.raises(ValueError):
        EvalRunConfig(
            dataset="d",
            profile_id="p",
            candidate_depth=candidate_depth,
            rerank_candidates=rerank_candidates,
            top_k=top_k,
            k_values=(1,),
            rerank=True,
        )


@pytest.mark.parametrize(
    ("candidate_depth", "rerank_candidates", "top_k"),
    [
        (100, 20, 10),  # deep pool, shallower reranker input
        (20, 20, 10),  # matched pool (the recorded rerank configuration)
        (20, 10, 10),  # reranker input equal to the final cut
    ],
)
def test_supported_rerank_geometry_is_accepted(
    candidate_depth: int, rerank_candidates: int, top_k: int
) -> None:
    config = EvalRunConfig(
        dataset="d",
        profile_id="p",
        candidate_depth=candidate_depth,
        rerank_candidates=rerank_candidates,
        top_k=top_k,
        k_values=(1, 3, 5, 10),
        rerank=True,
    )
    assert config.candidate_depth == candidate_depth
    assert config.rerank_candidates == rerank_candidates


def test_candidate_depth_beyond_service_max_is_rejected() -> None:
    with pytest.raises(ValueError):
        EvalRunConfig(dataset="d", profile_id="p", candidate_depth=500, top_k=10)
