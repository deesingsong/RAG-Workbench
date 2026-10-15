"""Unit tests for the evaluation runner's pool-vs-ranking semantics.

These drive :meth:`EvaluationRunner._evaluate_query` directly with a fixed,
hand-built search response, so the exact ordering contract can be asserted
without a database, a model or a corpus. They are the regression lock for the
rule that **reranking reorders the first-stage pool but never changes it**.
"""

from __future__ import annotations

from rag_workbench.domain.schemas import SearchHit, SearchResponse
from rag_workbench.evaluation.datasets import DOCUMENT_MAPPING
from rag_workbench.evaluation.runner import EvalRunConfig, EvaluationRunner


class _FixedSearch:
    """Return a pre-built response; count how many queries were retrieved."""

    def __init__(self, hits: list[SearchHit]) -> None:
        self._hits = hits
        self.calls = 0

    def search(self, **kwargs):  # noqa: ANN003, ANN201 - duck-typed stand-in
        self.calls += 1
        return SearchResponse(
            query=kwargs["query"],
            mode=kwargs.get("mode", "bm25"),
            profile_id=kwargs.get("profile_id", "p-test"),
            profile_name="fixed",
            config_hash="0" * 64,
            top_k=kwargs.get("top_k", 0),
            rerank=kwargs.get("rerank", False),
            rerank_candidates=kwargs.get("rerank_candidates"),
            timings_ms={"lexical": 1.0, "total": 2.0},
            total_ms=2.0,
            hits=list(self._hits),
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


def _runner(hits: list[SearchHit]) -> EvaluationRunner:
    runner = EvaluationRunner.__new__(EvaluationRunner)  # no database needed here
    runner._search = _FixedSearch(hits)  # type: ignore[attr-defined]
    return runner


def test_candidate_recall_uses_first_stage_pool_not_rerank_order() -> None:
    # Retrieval order is D1..D11; reranking promotes the relevant D11 (retrieval
    # rank 11) to the top. candidate@10 must reflect the *pool* (D11 is outside
    # its first 10), while recall@1 reflects the reranked order (D11 first).
    hits = [_hit(f"c{i}", f"doc-{i}", rank=i + 1, retrieval_rank=i) for i in range(1, 11)]
    hits.append(_hit("c11", "doc-11", rank=1, retrieval_rank=11))
    # Rank order as returned post-rerank: c11 first, then c1..c10.
    hits = [hits[-1]] + hits[:-1]
    runner = _runner(hits)
    config = EvalRunConfig(
        dataset="d",
        profile_id="p",
        top_k=1,
        candidate_depth=11,
        k_values=(1,),
        rerank=True,
        rerank_candidates=11,
    )
    external = {f"doc-{i}": f"D{i}" for i in range(1, 12)}
    outcome = runner._evaluate_query(
        query_id="Q1",
        query_text="q",
        relevance={"D11": 1},
        mapping=DOCUMENT_MAPPING,
        external_by_document=external,
        evidence_by_contract={},
        indexed_domain=set(external.values()),
        judged_domain=set(external.values()),
        collection="c",
        config=config,
    )

    # Ranking metric follows the reranked order: D11 is now rank 1.
    assert outcome.metrics is not None
    assert outcome.metrics["recall@1"] == 1.0
    # Candidate recall follows the original pool: D11 is at retrieval rank 11.
    assert outcome.candidate_recall["candidate_recall@10"] == 0.0
    assert outcome.candidate_recall["candidate_recall@100"] == 1.0
    # The trace preserves the original retrieval rank of the promoted result.
    assert outcome.trace[0].key == "D11"
    assert outcome.trace[0].retrieval_rank == 11


def test_without_rerank_candidate_recall_equals_ranking_pool() -> None:
    hits = [
        _hit("c1", "doc-1", rank=1, retrieval_rank=1),
        _hit("c2", "doc-2", rank=2, retrieval_rank=2),
    ]
    runner = _runner(hits)
    config = EvalRunConfig(
        dataset="d",
        profile_id="p",
        top_k=2,
        candidate_depth=2,
        k_values=(1,),
        rerank=False,
    )
    outcome = runner._evaluate_query(
        query_id="Q1",
        query_text="q",
        relevance={"D2": 1},
        mapping=DOCUMENT_MAPPING,
        external_by_document={"doc-1": "D1", "doc-2": "D2"},
        evidence_by_contract={},
        indexed_domain={"D1", "D2"},
        judged_domain={"D1", "D2"},
        collection="c",
        config=config,
    )
    assert outcome.candidate_recall["candidate_recall@10"] == 1.0
    assert outcome.candidate_recall["candidate_recall@100"] == 1.0
