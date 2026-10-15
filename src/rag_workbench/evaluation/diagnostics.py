"""Rule-based failure diagnostics for a single evaluated query.

Every diagnosis is derived from **observable evidence**: the query's judgments,
the candidate pool, the final ranking, and the set of units that were actually
indexed. The module never infers a root cause it did not measure, and falls back
to :data:`UNRESOLVED` when the evidence does not decide the case.

The categories mirror the Phase 4 specification. They describe *where in the
pipeline* the relevant evidence was lost:

* ``absent_from_corpus``            — the relevant unit is not indexed at all
  (dataset/corpus mismatch, or an unimplemented relevance mapping).
* ``no_relevant_chunk_retrieved``   — the unit is indexed but no chunk from it
  reached the candidate pool.
* ``below_top_k_cutoff``            — a relevant candidate was retrieved in the
  pool but ranked below the final top-K cut-off.
* ``poorly_positioned_after_rerank``— a relevant candidate was inside the
  pre-rerank top-K but pushed below the cut-off by reranking.
* ``provenance_missing``            — retrieved candidates could not be mapped
  back to a judged unit (dropped, never invented).
* ``evaluation_inconsistency``      — the judgments reference units that are
  outside the dataset's judged domain.
* ``unresolved``                    — the evidence does not determine a cause.
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import BaseModel, ConfigDict

ABSENT_FROM_CORPUS = "absent_from_corpus"
NO_RELEVANT_CHUNK_RETRIEVED = "no_relevant_chunk_retrieved"
BELOW_TOP_K_CUTOFF = "below_top_k_cutoff"
POORLY_POSITIONED_AFTER_RERANK = "poorly_positioned_after_rerank"
PROVENANCE_MISSING = "provenance_missing"
EVALUATION_INCONSISTENCY = "evaluation_inconsistency"
UNRESOLVED = "unresolved"

DIAGNOSIS_LABELS: dict[str, str] = {
    ABSENT_FROM_CORPUS: "Relevant unit is not present in the indexed corpus",
    NO_RELEVANT_CHUNK_RETRIEVED: "Relevant unit is indexed but no relevant chunk was retrieved",
    BELOW_TOP_K_CUTOFF: "Relevant candidate retrieved below the final top-K cut-off",
    POORLY_POSITIONED_AFTER_RERANK: "Relevant candidate demoted below the cut-off by reranking",
    PROVENANCE_MISSING: "Retrieved candidates could not be mapped to a judged unit",
    EVALUATION_INCONSISTENCY: "Evaluation data or relevance judgments are inconsistent",
    UNRESOLVED: "Cause not determined from observable evidence",
}


class Diagnostic(BaseModel):
    """One observed failure signal for a query."""

    model_config = ConfigDict(frozen=True)

    code: str
    detail: str

    @property
    def label(self) -> str:
        return DIAGNOSIS_LABELS.get(self.code, self.code)


def diagnose_query(
    *,
    relevant: set[str],
    candidate_keys: Sequence[str],
    final_keys: Sequence[str],
    indexed_domain: set[str],
    judged_domain: set[str],
    top_k: int,
    reranked: bool,
    baseline_keys: Sequence[str] | None = None,
    dropped_hits: int = 0,
) -> list[Diagnostic]:
    """Return the observable failure signals for one query.

    ``candidate_keys`` is the mapped, de-duplicated candidate ranking (first
    occurrence preserved); ``final_keys`` is the ranking after any rerank and the
    top-K cut-off. ``baseline_keys`` is the pre-rerank pooled ranking, used only to
    distinguish a reranker demotion from a shallow pool. ``indexed_domain`` is the
    set of units the *corpus* can produce; ``judged_domain`` is the set of units
    the *qrels* can reference.
    """

    diagnostics: list[Diagnostic] = []
    if dropped_hits:
        diagnostics.append(
            Diagnostic(
                code=PROVENANCE_MISSING,
                detail=f"{dropped_hits} retrieved candidate(s) had no resolvable unit mapping",
            )
        )

    missing = relevant - set(candidate_keys)
    if missing:
        absent = sorted(key for key in missing if key not in indexed_domain)
        judged_but_not_indexed = sorted(
            key for key in missing if key in indexed_domain and key not in judged_domain
        )
        if absent or judged_but_not_indexed:
            sample = ", ".join((absent + judged_but_not_indexed)[:5])
            diagnostics.append(
                Diagnostic(
                    code=ABSENT_FROM_CORPUS,
                    detail=(
                        f"{len(absent) + len(judged_but_not_indexed)} relevant unit(s) not in the "
                        f"indexed corpus: {sample}"
                    ),
                )
            )
        absent_set = set(absent)
        unindexed_set = set(judged_but_not_indexed)
        pool_missing = [
            key
            for key in missing
            if key in indexed_domain and key not in absent_set and key not in unindexed_set
        ]
        if pool_missing:
            sample = ", ".join(pool_missing[:5])
            diagnostics.append(
                Diagnostic(
                    code=NO_RELEVANT_CHUNK_RETRIEVED,
                    detail=(
                        f"{len(pool_missing)} indexed relevant unit(s) produced no "
                        f"candidate: {sample}"
                    ),
                )
            )

    final_set = set(final_keys[:top_k])
    pool_set = set(candidate_keys)
    cutoff = sorted(key for key in (relevant & pool_set) if key not in final_set)
    if cutoff:
        diagnostics.append(
            Diagnostic(
                code=BELOW_TOP_K_CUTOFF,
                detail=(
                    f"{len(cutoff)} relevant candidate(s) in the pool fell below top-{top_k}: "
                    + ", ".join(cutoff[:5])
                ),
            )
        )

    if reranked:
        # A unit inside the pre-rerank top-K but outside the post-rerank top-K was
        # demoted by the reranker; that is a reranking-quality signal, distinct
        # from a shallow candidate pool.
        baseline_topk = set((baseline_keys or candidate_keys)[:top_k])
        demoted = sorted(key for key in relevant if key in baseline_topk and key not in final_set)
        if demoted:
            diagnostics.append(
                Diagnostic(
                    code=POORLY_POSITIONED_AFTER_RERANK,
                    detail=(
                        f"{len(demoted)} relevant candidate(s) were inside the pre-rerank "
                        f"top-{top_k} but demoted below it: " + ", ".join(demoted[:5])
                    ),
                )
            )

    inconsistency = sorted(key for key in relevant if key not in judged_domain)
    if inconsistency:
        diagnostics.append(
            Diagnostic(
                code=EVALUATION_INCONSISTENCY,
                detail=(
                    f"{len(inconsistency)} judged unit(s) are outside the dataset domain: "
                    + ", ".join(inconsistency[:5])
                ),
            )
        )

    if relevant and relevant.isdisjoint(judged_domain):
        diagnostics.append(
            Diagnostic(
                code=UNRESOLVED,
                detail="no judged unit could be observed in the indexed corpus",
            )
        )
    return diagnostics


__all__ = [
    "ABSENT_FROM_CORPUS",
    "BELOW_TOP_K_CUTOFF",
    "DIAGNOSIS_LABELS",
    "EVALUATION_INCONSISTENCY",
    "NO_RELEVANT_CHUNK_RETRIEVED",
    "POORLY_POSITIONED_AFTER_RERANK",
    "PROVENANCE_MISSING",
    "UNRESOLVED",
    "Diagnostic",
    "diagnose_query",
]
