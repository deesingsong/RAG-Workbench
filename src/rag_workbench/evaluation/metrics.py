"""Pure-Python retrieval evaluation metrics.

The metrics here operate on a **ranked list of result keys** and a set (or, for
graded relevance, a mapping) of relevant keys. A "key" is whatever unit the
dataset's relevance judgments are expressed in: a benchmark document ID for
document-level datasets (BEIR SciFact), or a clause/evidence-unit key for the
curated legal set. Mapping retrieved chunks onto those keys happens in
:mod:`rag_workbench.evaluation.runner`; this module is deliberately free of any
retrieval or persistence concern and can be hand-verified in isolation.

Definitions (all standard IR definitions):

* ``Recall@K``      — fraction of relevant keys retrieved in the top ``K``.
* ``Precision@K``   — fraction of the top ``K`` results that are relevant
  (denominator is ``K``).
* ``MRR@K``         — reciprocal rank of the first relevant result within ``K``.
* ``nDCG@K``        — graded-gain DCG normalised by the ideal DCG.
* ``MAP@K``         — mean of per-query average precision; here the per-query
  value is *average precision*, the ``map_`` prefix is applied when aggregating.
* ``Candidate Recall@N`` — fraction of relevant keys present anywhere in a
  candidate pool of size ``N`` (a ranking-independent coverage measure).

Treatment of edge cases is explicit:

* A query with **no relevant keys** has no meaningful average precision; its
  per-query metrics are ``None`` and the query is excluded from aggregates.
  Aggregation never silently mixes such queries into an average.
* Ties and duplicate keys are resolved upstream: callers pass a de-duplicated,
  first-occurrence-preserving ranking, so a repeated key does not inflate
  precision or consume multiple positions.
* An "insufficient" candidate pool is not an error here; ``candidate_recall``
  and the per-query diagnostics surface it.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

#: Default cut-off depths for the ranking metrics.
DEFAULT_K_VALUES: tuple[int, ...] = (1, 3, 5, 10)

Relevance = Mapping[str, int]


def dedupe_first(keys: Sequence[str]) -> list[str]:
    """Collapse a ranked list to unique keys, keeping the first occurrence."""

    seen: set[str] = set()
    ordered: list[str] = []
    for key in keys:
        if key not in seen:
            seen.add(key)
            ordered.append(key)
    return ordered


def relevant_keys(relevance: Relevance) -> set[str]:
    """Keys judged relevant (grade > 0)."""

    return {key for key, grade in relevance.items() if grade > 0}


def recall_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant keys retrieved in the top ``k``."""

    if not relevant:
        raise ValueError("recall is undefined for a query with no relevant keys")
    hits = sum(1 for key in ranked[:k] if key in relevant)
    return hits / len(relevant)


def precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Fraction of the top ``k`` results that are relevant (denominator ``k``)."""

    if k <= 0:
        raise ValueError("k must be positive")
    hits = sum(1 for key in ranked[:k] if key in relevant)
    return hits / k


def mrr_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Reciprocal rank of the first relevant key within the top ``k`` (0 if none)."""

    for position, key in enumerate(ranked[:k], start=1):
        if key in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(gains: Sequence[float], k: int) -> float:
    """Discounted cumulative gain with the exponential gain form."""

    total = 0.0
    for position, gain in enumerate(gains[:k], start=1):
        total += (2.0**gain - 1.0) / math.log2(position + 1)
    return total


def ndcg_at_k(ranked: Sequence[str], relevance: Relevance, k: int) -> float:
    """Normalised DCG at ``k`` using graded relevance."""

    gains = [float(relevance.get(key, 0)) for key in ranked[:k]]
    dcg = dcg_at_k(gains, k)
    ideal_gains = sorted((float(g) for g in relevance.values() if g > 0), reverse=True)
    idcg = dcg_at_k(ideal_gains, k)
    if idcg == 0.0:
        return 0.0
    return dcg / idcg


def average_precision_at_k(ranked: Sequence[str], relevant: set[str], k: int) -> float:
    """Average precision at ``k`` for a single query."""

    if not relevant:
        raise ValueError("average precision is undefined for a query with no relevant keys")
    hits = 0
    precision_sum = 0.0
    for position, key in enumerate(ranked[:k], start=1):
        if key in relevant:
            hits += 1
            precision_sum += hits / position
    return precision_sum / min(len(relevant), k)


def candidate_recall_at_n(ranked: Sequence[str], relevant: set[str], n: int) -> float:
    """Fraction of relevant keys present in a candidate pool of ``n`` results.

    This is a coverage measure for candidate generation: it answers "did the
    first-stage retriever surface the relevant documents at all?", independent of
    how they were ordered. It uses the same formula as :func:`recall_at_k`, and is
    named separately so reports never conflate ranking quality with pool coverage.
    """

    return recall_at_k(ranked, relevant, n)


#: The ranking metrics keyed by their public name.
RANKING_METRICS = ("recall", "precision", "mrr", "ndcg", "map")


def query_metrics(
    ranked: Sequence[str],
    relevance: Relevance,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float] | None:
    """Compute every ranking metric at every ``k`` for one query.

    Returns ``None`` when the query has no relevant keys, signalling that it must
    be excluded from aggregates rather than counted as a zero.
    """

    relevant = relevant_keys(relevance)
    if not relevant:
        return None
    metrics: dict[str, float] = {}
    for k in k_values:
        metrics[f"recall@{k}"] = recall_at_k(ranked, relevant, k)
        metrics[f"precision@{k}"] = precision_at_k(ranked, relevant, k)
        metrics[f"mrr@{k}"] = mrr_at_k(ranked, relevant, k)
        metrics[f"ndcg@{k}"] = ndcg_at_k(ranked, relevance, k)
        metrics[f"map@{k}"] = average_precision_at_k(ranked, relevant, k)
    return metrics


def aggregate(
    per_query: Sequence[Mapping[str, float] | None],
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
) -> dict[str, float]:
    """Mean of each metric over the queries that were actually evaluated.

    ``None`` entries (queries without relevance judgments) are excluded from the
    mean and must be reported in a separate exclusion count by the caller.
    """

    evaluated = [item for item in per_query if item is not None]
    if not evaluated:
        return {}
    aggregate_metrics: dict[str, float] = {}
    for k in k_values:
        for name in RANKING_METRICS:
            key = f"{name}@{k}"
            values = [item[key] for item in evaluated if key in item]
            if values:
                aggregate_metrics[key] = sum(values) / len(values)
    return aggregate_metrics


__all__ = [
    "DEFAULT_K_VALUES",
    "RANKING_METRICS",
    "Relevance",
    "aggregate",
    "average_precision_at_k",
    "candidate_recall_at_n",
    "dcg_at_k",
    "dedupe_first",
    "mrr_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "query_metrics",
    "recall_at_k",
    "relevant_keys",
]
