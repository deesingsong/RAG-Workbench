"""Reciprocal Rank Fusion (RRF) for hybrid retrieval.

Fusion operates purely on *ranked ID lists*, never on raw scores: adding a BM25
score to a cosine similarity would be meaningless. The standard formulation is

    RRF(d) = Σ_i 1 / (k + rank_i(d))

with ``rank_i`` 1-based and ``k`` defaulting to 60. Documents found by only one
retriever stay eligible because they simply receive a contribution from one list.
Ties are broken deterministically by chunk ID.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

#: Default RRF constant (Cormack et al., 2009).
DEFAULT_RRF_K = 60


@dataclass(frozen=True)
class FusedItem:
    """One fused candidate with its RRF score and per-source ranks."""

    chunk_id: str
    score: float
    ranks: dict[str, int] = field(default_factory=dict)


def reciprocal_rank_fusion(
    rankings: Mapping[str, Sequence[str]], *, k: int = DEFAULT_RRF_K
) -> list[FusedItem]:
    """Fuse named ranked ID lists into one deterministically ordered list.

    ``rankings`` maps a source name (``"bm25"``, ``"dense"``) to a list of chunk
    IDs ordered best-first. Every distinct ID receives one contribution per list
    it appears in. Output is ordered by descending score, then ascending chunk ID.
    """

    if k < 1:
        raise ValueError("rrf k must be >= 1")

    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)

    for source, ids in rankings.items():
        for position, chunk_id in enumerate(ids, start=1):
            scores[chunk_id] += 1.0 / (k + position)
            # Keep the best (smallest) rank if a source lists an ID twice.
            existing = ranks[chunk_id].get(source)
            if existing is None or position < existing:
                ranks[chunk_id][source] = position

    items = [
        FusedItem(chunk_id=chunk_id, score=scores[chunk_id], ranks=dict(ranks[chunk_id]))
        for chunk_id in scores
    ]
    items.sort(key=lambda item: (-item.score, item.chunk_id))
    return items


__all__ = ["DEFAULT_RRF_K", "FusedItem", "reciprocal_rank_fusion"]
