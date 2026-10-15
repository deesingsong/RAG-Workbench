"""Evidence selection and bounded context assembly (Phase 5 §4).

Retrieved candidates and the *selected context* are different things, and this
module keeps them apart. It converts a ranked list of
:class:`~rag_workbench.domain.schemas.SearchHit` into a bounded, deterministically
ordered list of :class:`~rag_workbench.domain.schemas.EvidenceItem`, assigning
request-scoped ``[S1]``/``[S2]`` identifiers, removing exact duplicates and
contained overlaps, and refusing passages that would exceed the context token
budget.

Honesty rules:

* A passage is selected **verbatim**; provenance is copied, never inferred.
* Every retrieved candidate that is *not* selected is recorded with a reason, so
  selection is never a silent truncation.
* Token counts are **estimates** (a conservative word/character heuristic). The
  project's regex tokenizer does not represent an LLM's context window, so the
  method is labelled ``"estimated"`` and the limit is applied conservatively.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from rag_workbench.domain.schemas import (
    ContextDiagnostics,
    EvidenceItem,
    ExcludedEvidence,
    SearchHit,
)

#: A rough tokenizer used only for *estimating* size; not an LLM tokenizer.
_WORD = re.compile(r"\w+|[^\w\s]", re.UNICODE)

#: Conservative per-character divisor for the estimate (a real BPE tokenizer
#: averages ~4 characters/token; dividing by 4 and taking the max of the two
#: estimates biases upward, i.e. toward *under*-filling the budget).
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Estimate a text's token count conservatively (never exact).

    The estimate is the larger of a punctuation-aware word count and
    ``len(text) / 4``. It is intentionally an over-estimate: it is used to keep a
    context budget, so erring high is the safe direction.
    """

    if not text:
        return 0
    words = len(_WORD.findall(text))
    by_chars = -(-len(text) // _CHARS_PER_TOKEN)
    return max(words, by_chars)


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


@dataclass
class ContextAssembly:
    """The selected evidence plus the selection diagnostics."""

    evidence: list[EvidenceItem] = field(default_factory=list)
    excluded: list[ExcludedEvidence] = field(default_factory=list)
    diagnostics: ContextDiagnostics | None = None


class ContextAssembler:
    """Select a bounded, deduplicated evidence context from ranked hits.

    ``max_evidence`` caps the number of passages; ``token_budget`` caps the
    estimated size of the whole prompt (passed the budget *after* the caller has
    reserved room for trusted instructions, the question, the output schema and
    the expected completion).
    """

    def __init__(self, *, max_evidence: int = 6, token_budget: int = 2048) -> None:
        if max_evidence < 1:
            raise ValueError("max_evidence must be >= 1")
        if token_budget < 1:
            raise ValueError("token_budget must be >= 1")
        self._max_evidence = max_evidence
        self._token_budget = token_budget

    @property
    def max_evidence(self) -> int:
        return self._max_evidence

    def assemble(
        self,
        hits: list[SearchHit],
        *,
        reserve_tokens: int = 0,
    ) -> ContextAssembly:
        """Build the bounded evidence context from a ranked hit list.

        ``reserve_tokens`` is the estimated size of everything else in the prompt
        (instructions, question, schema, expected output); it is subtracted from
        the budget before passages are admitted.
        """

        available = max(self._token_budget - max(reserve_tokens, 0), 0)
        evidence: list[EvidenceItem] = []
        excluded: list[ExcludedEvidence] = []
        selected_norm: list[str] = []
        seen_norm: set[str] = set()
        seen_chunks: set[str] = set()
        used = 0
        duplicates = overlaps = by_budget = by_max = 0

        for hit in hits:
            text = hit.text.strip()
            norm = _normalize(text)

            if not norm or hit.chunk_id in seen_chunks or norm in seen_norm:
                duplicates += 1
                excluded.append(
                    ExcludedEvidence(
                        chunk_id=hit.chunk_id,
                        source_name=hit.source_name,
                        rank=hit.rank,
                        reason="duplicate",
                        detail="identical (or empty) passage already selected",
                    )
                )
                continue

            if any(norm in chosen or chosen in norm for chosen in selected_norm):
                overlaps += 1
                excluded.append(
                    ExcludedEvidence(
                        chunk_id=hit.chunk_id,
                        source_name=hit.source_name,
                        rank=hit.rank,
                        reason="overlap",
                        detail="passage is contained in (or contains) a selected passage",
                    )
                )
                continue

            if len(evidence) >= self._max_evidence:
                by_max += 1
                excluded.append(
                    ExcludedEvidence(
                        chunk_id=hit.chunk_id,
                        source_name=hit.source_name,
                        rank=hit.rank,
                        reason="max_evidence",
                        detail=f"already selected the maximum of {self._max_evidence} passages",
                    )
                )
                continue

            cost = estimate_tokens(text)
            if used + cost > available:
                by_budget += 1
                excluded.append(
                    ExcludedEvidence(
                        chunk_id=hit.chunk_id,
                        source_name=hit.source_name,
                        rank=hit.rank,
                        reason="context_budget",
                        detail=(
                            f"estimated {cost} tokens would exceed the remaining "
                            f"{max(available - used, 0)}-token budget"
                        ),
                    )
                )
                continue

            evidence.append(
                EvidenceItem(
                    source_id=f"S{len(evidence) + 1}",
                    chunk_id=hit.chunk_id,
                    document_id=hit.document_id,
                    source_name=hit.source_name,
                    collection=hit.collection,
                    section_path=hit.section_path,
                    section_number=hit.section_number,
                    section_title=hit.section_title,
                    page_start=hit.page_start,
                    page_end=hit.page_end,
                    rank=hit.rank,
                    score=hit.score,
                    score_kind=hit.score_kind,
                    text=text,
                    token_estimate=cost,
                )
            )
            selected_norm.append(norm)
            seen_norm.add(norm)
            seen_chunks.add(hit.chunk_id)
            used += cost

        notes: list[str] = []
        if not evidence and hits:
            notes.append(
                "no retrieved passage fitted the context budget; the answer will abstain"
            )
        diag = ContextDiagnostics(
            retrieved=len(hits),
            selected=len(evidence),
            duplicates_removed=duplicates,
            overlaps_removed=overlaps,
            excluded_by_budget=by_budget,
            excluded_by_max=by_max,
            token_estimate=used,
            token_budget=self._token_budget,
            token_count_method="estimated",
            notes=notes,
        )
        return ContextAssembly(evidence=evidence, excluded=excluded, diagnostics=diag)


__all__ = ["ContextAssembler", "ContextAssembly", "estimate_tokens"]
