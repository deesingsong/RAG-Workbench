"""Deterministic citation verification (Phase 5 §6).

This component is independent of the LLM. It cannot judge whether a passage
*actually supports* a claim — that is semantic support, which is not
deterministically decidable — so it never claims to. It verifies only the two
properties that *are* mechanical:

* **Structural validity** — the output conformed to the answer schema.
* **Citation validity** — every cited source id (in the claims and in the answer
  text) maps to a real selected passage, every identifier belongs to the current
  request, and an answer the model marked as evidence-supported actually carries
  citations.

Semantic support is reported as ``"not_determined"`` and requires separate
evaluation or human review.
"""

from __future__ import annotations

import re

from rag_workbench.domain.schemas import (
    AnswerPayload,
    CitationIssue,
    CitationValidation,
    EvidenceItem,
)


def abstained_validation(evidence: list[EvidenceItem]) -> CitationValidation:
    """Validation record for an abstention decided *before* any generation.

    Nothing was generated, so there is nothing to be structurally invalid about;
    the honest record is that generation was not attempted and the evidence set
    was empty.
    """

    return CitationValidation(
        structural_valid=True,
        citation_valid=True,
        known_source_ids=[item.source_id for item in evidence],
        notes=[
            "generation was not attempted because no evidence was available to ground an answer"
        ],
    )

#: Matches an ``[S3]`` / ``S3`` identifier anywhere in free text.
_CITATION_IN_TEXT = re.compile(r"\[?\b(S\d+)\b\]?", re.IGNORECASE)

_ANSWERED_STATES = ("answered", "partially_answered")


def normalize_source_id(raw: str) -> str:
    """Reduce ``"[s1]"``, ``" S1 "`` and ``"S1."`` to the canonical ``"S1"``."""

    match = _CITATION_IN_TEXT.search(raw or "")
    if match is not None:
        return match.group(1).upper()
    return (raw or "").strip().upper()


def inline_source_ids(text: str) -> list[str]:
    """Every distinct citation id that appears inline in ``text``, in order."""

    seen: set[str] = set()
    ordered: list[str] = []
    for match in _CITATION_IN_TEXT.finditer(text or ""):
        source_id = match.group(1).upper()
        if source_id not in seen:
            seen.add(source_id)
            ordered.append(source_id)
    return ordered


def verify_citations(
    payload: AnswerPayload | None,
    *,
    evidence: list[EvidenceItem],
    structural_valid: bool = True,
) -> CitationValidation:
    """Check a parsed answer's citations against the selected evidence."""

    known = [item.source_id for item in evidence]
    known_set = set(known)
    notes = [
        "citation validity is a mechanical check that cited ids exist in the "
        "selected context; it is not proof that the evidence supports the claim",
        "semantic support requires separate evaluation or human review",
    ]

    if payload is None:
        return CitationValidation(
            structural_valid=False,
            citation_valid=False,
            known_source_ids=known,
            issues=[
                CitationIssue(
                    code="malformed_generation",
                    detail="the model did not return a schema-valid answer object",
                )
            ],
            notes=notes,
        )

    issues: list[CitationIssue] = []
    cited: list[str] = []
    seen: set[str] = set()
    for claim in payload.claims:
        for raw in claim.citations:
            source_id = normalize_source_id(raw)
            if source_id not in seen:
                seen.add(source_id)
                cited.append(source_id)

    for source_id in inline_source_ids(payload.answer):
        if source_id not in seen:
            seen.add(source_id)
            cited.append(source_id)

    unknown = [source_id for source_id in cited if source_id not in known_set]
    for source_id in unknown:
        issues.append(
            CitationIssue(
                code="unknown_citation",
                source_id=source_id,
                detail=f"cited source {source_id} is not one of the selected passages",
            )
        )

    uncited_claim_indices = [
        index for index, claim in enumerate(payload.claims) if not claim.citations
    ]
    for index in uncited_claim_indices:
        issues.append(
            CitationIssue(
                code="uncited_claim",
                claim_index=index,
                detail="this claim carries no citation",
            )
        )

    if payload.status in _ANSWERED_STATES:
        if not payload.claims:
            issues.append(
                CitationIssue(
                    code="answered_without_claims",
                    detail="the answer is marked supported but lists no claims",
                )
            )
        elif all(not claim.citations for claim in payload.claims):
            issues.append(
                CitationIssue(
                    code="answered_without_citations",
                    detail="the answer is marked supported but cites no passages",
                )
            )

    citation_valid = structural_valid and not unknown
    if payload.status in _ANSWERED_STATES:
        citation_valid = citation_valid and bool(payload.claims) and bool(
            any(claim.citations for claim in payload.claims)
        )

    return CitationValidation(
        structural_valid=structural_valid,
        citation_valid=citation_valid,
        semantic_support="not_determined",
        cited_source_ids=cited,
        known_source_ids=known,
        unknown_source_ids=unknown,
        uncited_claim_indices=uncited_claim_indices,
        issues=issues,
        notes=notes,
    )


__all__ = [
    "abstained_validation",
    "inline_source_ids",
    "normalize_source_id",
    "verify_citations",
]
