"""Unit tests for deterministic citation verification."""

from __future__ import annotations

from rag_workbench.answering.citations import (
    abstained_validation,
    inline_source_ids,
    normalize_source_id,
    verify_citations,
)
from rag_workbench.domain.schemas import AnswerPayload, Claim, EvidenceItem


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_id="S1",
            chunk_id="c1",
            document_id="d1",
            source_name="nda.txt",
            rank=1,
            score=0.0,
            text="The term is three years.",
        ),
        EvidenceItem(
            source_id="S2",
            chunk_id="c2",
            document_id="d1",
            source_name="nda.txt",
            rank=2,
            score=0.0,
            text="Obligations survive for two years.",
        ),
    ]


def test_valid_citations_pass_but_semantic_support_is_not_asserted() -> None:
    payload = AnswerPayload(
        status="answered",
        answer="The term is three years [S1].",
        claims=[Claim(text="The term is three years.", citations=["S1"])],
    )
    validation = verify_citations(payload, evidence=_evidence())
    assert validation.structural_valid is True
    assert validation.citation_valid is True
    assert validation.unknown_source_ids == []
    assert validation.semantic_support == "not_determined"
    assert validation.cited_source_ids == ["S1"]


def test_unknown_citation_is_flagged() -> None:
    payload = AnswerPayload(
        status="answered",
        answer="See [S9].",
        claims=[Claim(text="A claim.", citations=["S9"])],
    )
    validation = verify_citations(payload, evidence=_evidence())
    assert validation.citation_valid is False
    assert validation.unknown_source_ids == ["S9"]
    assert any(issue.code == "unknown_citation" for issue in validation.issues)


def test_bracketed_and_lowercase_ids_are_normalized() -> None:
    assert normalize_source_id("[s1]") == "S1"
    assert normalize_source_id(" S1 ") == "S1"
    assert normalize_source_id("S1.") == "S1"


def test_inline_ids_in_answer_text_are_checked() -> None:
    payload = AnswerPayload(
        status="answered",
        answer="A statement without bracket claim, but [S2] appears inline.",
        claims=[Claim(text="A claim.", citations=["S1"])],
    )
    validation = verify_citations(payload, evidence=_evidence())
    assert "S2" in validation.cited_source_ids
    assert inline_source_ids("[S1] and S2 and [s1]") == ["S1", "S2"]


def test_uncited_claim_is_flagged_and_invalidates_answer() -> None:
    payload = AnswerPayload(
        status="answered",
        answer="An answer with no citations.",
        claims=[Claim(text="A claim with no citation.", citations=[])],
    )
    validation = verify_citations(payload, evidence=_evidence())
    assert validation.citation_valid is False
    assert validation.uncited_claim_indices == [0]
    assert any(issue.code == "answered_without_citations" for issue in validation.issues)


def test_abstention_needs_no_citation() -> None:
    payload = AnswerPayload(
        status="insufficient_evidence",
        answer="",
        claims=[],
        reason="not in the passages",
    )
    validation = verify_citations(payload, evidence=_evidence())
    assert validation.citation_valid is True
    assert validation.structural_valid is True


def test_missing_payload_is_structural_failure() -> None:
    validation = verify_citations(None, evidence=_evidence())
    assert validation.structural_valid is False
    assert validation.citation_valid is False
    assert any(issue.code == "malformed_generation" for issue in validation.issues)


def test_abstained_validation_is_not_a_failure() -> None:
    validation = abstained_validation([])
    assert validation.structural_valid is True
    assert validation.citation_valid is True
    assert validation.notes
