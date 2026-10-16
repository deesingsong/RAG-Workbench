"""Unit tests for reference-evidence coverage classification (audit §1, §3).

They pin the distinction the quality runner relies on: selecting *one* of two
required evidence units is partial coverage, not complete coverage.
"""

from __future__ import annotations

from rag_workbench.answering.reference import classify_reference_coverage
from rag_workbench.domain.schemas import EvidenceItem, SearchHit


def _evidence(
    source_id: str,
    text: str,
    *,
    source: str = "doc.txt",
    section: str | None = None,
) -> EvidenceItem:
    return EvidenceItem(
        source_id=source_id,
        chunk_id=f"c-{source_id}",
        document_id="d1",
        source_name=source,
        section_number=section,
        rank=1,
        score=0.0,
        text=text,
    )


def _hit(
    rank: int,
    text: str,
    *,
    source: str = "doc.txt",
    section: str | None = None,
) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=f"c-{rank}",
        document_id="d1",
        source_name=source,
        collection="contracts",
        chunk_order=rank,
        section_number=section,
        text=text,
        score=-1.0,
    )


def test_one_of_two_anchors_selected_is_partial_not_complete() -> None:
    selected = [_evidence("S1", "invoices are due within thirty days")]
    retrieved = [
        _hit(1, "invoices are due within thirty days"),
        _hit(2, "interest accrues at one percent per month"),
    ]
    coverage = classify_reference_coverage(
        keywords=["thirty days", "one percent per month"],
        sections=[],
        selected=selected,
        retrieved=retrieved,
    )
    assert coverage.required == 2
    assert coverage.selected_count == 1
    assert coverage.retrieved_count == 2
    assert coverage.complete_selected is False
    assert coverage.partial_selected is True
    assert coverage.classification == "retrieved_but_excluded"
    assert coverage.missing_selected() == ["one percent per month"]
    assert coverage.missing_retrieved() == []


def test_anchor_absent_from_the_pool_is_never_retrieved() -> None:
    selected = [_evidence("S1", "invoices are due within thirty days")]
    retrieved = [_hit(1, "invoices are due within thirty days")]
    coverage = classify_reference_coverage(
        keywords=["thirty days", "one percent per month"],
        sections=[],
        selected=selected,
        retrieved=retrieved,
    )
    assert coverage.classification == "never_retrieved"
    assert coverage.complete_retrieved is False
    assert coverage.missing_retrieved() == ["one percent per month"]


def test_all_anchors_selected_is_complete_coverage() -> None:
    selected = [
        _evidence("S1", "within thirty days"),
        _evidence("S2", "one percent per month"),
    ]
    retrieved = [_hit(1, "within thirty days"), _hit(2, "one percent per month")]
    coverage = classify_reference_coverage(
        keywords=["thirty days", "one percent per month"],
        sections=[],
        selected=selected,
        retrieved=retrieved,
    )
    assert coverage.complete_selected is True
    assert coverage.partial_selected is False
    assert coverage.classification == "included"


def test_no_anchors_is_undefined_coverage() -> None:
    coverage = classify_reference_coverage(
        keywords=[], sections=[], selected=[], retrieved=[]
    )
    assert coverage.required == 0
    assert coverage.complete_selected is None
    assert coverage.partial_selected is None
    assert coverage.classification is None


def test_section_anchors_used_when_no_keywords() -> None:
    selected = [
        _evidence("S1", "clause text", source="msa-02.txt", section="3.2"),
    ]
    coverage = classify_reference_coverage(
        keywords=[],
        sections=["msa-02::3.2"],
        selected=selected,
        retrieved=list(selected),
    )
    assert coverage.complete_selected is True
    assert coverage.classification == "included"


def test_keyword_anchor_is_matched_lexically_not_by_section() -> None:
    # A keyword anchor is matched by its text, not by a section that happens to be
    # selected, so a missing phrase is not counted as covered — and if the phrase is
    # nowhere in the pool, it is "never_retrieved", not merely excluded.
    selected = [
        _evidence("S1", "the maximum rate permitted by law", section="3.2"),
    ]
    coverage = classify_reference_coverage(
        keywords=["one percent per month"],
        sections=[],
        selected=selected,
        retrieved=list(selected),
    )
    assert coverage.complete_selected is False
    assert coverage.classification == "never_retrieved"
