"""Unit tests for evidence selection and bounded context assembly."""

from __future__ import annotations

from rag_workbench.answering.context import ContextAssembler, estimate_tokens
from rag_workbench.domain.schemas import SearchHit


def _hit(rank: int, text: str, *, chunk_id: str | None = None, score: float = 0.0) -> SearchHit:
    return SearchHit(
        rank=rank,
        chunk_id=chunk_id or f"chunk-{rank}",
        document_id="doc-1",
        source_name="contract.txt",
        collection="legal",
        chunk_order=rank,
        text=text,
        score=score,
        section_number=f"{rank}.1",
    )


def test_estimate_tokens_is_positive_and_monotonic() -> None:
    assert estimate_tokens("") == 0
    short = estimate_tokens("termination for convenience")
    long = estimate_tokens("termination for convenience " * 10)
    assert short > 0
    assert long > short


def test_assigns_sequential_source_ids_in_rank_order() -> None:
    hits = [_hit(1, "alpha passage"), _hit(2, "beta passage"), _hit(3, "gamma passage")]
    assembly = ContextAssembler(max_evidence=6, token_budget=2048).assemble(hits)
    ids = [item.source_id for item in assembly.evidence]
    assert ids == ["S1", "S2", "S3"]
    # Each id maps back to the exact retrieved chunk text (verbatim).
    assert [item.text for item in assembly.evidence] == [
        "alpha passage",
        "beta passage",
        "gamma passage",
    ]
    assert assembly.diagnostics is not None
    assert assembly.diagnostics.retrieved == 3
    assert assembly.diagnostics.selected == 3


def test_exact_duplicate_is_excluded_as_duplicate() -> None:
    hits = [_hit(1, "Same   text here"), _hit(2, "same text here")]
    assembly = ContextAssembler(max_evidence=6, token_budget=2048).assemble(hits)
    assert len(assembly.evidence) == 1
    assert assembly.diagnostics is not None
    assert assembly.diagnostics.duplicates_removed == 1
    assert assembly.excluded[0].reason == "duplicate"


def test_contained_overlap_is_excluded() -> None:
    hits = [_hit(1, "The Client shall pay within thirty (30) days."), _hit(2, "pay within thirty")]
    assembly = ContextAssembler(max_evidence=6, token_budget=2048).assemble(hits)
    assert len(assembly.evidence) == 1
    assert assembly.diagnostics is not None
    assert assembly.diagnostics.overlaps_removed == 1
    assert assembly.excluded[0].reason == "overlap"


def test_max_evidence_is_enforced_and_reasoned() -> None:
    hits = [_hit(i, f"passage number {i}") for i in range(1, 6)]
    assembly = ContextAssembler(max_evidence=2, token_budget=2048).assemble(hits)
    assert len(assembly.evidence) == 2
    assert assembly.diagnostics is not None
    assert assembly.diagnostics.excluded_by_max == 3
    assert all(item.reason == "max_evidence" for item in assembly.excluded)
    # Nothing is silently dropped: every non-selected hit is accounted for.
    assert assembly.diagnostics.selected + len(assembly.excluded) == assembly.diagnostics.retrieved


def test_token_budget_excludes_oversized_passage_without_truncation() -> None:
    hits = [_hit(1, "short passage"), _hit(2, "x " * 500)]
    assembly = ContextAssembler(max_evidence=6, token_budget=40).assemble(hits)
    assert [item.text for item in assembly.evidence] == ["short passage"]
    assert assembly.diagnostics is not None
    assert assembly.diagnostics.excluded_by_budget == 1
    assert assembly.excluded[0].reason == "context_budget"


def test_no_passage_fits_reports_empty_selection_note() -> None:
    hits = [_hit(1, "x " * 500)]
    assembly = ContextAssembler(max_evidence=6, token_budget=10).assemble(hits)
    assert assembly.evidence == []
    assert assembly.diagnostics is not None
    assert assembly.diagnostics.notes


def test_reserve_tokens_reduces_available_budget() -> None:
    hits = [_hit(1, "a passage of moderate length")]
    generous = ContextAssembler(max_evidence=6, token_budget=1000).assemble(hits)
    reserved = ContextAssembler(max_evidence=6, token_budget=1000).assemble(
        hits, reserve_tokens=1000
    )
    assert len(generous.evidence) == 1
    assert reserved.evidence == []


def test_assembly_is_deterministic() -> None:
    hits = [_hit(i, f"passage {i}") for i in range(1, 5)]
    first = ContextAssembler(max_evidence=3, token_budget=2048).assemble(hits)
    second = ContextAssembler(max_evidence=3, token_budget=2048).assemble(hits)
    assert [item.model_dump() for item in first.evidence] == [
        item.model_dump() for item in second.evidence
    ]
