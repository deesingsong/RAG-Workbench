"""Unit tests for the clause-aware legal chunker and clause-marker detection."""

from __future__ import annotations

from pathlib import Path

from rag_workbench.config import ChunkerConfig
from rag_workbench.domain.schemas import ParsedBlock, ParsedDocument
from rag_workbench.ingestion.chunkers import (
    ClauseAwareChunker,
    count_tokens,
    detect_clause_marker,
)
from rag_workbench.ingestion.parsers import get_parser

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


def make_document(blocks: list[ParsedBlock]) -> ParsedDocument:
    return ParsedDocument(
        source_name="doc.txt",
        mime_type="text/plain",
        checksum="a" * 64,
        parser_version="test",
        blocks=blocks,
    )


def block(text: str, order: int, *, section: str | None = None, page: int | None = None):
    start = order * 100
    return ParsedBlock(
        text=text,
        order=order,
        section_path=section,
        page=page,
        char_start=start,
        char_end=start + len(text),
    )


def clause_config(max_tokens: int = 120, overlap_tokens: int = 20) -> ChunkerConfig:
    return ChunkerConfig(
        strategy="clause_aware", max_tokens=max_tokens, overlap_tokens=overlap_tokens
    )


def chunk(document: ParsedDocument, config: ChunkerConfig):
    return ClauseAwareChunker().chunk(
        document, config, profile_key="p-test", document_id="doc-1"
    )


def parse_legal_fixture() -> ParsedDocument:
    data = (FIXTURES_DIR / "legal_contract.txt").read_bytes()
    parser = get_parser("legal_contract.txt", "text/plain")
    return parser.parse(data, source_name="legal_contract.txt", mime_type="text/plain")


# --------------------------------------------------------------------------- #
# Marker detection
# --------------------------------------------------------------------------- #


def test_detects_numbered_article_schedule_and_alpha_markers():
    assert detect_clause_marker("2.1 Payment Terms") == (
        2,
        "2.1 Payment Terms",
        "2.1",
        "Payment Terms",
    )
    assert detect_clause_marker("3.3(a) Accrued Fees") == (
        3,
        "3.3(a) Accrued Fees",
        "3.3(a)",
        "Accrued Fees",
    )
    assert detect_clause_marker("ARTICLE IV") == (1, "ARTICLE IV", "ARTICLE IV", None)
    assert detect_clause_marker("SCHEDULE 1") == (1, "SCHEDULE 1", "SCHEDULE 1", None)
    assert detect_clause_marker("(b) The Supplier shall return materials.") == (
        None,
        "(b)",
        "(b)",
        None,
    )


def test_rejects_prose_numbers_dates_amounts_and_letters():
    # None of these are clause headings, even though they start with digits.
    assert detect_clause_marker("30 days written notice to the other party.") is None
    assert detect_clause_marker("1.5% per month may accrue on overdue amounts.") is None
    assert detect_clause_marker("2026-01-15 report issued by the Supplier.") is None
    assert detect_clause_marker("(three) days after written notice.") is None
    assert detect_clause_marker("12 months before the claim arose.") is None


def test_numbered_title_requires_title_case_and_bounded_length():
    # A lowercase continuation and an over-long "title" are not headings.
    assert detect_clause_marker("3.1 termination for convenience") is None
    long_line = "3.1 " + " ".join(["Word"] * 40)
    assert detect_clause_marker(long_line) is None


# --------------------------------------------------------------------------- #
# Structure preservation on the legal fixture
# --------------------------------------------------------------------------- #


def test_fixture_preserves_article_hierarchy():
    chunks = chunk(parse_legal_fixture(), clause_config())
    paths = {c.section_path for c in chunks}

    assert 'ARTICLE I > 1.1 "Confidential Information"' in paths
    assert "ARTICLE III > 3.1 Termination for Convenience" in paths
    assert "ARTICLE III > 3.3 Consequences of Termination > 3.3(a) Accrued Fees" in paths
    assert "ARTICLE IV > 4.1 Confidentiality" in paths
    assert "SCHEDULE 1" in paths


def test_fixture_section_number_title_and_parent_are_populated():
    chunks = chunk(parse_legal_fixture(), clause_config())
    by_number = {c.section_number: c for c in chunks if c.section_number}

    term = by_number["3.1"]
    assert term.section_title == "Termination for Convenience"
    assert term.parent_clause == "ARTICLE III"
    assert term.section_level == 2

    definition = by_number["1.2"]
    assert definition.section_title == '"Deliverable"'
    assert definition.parent_clause == "ARTICLE I"


def test_nested_alpha_subsections_are_separate_chunks():
    chunks = chunk(parse_legal_fixture(), clause_config())
    numbers = [c.section_number for c in chunks]
    assert numbers.count("3.3(a)") == 1
    assert numbers.count("3.3(b)") == 1

    alpha_chunk = next(c for c in chunks if c.section_number == "3.3(a)")
    assert alpha_chunk.section_level == 3
    assert alpha_chunk.parent_clause == "ARTICLE III > 3.3 Consequences of Termination"
    assert "Accrued Fees" in alpha_chunk.text


def test_definition_entries_are_grouped_with_their_own_text():
    chunks = chunk(parse_legal_fixture(), clause_config())
    definitions = [c for c in chunks if c.section_number in {"1.1", "1.2", "1.3"}]
    assert len(definitions) == 3
    for definition in definitions:
        # A definition keeps its explanatory text and is not merged with siblings.
        assert "means" in definition.text
        assert definition.text.count("means") == 1


# --------------------------------------------------------------------------- #
# Behavioural guarantees
# --------------------------------------------------------------------------- #


def test_groups_multiple_paragraphs_of_one_clause():
    document = make_document(
        [
            block("[intro]", 0),
            block("2.1 Payment Terms", 1, section="2.1 Payment Terms"),
            block("First paragraph of the clause.", 2, section="2.1 Payment Terms"),
            block("Second paragraph of the same clause.", 3, section="2.1 Payment Terms"),
            block("2.2 Set-Off", 4, section="2.2 Set-Off"),
            block("A different clause.", 5, section="2.2 Set-Off"),
        ]
    )
    chunks = chunk(document, clause_config(max_tokens=200, overlap_tokens=0))
    payment = next(c for c in chunks if c.section_path == "2.1 Payment Terms")
    assert "First paragraph" in payment.text
    assert "Second paragraph" in payment.text
    assert "different clause" not in payment.text


def test_does_not_merge_across_section_boundaries():
    document = make_document(
        [
            block("Alpha clause body.", 0, section="1.1 Alpha"),
            block("Beta clause body.", 1, section="1.2 Beta"),
        ]
    )
    chunks = chunk(document, clause_config(max_tokens=500, overlap_tokens=0))
    assert [c.section_path for c in chunks] == ["1.1 Alpha", "1.2 Beta"]


def test_splits_long_clause_and_keeps_parent_identity():
    body = " ".join(["term"] * 300)
    document = make_document([block(body, 0, section="7.2 Indemnity")])
    chunks = chunk(document, clause_config(max_tokens=50, overlap_tokens=5))
    assert len(chunks) > 1
    for index, split in enumerate(chunks, start=1):
        assert split.section_path == "7.2 Indemnity"
        assert split.section_number == "7.2"
        assert split.section_title == "Indemnity"
        assert any(f"part {index} of {len(chunks)}" in note for note in split.diagnostics)


def test_heading_only_chunk_is_flagged_not_dropped():
    document = make_document([block("SCHEDULE 2", 0)])
    chunks = chunk(document, clause_config(max_tokens=200, overlap_tokens=0))
    assert len(chunks) == 1
    assert chunks[0].section_number == "SCHEDULE 2"
    assert any("heading-only" in note for note in chunks[0].diagnostics)


def test_overlap_is_bounded_and_stays_within_a_clause():
    first = " ".join(["alpha"] * 40)
    second = " ".join(["beta"] * 40)
    same_clause = make_document(
        [
            block(first, 0, section="1.1 Alpha"),
            block(second, 1, section="1.1 Alpha"),
        ]
    )
    chunks = chunk(same_clause, clause_config(max_tokens=45, overlap_tokens=10))
    assert len(chunks) == 2
    assert "alpha" in chunks[1].text  # overlap carries context forward

    # A different clause starts clean: no alpha text leaks across sections.
    other = make_document(
        [
            block(first, 0, section="1.1 Alpha"),
            block(second, 1, section="1.2 Beta"),
        ]
    )
    other_chunks = chunk(other, clause_config(max_tokens=45, overlap_tokens=10))
    beta_only = next(c for c in other_chunks if c.section_path == "1.2 Beta")
    assert "alpha" not in beta_only.text


def test_chunking_is_deterministic():
    document = parse_legal_fixture()
    config = clause_config()
    first = chunk(document, config)
    second = chunk(document, config)
    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert [c.text for c in first] == [c.text for c in second]


def test_no_content_is_dropped():
    document = parse_legal_fixture()
    chunks = chunk(document, clause_config())
    combined = "\n".join(c.text for c in chunks)
    for token in ("Confidential", "availability", "aggregate liability"):
        assert token in combined
    # A heading the parser consumed stays available as provenance.
    assert any(c.section_path == "ARTICLE II > 2.2 Set-Off" for c in chunks)
    assert sum(c.token_count for c in chunks) >= sum(
        count_tokens(b.text) for b in document.blocks
    )
