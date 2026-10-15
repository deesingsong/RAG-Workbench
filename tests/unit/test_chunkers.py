"""Unit tests for the deterministic chunkers and the tokenizer."""

from __future__ import annotations

from rag_workbench.config import ChunkerConfig
from rag_workbench.domain.schemas import ParsedBlock, ParsedDocument
from rag_workbench.ingestion.chunkers import (
    FixedTokenChunker,
    ParagraphChunker,
    count_tokens,
    tokenize_spans,
)


def make_document(blocks: list[ParsedBlock]) -> ParsedDocument:
    return ParsedDocument(
        source_name="doc.txt",
        mime_type="text/plain",
        checksum="a" * 64,
        parser_version="test",
        blocks=blocks,
    )


def block(text: str, order: int, *, section: str | None = None, page: int | None = None):
    return ParsedBlock(text=text, order=order, section_path=section, page=page)


def test_tokenizer_counts_words_and_punctuation():
    assert count_tokens("hello world") == 2
    assert count_tokens("hello, world!") == 4  # two words + comma + bang
    assert count_tokens("") == 0
    spans = tokenize_spans("ab cd")
    assert spans == [(0, 2), (3, 5)]


def test_fixed_token_windows_and_overlap():
    text = "one two three four five six seven eight nine ten"
    document = make_document([block(text, 0)])
    config = ChunkerConfig(strategy="fixed_token", max_tokens=4, overlap_tokens=1)

    chunks = FixedTokenChunker().chunk(document, config, profile_key="p-test", document_id="doc-1")

    assert [chunk.text for chunk in chunks] == [
        "one two three four",
        "four five six seven",
        "seven eight nine ten",
    ]
    assert [chunk.token_count for chunk in chunks] == [4, 4, 4]
    assert [chunk.order for chunk in chunks] == [0, 1, 2]


def test_fixed_token_never_exceeds_max_tokens():
    text = " ".join(f"w{i}" for i in range(100))
    document = make_document([block(text, 0)])
    config = ChunkerConfig(strategy="fixed_token", max_tokens=10, overlap_tokens=3)

    chunks = FixedTokenChunker().chunk(document, config, profile_key="p-test", document_id="doc-1")

    assert all(chunk.token_count <= 10 for chunk in chunks)
    assert sum(1 for _ in chunks) >= 10


def test_paragraph_packing_respects_budget():
    document = make_document(
        [
            block("a b c", 0, section="S"),
            block("d e f", 1, section="S"),
            block("g h i", 2, section="S"),
            block("j k l", 3, section="S"),
        ]
    )
    config = ChunkerConfig(strategy="paragraph", max_tokens=7, overlap_tokens=0)

    chunks = ParagraphChunker().chunk(document, config, profile_key="p-test", document_id="doc-1")

    assert len(chunks) == 2
    assert chunks[0].token_count == 6
    assert chunks[0].text == "a b c\n\nd e f"
    assert chunks[1].token_count == 6


def test_paragraph_never_spans_sections():
    document = make_document(
        [
            block("alpha beta", 0, section="One"),
            block("gamma delta", 1, section="Two"),
        ]
    )
    config = ChunkerConfig(strategy="paragraph", max_tokens=100, overlap_tokens=0)

    chunks = ParagraphChunker().chunk(document, config, profile_key="p-test", document_id="doc-1")

    assert len(chunks) == 2
    assert chunks[0].section_path == "One"
    assert chunks[1].section_path == "Two"


def test_paragraph_carries_bounded_overlap():
    document = make_document(
        [
            block("a b c d e f", 0, section="S"),
            block("g h i j k l", 1, section="S"),
        ]
    )
    config = ChunkerConfig(strategy="paragraph", max_tokens=10, overlap_tokens=2)

    chunks = ParagraphChunker().chunk(document, config, profile_key="p-test", document_id="doc-1")

    assert len(chunks) == 2
    assert chunks[0].token_count == 6
    assert chunks[1].token_count == 8  # 2 overlap tokens + 6 new
    assert chunks[1].text.endswith("g h i j k l")
    assert chunks[1].text.startswith("e f")


def test_long_paragraph_is_split_by_tokens():
    text = " ".join(f"t{i}" for i in range(12))
    document = make_document([block(text, 0, section="S")])
    config = ChunkerConfig(strategy="paragraph", max_tokens=5, overlap_tokens=0)

    chunks = ParagraphChunker().chunk(document, config, profile_key="p-test", document_id="doc-1")

    assert len(chunks) == 3
    assert [chunk.token_count for chunk in chunks] == [5, 5, 2]


def test_chunk_ids_are_deterministic_and_profile_scoped():
    document = make_document([block("alpha beta gamma", 0)])
    config = ChunkerConfig(strategy="paragraph", max_tokens=50, overlap_tokens=0)

    first = ParagraphChunker().chunk(document, config, profile_key="p-one", document_id="doc-1")
    second = ParagraphChunker().chunk(document, config, profile_key="p-one", document_id="doc-1")
    other = ParagraphChunker().chunk(document, config, profile_key="p-two", document_id="doc-1")

    assert [c.chunk_id for c in first] == [c.chunk_id for c in second]
    assert first[0].chunk_id != other[0].chunk_id
    assert len(first[0].chunk_id) == 36


def test_empty_document_produces_no_chunks():
    document = make_document([])
    config = ChunkerConfig(strategy="paragraph")
    assert ParagraphChunker().chunk(document, config, profile_key="p", document_id="doc-1") == []
    assert FixedTokenChunker().chunk(document, config, profile_key="p", document_id="doc-1") == []


def test_char_offsets_are_preserved():
    blocks = [ParsedBlock(text="alpha beta gamma", order=0, char_start=10, char_end=26)]
    document = make_document(blocks)
    config = ChunkerConfig(strategy="fixed_token", max_tokens=1, overlap_tokens=0)

    chunks = FixedTokenChunker().chunk(document, config, profile_key="p", document_id="doc-1")

    assert chunks[0].char_start == 10
    assert chunks[0].char_end == 15  # "alpha"
