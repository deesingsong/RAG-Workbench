"""Unit tests for parsers and provenance extraction."""

from __future__ import annotations

import pytest

from rag_workbench.domain.errors import ParsingError, UnsupportedFileTypeError
from rag_workbench.ingestion.parsers import (
    MarkdownParser,
    PdfParser,
    PlainTextParser,
    get_parser,
    mime_type_for,
)


def test_plain_text_numbered_headings_produce_section_paths():
    text = (
        "1. Definitions\n"
        "\n"
        "The following terms apply.\n"
        "\n"
        "2.1 Payment Terms\n"
        "\n"
        "Invoices are due in thirty days.\n"
    )
    document = PlainTextParser().parse(
        text.encode("utf-8"), source_name="c.txt", mime_type="text/plain"
    )

    sections = [block.section_path for block in document.blocks]
    assert "1. Definitions" in sections
    assert "1. Definitions > 2.1 Payment Terms" in sections


def test_markdown_headings_build_hierarchy():
    text = "# Title\n\n## Section A\n\nBody text here.\n"
    document = MarkdownParser().parse(
        text.encode("utf-8"), source_name="d.md", mime_type="text/markdown"
    )

    assert document.blocks[0].section_path == "Title > Section A"
    assert document.blocks[0].text == "Body text here."


def test_sentence_is_not_mistaken_for_heading():
    text = "Payment shall be made within 30 days.\n"
    document = PlainTextParser().parse(
        text.encode("utf-8"), source_name="c.txt", mime_type="text/plain"
    )

    assert len(document.blocks) == 1
    assert document.blocks[0].section_path is None


def test_checksum_is_sha256_of_bytes():
    data = b"hello"
    document = PlainTextParser().parse(data, source_name="c.txt", mime_type="text/plain")
    import hashlib

    assert document.checksum == hashlib.sha256(data).hexdigest()


def test_cp1252_fallback_records_diagnostic():
    data = "caf\u00e9".encode("cp1252")  # not valid UTF-8 (0xE9 alone)
    document = PlainTextParser().parse(data, source_name="c.txt", mime_type="text/plain")

    assert any("cp1252" in note for note in document.diagnostics)
    assert "caf\u00e9" in document.blocks[0].text


def test_empty_text_is_not_an_error():
    document = PlainTextParser().parse(b"", source_name="empty.txt", mime_type="text/plain")

    assert document.blocks == []
    assert any("no extractable text" in note for note in document.diagnostics)


def test_unsupported_extension_is_rejected():
    with pytest.raises(UnsupportedFileTypeError):
        get_parser("malware.exe", "application/octet-stream")
    with pytest.raises(UnsupportedFileTypeError):
        mime_type_for("notes.docx")


def _make_pdf(text: str) -> bytes:
    import pymupdf

    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), text)
    data = document.tobytes()
    document.close()
    return data


def test_pdf_parser_records_page_numbers():
    data = _make_pdf("Clause 1 Termination for Convenience")
    document = PdfParser().parse(data, source_name="contract.pdf", mime_type="application/pdf")

    assert document.blocks, "expected at least one block"
    assert document.blocks[0].page == 1
    assert "Termination" in document.blocks[0].text


def test_malformed_pdf_raises_parsing_error():
    with pytest.raises(ParsingError):
        PdfParser().parse(b"this is not a pdf", source_name="bad.pdf", mime_type="application/pdf")


def test_empty_pdf_reports_no_extractable_text():
    import pymupdf

    document = pymupdf.open()
    document.new_page()
    data = document.tobytes()
    document.close()

    parsed = PdfParser().parse(data, source_name="blank.pdf", mime_type="application/pdf")
    assert parsed.blocks == []
    assert any("no extractable text" in note for note in parsed.diagnostics)
