"""Document parsers for plain text, Markdown and text-based PDFs.

Each parser returns a :class:`~rag_workbench.domain.schemas.ParsedDocument` of
ordered blocks carrying character offsets, an optional page number and an
optional heading path. Provenance is only ever filled in when it was genuinely
extractable; missing page/section data stays ``None``.

Limitations are deliberate and documented rather than hidden:

* Only text-extractable PDFs are supported. Scanned/image PDFs yield zero blocks
  and a diagnostic, never fabricated text.
* Heading detection combines Markdown ATX headings with a conservative numbered
  heading heuristic (``2.1 Definitions``). It is a heuristic, not a universal
  contract parser.
"""

from __future__ import annotations

import bisect
import hashlib
import re

from rag_workbench.config import DEFAULT_PARSER_VERSION
from rag_workbench.domain.errors import ParsingError, UnsupportedFileTypeError
from rag_workbench.domain.schemas import ParsedBlock, ParsedDocument

TEXT_EXTENSIONS = {".txt"}
MARKDOWN_EXTENSIONS = {".md", ".markdown"}
PDF_EXTENSIONS = {".pdf"}

MIME_BY_EXTENSION = {
    ".txt": "text/plain",
    ".md": "text/markdown",
    ".markdown": "text/markdown",
    ".pdf": "application/pdf",
}

_MD_HEADING = re.compile(r"^(#{1,6})\s+(\S.*?)\s*$")
# `2.1 Definitions of Terms` — a numbered label followed by a capitalised title
# that does not end a sentence.
_NUMBERED_HEADING = re.compile(r"^(\d+(?:\.\d+)*)[.)]?\s+([A-Z][^.]{0,90}?)\s*$")


def _decode_text(data: bytes) -> tuple[str, str | None]:
    """Decode bytes deterministically, recording the encoding actually used."""

    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if encoding == "utf-8-sig":
            return text, None
        return text, f"content was not valid UTF-8; decoded using {encoding}"
    # latin-1 never fails, so this is unreachable in practice.
    return data.decode("latin-1", errors="replace"), "content decoded with replacement characters"


def _detect_heading(line: str, *, allow_markdown: bool) -> tuple[int, str] | None:
    """Return ``(level, label)`` if the line looks like a heading."""

    stripped = line.strip()
    if not stripped:
        return None
    if allow_markdown:
        md = _MD_HEADING.match(stripped)
        if md:
            level = len(md.group(1))
            return level, md.group(2).strip()
    numbered = _NUMBERED_HEADING.match(stripped)
    if numbered:
        number = numbered.group(1)
        level = number.count(".") + 1
        # Use the original heading line so punctuation (e.g. "2.1") is preserved.
        return level, stripped
    return None


def _split_lines(text: str) -> list[tuple[str, int, int]]:
    """Yield ``(line, start, end)`` with offsets into ``text`` (newlines stripped)."""

    lines: list[tuple[str, int, int]] = []
    offset = 0
    for raw in text.splitlines(keepends=True):
        line = raw.rstrip("\r\n")
        lines.append((line, offset, offset + len(line)))
        offset += len(raw)
    return lines


def _parse_text(
    text: str,
    *,
    source_name: str,
    mime_type: str,
    checksum: str,
    allow_markdown: bool,
    page_spans: list[tuple[int, int, int]] | None = None,
) -> ParsedDocument:
    """Shared line-oriented parser used by the TXT/MD/PDF front-ends."""

    diagnostics: list[str] = []
    blocks: list[ParsedBlock] = []
    heading_stack: list[tuple[int, str]] = []

    paragraph_lines: list[tuple[str, int, int]] = []

    def current_section() -> str | None:
        if not heading_stack:
            return None
        return " > ".join(label for _, label in heading_stack)

    def flush_paragraph() -> None:
        if not paragraph_lines:
            return
        text_value = "\n".join(line for line, _, _ in paragraph_lines)
        if text_value.strip():
            char_start = paragraph_lines[0][1]
            char_end = paragraph_lines[-1][2]
            page = _page_for(char_start, page_spans)
            blocks.append(
                ParsedBlock(
                    text=text_value,
                    order=len(blocks),
                    page=page,
                    section_path=current_section(),
                    char_start=char_start,
                    char_end=char_end,
                )
            )
        paragraph_lines.clear()

    for line, start, end in _split_lines(text):
        heading = _detect_heading(line, allow_markdown=allow_markdown)
        if heading is not None:
            flush_paragraph()
            level, label = heading
            while heading_stack and heading_stack[-1][0] >= level:
                heading_stack.pop()
            heading_stack.append((level, label))
            continue
        if not line.strip():
            flush_paragraph()
            continue
        paragraph_lines.append((line, start, end))

    flush_paragraph()

    if not blocks:
        diagnostics.append("no extractable text blocks were found in this document")
    if page_spans is not None and not any(block.page is not None for block in blocks) and blocks:
        diagnostics.append("page numbers could not be associated with extracted blocks")

    return ParsedDocument(
        source_name=source_name,
        mime_type=mime_type,
        checksum=checksum,
        parser_version=DEFAULT_PARSER_VERSION,
        blocks=blocks,
        diagnostics=diagnostics,
        text_length=len(text),
    )


def _page_for(offset: int, page_spans: list[tuple[int, int, int]] | None) -> int | None:
    if not page_spans:
        return None
    starts = [span[0] for span in page_spans]
    index = bisect.bisect_right(starts, offset) - 1
    if index < 0:
        return page_spans[0][2]
    return page_spans[index][2]


class PlainTextParser:
    """Parse ``.txt`` content."""

    allow_markdown = False

    def parse(self, data: bytes, *, source_name: str, mime_type: str) -> ParsedDocument:
        checksum = hashlib.sha256(data).hexdigest()
        text, encoding_note = _decode_text(data)
        document = _parse_text(
            text,
            source_name=source_name,
            mime_type=mime_type,
            checksum=checksum,
            allow_markdown=self.allow_markdown,
        )
        if encoding_note:
            document.diagnostics.append(encoding_note)
        return document


class MarkdownParser(PlainTextParser):
    """Parse Markdown, treating ATX headings as section boundaries."""

    allow_markdown = True


class PdfParser:
    """Parse text-extractable PDFs using PyMuPDF, preserving page numbers."""

    def parse(self, data: bytes, *, source_name: str, mime_type: str) -> ParsedDocument:
        checksum = hashlib.sha256(data).hexdigest()
        fitz = _import_fitz()
        try:
            pdf = fitz.open(stream=data, filetype="pdf")
        except Exception as exc:  # noqa: BLE001 - PyMuPDF raises many exception types
            raise ParsingError(f"could not open PDF: {exc}") from exc

        diagnostics: list[str] = []
        page_texts: list[str] = []
        try:
            if getattr(pdf, "needs_pass", False):
                raise ParsingError("PDF is encrypted and cannot be parsed")
            for page_number, page in enumerate(pdf, start=1):
                try:
                    page_texts.append(page.get_text("text"))
                except Exception as exc:  # noqa: BLE001
                    page_texts.append("")
                    diagnostics.append(f"page {page_number} could not be extracted: {exc}")
        finally:
            pdf.close()

        if not any(text.strip() for text in page_texts):
            diagnostics.append(
                "PDF contains no extractable text (it may be a scanned or image-only PDF)"
            )

        full_text, page_spans = _join_pages(page_texts)
        document = _parse_text(
            full_text,
            source_name=source_name,
            mime_type=mime_type,
            checksum=checksum,
            allow_markdown=False,
            page_spans=page_spans,
        )
        document.diagnostics.extend(diagnostics)
        return document


def _join_pages(page_texts: list[str]) -> tuple[str, list[tuple[int, int, int]]]:
    """Join page texts with newlines and record each page's offset span."""

    parts: list[str] = []
    spans: list[tuple[int, int, int]] = []
    offset = 0
    for page_number, page_text in enumerate(page_texts, start=1):
        start = offset
        parts.append(page_text)
        offset += len(page_text) + 1  # +1 for the joining newline
        spans.append((start, offset, page_number))
    return "\n".join(parts), spans


def _import_fitz():  # noqa: ANN202 - returns the PyMuPDF module
    try:
        import pymupdf as fitz  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover - fallback for older PyMuPDF
        import fitz  # type: ignore[no-redef]
    return fitz


def mime_type_for(source_name: str) -> str:
    """Return the MIME type for a filename, or raise for unsupported types."""

    extension = _extension(source_name)
    if extension not in MIME_BY_EXTENSION:
        supported = ", ".join(sorted(MIME_BY_EXTENSION))
        raise UnsupportedFileTypeError(
            f"unsupported file type '{extension or source_name}'; supported: {supported}",
            code="unsupported_file_type",
        )
    return MIME_BY_EXTENSION[extension]


def _extension(source_name: str) -> str:
    lowered = source_name.lower()
    dot = lowered.rfind(".")
    return lowered[dot:] if dot != -1 else ""


def get_parser(source_name: str, mime_type: str):  # noqa: ANN201 - returns DocumentParser
    """Select a parser for a filename, validating the extension."""

    extension = _extension(source_name)
    if extension in PDF_EXTENSIONS:
        return PdfParser()
    if extension in MARKDOWN_EXTENSIONS:
        return MarkdownParser()
    if extension in TEXT_EXTENSIONS:
        return PlainTextParser()
    supported = ", ".join(sorted(TEXT_EXTENSIONS | MARKDOWN_EXTENSIONS | PDF_EXTENSIONS))
    raise UnsupportedFileTypeError(
        f"unsupported file type '{extension or source_name}'; supported: {supported}"
    )


__all__ = [
    "MIME_BY_EXTENSION",
    "MarkdownParser",
    "PdfParser",
    "PlainTextParser",
    "get_parser",
    "mime_type_for",
]
