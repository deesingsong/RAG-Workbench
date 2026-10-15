"""Deterministic chunkers.

Token counting uses a **regex word tokenizer**: Unicode word characters form one
token, and each punctuation mark is its own token. This is a real, reproducible
token unit that needs no model or download — unlike a BPE tokenizer — so chunk
boundaries stay deterministic and testable offline. It is not identical to
GPT/Llama token counts, and that limitation is stated in the README.

Two strategies are implemented for Phase 1:

* ``fixed_token`` — sliding token windows of ``max_tokens`` with ``overlap_tokens``
  of overlap. Windows never cross a parsed block boundary, so page and section
  provenance stay exact.
* ``paragraph`` — recursive/paragraph-aware packing: whole paragraphs are packed
  up to the token budget, a paragraph that exceeds the budget alone is split with
  fixed windows, and a bounded trailing-token overlap is carried into the next
  chunk. Chunks never span a section boundary.
"""

from __future__ import annotations

import re
import uuid

from rag_workbench.config import ChunkerConfig, ChunkStrategy
from rag_workbench.domain.schemas import ChunkData, ParsedBlock, ParsedDocument

#: Namespace for deterministic chunk UUIDs (kept stable forever).
CHUNK_NAMESPACE = uuid.UUID("6f0a2c1e-9b2d-4f7a-8f0e-2c7b3d9a4e11")

TOKEN_PATTERN = re.compile(r"\w+|[^\w\s]", re.UNICODE)


def tokenize_spans(text: str) -> list[tuple[int, int]]:
    """Return ``(start, end)`` character spans for each token in ``text``."""

    return [(match.start(), match.end()) for match in TOKEN_PATTERN.finditer(text)]


def count_tokens(text: str) -> int:
    """Count tokens in ``text`` using the documented regex word tokenizer."""

    return sum(1 for _ in TOKEN_PATTERN.finditer(text))


def make_chunk_id(profile_key: str, document_id: str, order: int) -> str:
    """Deterministic, profile- and document-scoped chunk identifier."""

    name = f"rag-workbench:{profile_key}:{document_id}:{order}"
    return str(uuid.uuid5(CHUNK_NAMESPACE, name))


def _fixed_windows(
    text: str, *, max_tokens: int, overlap_tokens: int
) -> list[tuple[str, int, int | None, int | None]]:
    """Split ``text`` into overlapping token windows.

    Returns ``(window_text, token_count, char_start, char_end)`` tuples where the
    char offsets are relative to ``text`` and ``None`` when the window is empty.
    """

    spans = tokenize_spans(text)
    if not spans:
        return []

    step = max(1, max_tokens - overlap_tokens)
    windows: list[tuple[str, int, int | None, int | None]] = []
    start = 0
    total = len(spans)
    while start < total:
        end = min(start + max_tokens, total)
        window = spans[start:end]
        char_start = window[0][0]
        char_end = window[-1][1]
        windows.append((text[char_start:char_end], len(window), char_start, char_end))
        if end >= total:
            break
        start += step
    return windows


def _block_char_range(
    block: ParsedBlock, rel_start: int | None, rel_end: int | None
) -> tuple[int | None, int | None]:
    if block.char_start is None or rel_start is None or rel_end is None:
        return None, None
    return block.char_start + rel_start, block.char_start + rel_end


class FixedTokenChunker:
    """Sliding-window chunker that never crosses a parsed block boundary."""

    def chunk(
        self, document: ParsedDocument, config: ChunkerConfig, *, profile_key: str, document_id: str
    ) -> list[ChunkData]:
        chunks: list[ChunkData] = []
        for block in document.blocks:
            for text, tokens, rel_start, rel_end in _fixed_windows(
                block.text,
                max_tokens=config.max_tokens,
                overlap_tokens=config.overlap_tokens,
            ):
                char_start, char_end = _block_char_range(block, rel_start, rel_end)
                order = len(chunks)
                chunks.append(
                    ChunkData(
                        chunk_id=make_chunk_id(profile_key, document_id, order),
                        order=order,
                        text=text,
                        token_count=tokens,
                        section_path=block.section_path,
                        page_start=block.page,
                        page_end=block.page,
                        char_start=char_start,
                        char_end=char_end,
                    )
                )
        return chunks


class ParagraphChunker:
    """Paragraph-aware packing with a bounded trailing-token overlap."""

    def chunk(
        self, document: ParsedDocument, config: ChunkerConfig, *, profile_key: str, document_id: str
    ) -> list[ChunkData]:
        chunks: list[ChunkData] = []
        pending = _Pending.empty()

        for block in document.blocks:
            block_tokens = count_tokens(block.text)
            if block_tokens == 0:
                continue

            section_changed = pending.has_content and block.section_path != pending.section_path
            over_budget = pending.has_content and pending.tokens + block_tokens > config.max_tokens

            if section_changed or over_budget:
                finished = pending.finalize(len(chunks), document_id, profile_key)
                chunks.append(finished)
                pending = (
                    _Pending.empty()
                    if section_changed
                    else _Pending.carry(finished, config, pending)
                )

            if block_tokens > config.max_tokens:
                if pending.has_content:
                    chunks.append(pending.finalize(len(chunks), document_id, profile_key))
                    pending = _Pending.empty()
                for text, tokens, rel_start, rel_end in _fixed_windows(
                    block.text,
                    max_tokens=config.max_tokens,
                    overlap_tokens=config.overlap_tokens,
                ):
                    char_start, char_end = _block_char_range(block, rel_start, rel_end)
                    order = len(chunks)
                    chunks.append(
                        ChunkData(
                            chunk_id=make_chunk_id(profile_key, document_id, order),
                            order=order,
                            text=text,
                            token_count=tokens,
                            section_path=block.section_path,
                            page_start=block.page,
                            page_end=block.page,
                            char_start=char_start,
                            char_end=char_end,
                        )
                    )
                continue

            pending.add(block, block_tokens)

        if pending.has_content:
            chunks.append(pending.finalize(len(chunks), document_id, profile_key))
        return chunks


class _Pending:
    """Mutable accumulator for the paragraph chunker."""

    __slots__ = (
        "parts",
        "tokens",
        "section_path",
        "page_start",
        "page_end",
        "char_start",
        "char_end",
    )

    def __init__(
        self,
        *,
        parts: list[str],
        tokens: int,
        section_path: str | None,
        page_start: int | None,
        page_end: int | None,
        char_start: int | None,
        char_end: int | None,
    ) -> None:
        self.parts = parts
        self.tokens = tokens
        self.section_path = section_path
        self.page_start = page_start
        self.page_end = page_end
        self.char_start = char_start
        self.char_end = char_end

    @classmethod
    def empty(cls) -> _Pending:
        return cls(
            parts=[],
            tokens=0,
            section_path=None,
            page_start=None,
            page_end=None,
            char_start=None,
            char_end=None,
        )

    @classmethod
    def carry(cls, finished: ChunkData, config: ChunkerConfig, previous: _Pending) -> _Pending:
        """Seed the next chunk with up to ``overlap_tokens`` trailing tokens."""

        if config.overlap_tokens <= 0:
            return cls.empty()
        spans = tokenize_spans(finished.text)
        if len(spans) <= config.overlap_tokens:
            return cls.empty()
        rel_start = spans[-config.overlap_tokens][0]
        prefix = finished.text[rel_start:]
        char_start = (
            finished.char_start + rel_start if finished.char_start is not None else None
        )
        return cls(
            parts=[prefix],
            tokens=config.overlap_tokens,
            section_path=finished.section_path,
            page_start=finished.page_start,
            page_end=finished.page_end,
            char_start=char_start,
            char_end=finished.char_end,
        )

    @property
    def has_content(self) -> bool:
        return bool(self.parts)

    def add(self, block: ParsedBlock, tokens: int) -> None:
        if not self.parts:
            self.section_path = block.section_path
            self.char_start = block.char_start
            self.char_end = block.char_end
            self.page_start = block.page
            self.page_end = block.page
        else:
            if block.page is not None:
                self.page_start = (
                    block.page
                    if self.page_start is None
                    else min(self.page_start, block.page)
                )
                self.page_end = (
                    block.page if self.page_end is None else max(self.page_end, block.page)
                )
            if block.char_end is not None:
                self.char_end = block.char_end
        self.parts.append(block.text)
        self.tokens += tokens

    def finalize(self, order: int, document_id: str, profile_key: str) -> ChunkData:
        text = "\n\n".join(self.parts)
        return ChunkData(
            chunk_id=make_chunk_id(profile_key, document_id, order),
            order=order,
            text=text,
            token_count=self.tokens,
            section_path=self.section_path,
            page_start=self.page_start,
            page_end=self.page_end,
            char_start=self.char_start,
            char_end=self.char_end,
        )


# ---------------------------------------------------------------------------
# Clause-aware legal chunking
# ---------------------------------------------------------------------------
#
# Conservative, deterministic clause detection. This is a structural
# *heuristic*, not a contract interpreter: it recognises common English legal
# layout (numbered sections, ``(a)`` subsections, ``ARTICLE`` headings,
# schedules/annexes, and definition entries) and otherwise falls back to the
# parser-provided section path. Unrecognised text is never discarded.

#: Maximum number of words for a line to count as a clause *title*.
_TITLE_TOKEN_LIMIT = 14

_ARTICLE_MARKER = re.compile(r"^ARTICLE\s+([IVXLCDM]+|\d+)\b[.:]?\s*(.*)$", re.IGNORECASE)
_SCHEDULE_MARKER = re.compile(
    r"^(SCHEDULE|ANNEX|APPENDIX|EXHIBIT)\s+([A-Z0-9][A-Z0-9IVX.-]*)\b[.:]?\s*(.*)$",
    re.IGNORECASE,
)
_NUMBERED_MARKER = re.compile(r"^(\d+(?:\.\d+)*)(?:\(([a-z])\))?[.)]?\s+(\S.*)$")
_ALPHA_MARKER = re.compile(r"^\(([a-z])\)\s*(.*)$")
_DEFINITION_MARKER = re.compile(
    r'^(?:(?P<number>\d+(?:\.\d+)*)[.)]?\s+)?'
    r'"(?P<term>[^"]{2,160})"\s+(?:means|includes|shall mean|has the meaning)\b',
    re.IGNORECASE,
)

#: Splits a stored section-path label into its number and title.
_PATH_LABEL = re.compile(
    r"^(?P<number>"
    r"\d+(?:\.\d+)*(?:\([a-z]\))?"
    r"|ARTICLE\s+\S+"
    r"|SCHEDULE\s+\S+"
    r"|ANNEX\s+\S+"
    r"|APPENDIX\s+\S+"
    r"|EXHIBIT\s+\S+"
    r"|\([a-z]\)"
    r")[.)]?\s*(?P<title>.*)$",
    re.IGNORECASE,
)


def _label_level(label: str, fallback: int) -> int:
    """Derive a nesting level for one section-path label."""

    number, _title = _split_label(label)
    if number:
        base = number.split("(")[0]
        if re.fullmatch(r"\d+(?:\.\d+)*", base):
            return base.count(".") + 1 + (1 if "(" in number else 0)
    return fallback


def _title_like(text: str) -> bool:
    """Conservatively decide whether ``text`` looks like a clause title."""

    stripped = text.strip().rstrip(".").strip()
    if not stripped:
        return False
    if len(stripped.split()) > _TITLE_TOKEN_LIMIT:
        return False
    if not stripped[0].isupper():
        return False
    # A sentence-like interior (``. ``/``; ``/``: `` followed by a lowercase word)
    # means prose, not a heading.
    return not re.search(r"[.;:]\s+[a-z]", stripped)


def detect_clause_marker(line: str) -> tuple[int | None, str, str | None, str | None] | None:
    """Detect a clause marker at the start of ``line``.

    Returns ``(level, label, number, title)`` where ``level`` is the intended
    nesting depth (``None`` for an ``(a)`` subsection, which takes the parent's
    depth + 1), or ``None`` when the line is not a recognised clause marker.
    Only conservative, high-confidence patterns are recognised so ordinary
    numbers, dates, amounts and decimals are not promoted to headings.
    """

    stripped = line.strip()
    if not stripped:
        return None

    article = _ARTICLE_MARKER.match(stripped)
    if article and (not article.group(2).strip() or _title_like(article.group(2))):
        number = f"ARTICLE {article.group(1).upper()}"
        title = article.group(2).strip() or None
        return 1, (f"{number} {title}".strip() if title else number), number, title

    schedule = _SCHEDULE_MARKER.match(stripped)
    if schedule and (not schedule.group(3).strip() or _title_like(schedule.group(3))):
        number = f"{schedule.group(1).upper()} {schedule.group(2)}"
        title = schedule.group(3).strip() or None
        return 1, (f"{number} {title}".strip() if title else number), number, title

    alpha = _ALPHA_MARKER.match(stripped)
    if alpha:
        # A standalone ``(a)`` is a subsection boundary. The remainder is usually
        # the subsection *text*, not a title, so it is never promoted to one.
        number = f"({alpha.group(1)})"
        return None, number, number, None

    numbered = _NUMBERED_MARKER.match(stripped)
    if numbered:
        remainder = numbered.group(3)
        # Require an uppercase *letter* start: rejects "(three) days", dates,
        # amounts and other numeric prose while accepting "Payment Terms".
        if remainder[:1].isupper() and _title_like(remainder):
            base = numbered.group(1)
            letter = numbered.group(2)
            number = f"{base}({letter})" if letter else base
            depth = base.count(".") + 1 + (1 if letter else 0)
            title = remainder.strip()
            return depth, f"{number} {title}", number, title
    return None


def _definition_info(text: str) -> tuple[str | None, str] | None:
    """Return ``(number, term)`` when ``text`` is a definition entry, else ``None``."""

    match = _DEFINITION_MARKER.match(text.strip())
    if match is None:
        return None
    number = match.group("number")
    return (number.strip() if number else None), match.group("term").strip()


def _labels_from_section_path(section_path: str | None) -> list[str]:
    if not section_path:
        return []
    return [part.strip() for part in section_path.split(" > ") if part.strip()]


def _split_label(label: str) -> tuple[str | None, str | None]:
    """Return ``(number, title)`` for one section-path label."""

    match = _PATH_LABEL.match(label.strip())
    if match:
        number = match.group("number").strip()
        title = match.group("title").strip() or None
        return number, title
    return None, label.strip() or None


class _ClauseUnit:
    """One block annotated with its clause context."""

    __slots__ = (
        "text",
        "tokens",
        "labels",
        "page_start",
        "page_end",
        "char_start",
        "char_end",
        "is_definition",
        "is_heading_only",
        "diagnostics",
    )

    def __init__(
        self,
        *,
        text: str,
        tokens: int,
        labels: list[str],
        page_start: int | None,
        page_end: int | None,
        char_start: int | None,
        char_end: int | None,
        is_definition: bool,
        is_heading_only: bool,
        diagnostics: list[str],
    ) -> None:
        self.text = text
        self.tokens = tokens
        self.labels = labels
        self.page_start = page_start
        self.page_end = page_end
        self.char_start = char_start
        self.char_end = char_end
        self.is_definition = is_definition
        self.is_heading_only = is_heading_only
        self.diagnostics = diagnostics

    @property
    def section_path(self) -> str | None:
        return " > ".join(self.labels) if self.labels else None


def _build_clause_units(document: ParsedDocument) -> list[_ClauseUnit]:
    """Annotate each block with the clause hierarchy it belongs to."""

    units: list[_ClauseUnit] = []
    stack: list[tuple[int, str]] = []
    current_section_path: str | None = None
    definition_level: int | None = None

    for block in document.blocks:
        if block.section_path and block.section_path != current_section_path:
            current_section_path = block.section_path
            definition_level = None
            labels = _labels_from_section_path(block.section_path)
            levels = [_label_level(label, index + 1) for index, label in enumerate(labels)]
            base_level = levels[0] if levels else 1
            # Keep any shallower ancestors (e.g. an ARTICLE heading) that the
            # parser-level path does not itself encode.
            stack = [(lvl, lbl) for lvl, lbl in stack if lvl < base_level]
            for level, label in zip(levels, labels, strict=True):
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, label))

        definition = _definition_info(block.text)
        marker = None
        heading_only = False
        if definition is not None:
            # Retire a previous definition's synthetic label, then give this
            # definition its own identity, e.g. ``1.1 "Confidential Information"``.
            if definition_level is not None:
                while stack and stack[-1][0] >= definition_level:
                    stack.pop()
            number, term = definition
            if number:
                level = number.count(".") + 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, f'{number} "{term}"'))
                definition_level = level
            else:
                definition_level = None
        else:
            if definition_level is not None:
                while stack and stack[-1][0] >= definition_level:
                    stack.pop()
                definition_level = None
            marker = detect_clause_marker(block.text.split("\n", 1)[0])
            if marker is not None:
                level, label, number, _title = marker
                # A bare marker line (no title/body, e.g. "ARTICLE IV") is a
                # heading that must not become a chunk on its own.
                heading_only = number is not None and label == number
                if level is None:  # (a) subsection: one level below the current deepest
                    level = (stack[-1][0] + 1) if stack else 1
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, label))

        labels = [part for _, part in stack]
        units.append(
            _ClauseUnit(
                text=block.text,
                tokens=count_tokens(block.text),
                labels=labels,
                page_start=block.page,
                page_end=block.page,
                char_start=block.char_start,
                char_end=block.char_end,
                is_definition=definition is not None,
                is_heading_only=heading_only,
                diagnostics=[],
            )
        )
    return units


class _ClauseBuilder:
    """Accumulator for one clause-aware chunk."""

    __slots__ = (
        "parts",
        "tokens",
        "labels",
        "page_start",
        "page_end",
        "char_start",
        "char_end",
        "diagnostics",
        "has_body",
        "has_units",
        "has_definition",
    )

    def __init__(
        self,
        *,
        labels: list[str] | None = None,
        page_start: int | None = None,
        page_end: int | None = None,
        char_start: int | None = None,
        char_end: int | None = None,
    ) -> None:
        self.parts: list[str] = []
        self.tokens = 0
        self.labels: list[str] = list(labels or [])
        self.page_start = page_start
        self.page_end = page_end
        self.char_start = char_start
        self.char_end = char_end
        self.diagnostics: list[str] = []
        self.has_body = False
        self.has_units = False
        self.has_definition = False

    @property
    def has_content(self) -> bool:
        return bool(self.parts)

    @property
    def section_path(self) -> str | None:
        return " > ".join(self.labels) if self.labels else None

    def add(self, unit: _ClauseUnit) -> None:
        self.has_units = True
        if not self.parts:
            self.labels = list(unit.labels)
            self.char_start = unit.char_start
            self.page_start = unit.page_start
        if unit.page_start is not None:
            self.page_start = (
                unit.page_start
                if self.page_start is None
                else min(self.page_start, unit.page_start)
            )
        if unit.page_end is not None:
            self.page_end = (
                unit.page_end if self.page_end is None else max(self.page_end, unit.page_end)
            )
        if unit.char_end is not None:
            self.char_end = unit.char_end
        self.parts.append(unit.text)
        self.tokens += unit.tokens
        if unit.tokens > 0 and not unit.is_heading_only:
            self.has_body = True
        if unit.is_definition:
            self.has_definition = True
        for note in unit.diagnostics:
            if note not in self.diagnostics:
                self.diagnostics.append(note)

    def finalize(self, order: int, document_id: str, profile_key: str) -> ChunkData:
        labels = self.labels
        number: str | None = None
        title: str | None = None
        if labels:
            number, title = _split_label(labels[-1])
        parent = " > ".join(labels[:-1]) if len(labels) > 1 else None
        diagnostics = list(self.diagnostics)
        if not self.has_body:
            diagnostics.append("heading-only chunk: no body text followed this section heading")
        return ChunkData(
            chunk_id=make_chunk_id(profile_key, document_id, order),
            order=order,
            text="\n\n".join(self.parts),
            token_count=self.tokens,
            section_path=self.section_path,
            page_start=self.page_start,
            page_end=self.page_end,
            char_start=self.char_start,
            char_end=self.char_end,
            section_number=number,
            section_title=title,
            parent_clause=parent,
            section_level=len(labels) if labels else None,
            diagnostics=diagnostics,
        )


class ClauseAwareChunker:
    """Structure-aware chunker for English-language legal documents.

    It preserves logical clause boundaries, groups the paragraphs of a clause
    while they fit the token budget, splits long clauses without losing their
    parent section identity, and carries a bounded contextual overlap only
    *within* the same clause. Provenance is only recorded when it could be
    derived deterministically.
    """

    def chunk(
        self, document: ParsedDocument, config: ChunkerConfig, *, profile_key: str, document_id: str
    ) -> list[ChunkData]:
        units = _build_clause_units(document)
        chunks: list[ChunkData] = []
        builder = _ClauseBuilder()
        previous: ChunkData | None = None

        def flush() -> None:
            nonlocal builder, previous
            if builder.has_units:
                finished = builder.finalize(len(chunks), document_id, profile_key)
                chunks.append(finished)
                previous = finished
            builder = _ClauseBuilder()

        def seed_overlap(new_labels: list[str], path: str | None) -> _ClauseBuilder:
            started = _ClauseBuilder(labels=new_labels)
            if (
                config.overlap_tokens > 0
                and previous is not None
                and previous.section_path == path
                and path is not None
            ):
                spans = tokenize_spans(previous.text)
                if len(spans) > config.overlap_tokens:
                    rel_start = spans[-config.overlap_tokens][0]
                    prefix = previous.text[rel_start:]
                    char_start = (
                        previous.char_end - len(prefix)
                        if previous.char_end is not None
                        else None
                    )
                    started.parts.append(prefix)
                    started.tokens = config.overlap_tokens
                    started.char_start = char_start
                    started.char_end = previous.char_end
                    started.page_start = previous.page_start
                    started.page_end = previous.page_end
            return started

        for unit in units:
            if unit.tokens == 0:
                continue
            unit_path = unit.section_path

            if builder.has_content and (unit_path != builder.section_path):
                # A pure overlap carry-over is dropped rather than emitted alone.
                if builder.has_units:
                    flush()
                else:
                    builder = _ClauseBuilder()
            if builder.has_units and builder.has_definition and unit.is_definition:
                # Keep each definition entry with its own explanatory text.
                flush()
            if not builder.has_content:
                builder = seed_overlap(list(unit.labels), unit_path)

            if builder.has_content and builder.tokens + unit.tokens > config.max_tokens:
                if builder.has_units:
                    flush()
                    builder = seed_overlap(list(unit.labels), unit_path)
                else:
                    builder = _ClauseBuilder(labels=list(unit.labels))

            if unit.tokens > config.max_tokens:
                if builder.has_units:
                    flush()
                builder = _ClauseBuilder(labels=list(unit.labels))
                windows = _fixed_windows(
                    unit.text,
                    max_tokens=config.max_tokens,
                    overlap_tokens=config.overlap_tokens,
                )
                number, title = _split_label(unit.labels[-1]) if unit.labels else (None, None)
                parent = " > ".join(unit.labels[:-1]) if len(unit.labels) > 1 else None
                for index, (text, tokens, rel_start, rel_end) in enumerate(windows):
                    char_start = (
                        unit.char_start + rel_start
                        if unit.char_start is not None and rel_start is not None
                        else None
                    )
                    char_end = (
                        unit.char_start + rel_end
                        if unit.char_start is not None and rel_end is not None
                        else None
                    )
                    order = len(chunks)
                    chunks.append(
                        ChunkData(
                            chunk_id=make_chunk_id(profile_key, document_id, order),
                            order=order,
                            text=text,
                            token_count=tokens,
                            section_path=unit_path,
                            page_start=unit.page_start,
                            page_end=unit.page_end,
                            char_start=char_start,
                            char_end=char_end,
                            section_number=number,
                            section_title=title,
                            parent_clause=parent,
                            section_level=len(unit.labels) if unit.labels else None,
                            diagnostics=[
                                f"split: clause exceeded the {config.max_tokens}-token budget "
                                f"(part {index + 1} of {len(windows)})"
                            ],
                        )
                    )
                previous = chunks[-1] if chunks else None
                builder = _ClauseBuilder()
                continue

            builder.add(unit)

        flush()
        return chunks


_CHUNKERS: dict[ChunkStrategy, type] = {
    "fixed_token": FixedTokenChunker,
    "paragraph": ParagraphChunker,
    "clause_aware": ClauseAwareChunker,
}


def get_chunker(config: ChunkerConfig):  # noqa: ANN201 - returns a Chunker
    """Instantiate the chunker named by ``config.strategy``."""

    return _CHUNKERS[config.strategy]()


__all__ = [
    "CHUNK_NAMESPACE",
    "ClauseAwareChunker",
    "FixedTokenChunker",
    "ParagraphChunker",
    "count_tokens",
    "detect_clause_marker",
    "get_chunker",
    "make_chunk_id",
    "tokenize_spans",
]
