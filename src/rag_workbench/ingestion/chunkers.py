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


_CHUNKERS: dict[ChunkStrategy, type] = {
    "fixed_token": FixedTokenChunker,
    "paragraph": ParagraphChunker,
}


def get_chunker(config: ChunkerConfig):  # noqa: ANN201 - returns a Chunker
    """Instantiate the chunker named by ``config.strategy``."""

    return _CHUNKERS[config.strategy]()


__all__ = [
    "CHUNK_NAMESPACE",
    "FixedTokenChunker",
    "ParagraphChunker",
    "count_tokens",
    "get_chunker",
    "make_chunk_id",
    "tokenize_spans",
]
