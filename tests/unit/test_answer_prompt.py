"""Unit tests for prompt construction and the retrieved-data trust boundary."""

from __future__ import annotations

from rag_workbench.answering.prompt import (
    ANSWER_JSON_SCHEMA,
    build_instructions,
    build_user_message,
    render_evidence_block,
)
from rag_workbench.domain.schemas import EvidenceItem


def _evidence() -> list[EvidenceItem]:
    return [
        EvidenceItem(
            source_id="S1",
            chunk_id="c1",
            document_id="d1",
            source_name="nda.txt",
            section_number="6",
            section_path="6. Term",
            page_start=2,
            rank=1,
            score=0.0,
            text="This Agreement remains in effect for three years.",
        )
    ]


def test_schema_excludes_application_only_states() -> None:
    statuses = ANSWER_JSON_SCHEMA["properties"]["status"]["enum"]  # type: ignore[index]
    assert "generation_error" not in statuses
    assert "insufficient_evidence" in statuses


def test_instructions_label_passages_as_untrusted() -> None:
    instructions = build_instructions()
    assert "untrusted" in instructions.lower()
    assert "insufficient_evidence" in instructions
    assert "cite" in instructions.lower()


def test_retrieved_text_is_data_not_instructions() -> None:
    evidence = _evidence()
    user_message = build_user_message("How long is the term?", evidence)
    instructions = build_instructions()
    # The passage appears only in the delimited user message, never in the
    # trusted system instructions.
    assert evidence[0].text in user_message
    assert evidence[0].text not in instructions
    assert "BEGIN SOURCE PASSAGES" in user_message
    assert "END SOURCE PASSAGES" in user_message
    assert "=== PASSAGE S1 ===" in user_message


def test_evidence_block_carries_provenance() -> None:
    block = render_evidence_block(_evidence())
    assert "file: nda.txt" in block
    assert "page: 2" in block
    assert "section: 6. Term" in block
