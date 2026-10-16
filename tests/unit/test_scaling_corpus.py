"""Deterministic synthetic corpus generation."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_workbench.scaling.corpus import (
    CorpusConfig,
    corpus_fingerprint,
    generate_corpus,
    generate_document,
    size_class_counts,
    write_corpus,
)


def test_generation_is_deterministic_across_runs() -> None:
    config = CorpusConfig(documents=40, seed=7)

    first = generate_corpus(config)
    second = generate_corpus(config)

    assert [document.id for document in first] == [document.id for document in second]
    assert [document.text for document in first] == [document.text for document in second]
    assert corpus_fingerprint(first) == corpus_fingerprint(second)


def test_different_seeds_produce_different_text() -> None:
    first = generate_corpus(CorpusConfig(documents=10, seed=1))
    second = generate_corpus(CorpusConfig(documents=10, seed=2))

    assert corpus_fingerprint(first) != corpus_fingerprint(second)


def test_document_ids_are_stable_and_ordered() -> None:
    documents = generate_corpus(CorpusConfig(documents=3, seed=0, id_prefix="doc"))

    assert [document.id for document in documents] == [
        "doc-0000000",
        "doc-0000001",
        "doc-0000002",
    ]


def test_generate_document_matches_corpus_member() -> None:
    config = CorpusConfig(documents=5, seed=3)

    assert generate_document(config, 4).text == generate_corpus(config)[4].text


def test_corpus_mixes_short_and_long_documents() -> None:
    config = CorpusConfig(documents=200, seed=0)
    counts = size_class_counts(generate_corpus(config))

    assert counts["short"] > 0
    assert counts["typical"] > 0
    assert counts["large"] > 0
    assert sum(counts.values()) == 200


def test_every_document_has_a_distinctive_token() -> None:
    documents = generate_corpus(CorpusConfig(documents=30, seed=5))

    # Each document carries a per-document ``tok<index>`` family of tokens, so
    # retrieval queries built from them can be traced back to one document.
    for index, document in enumerate(documents):
        assert f"tok{index:05d}" in document.text


def test_write_corpus_creates_files_and_manifest(tmp_path: Path) -> None:
    documents = generate_corpus(CorpusConfig(documents=4, seed=0))
    manifest = write_corpus(documents, tmp_path)

    assert manifest["documents"] == 4
    assert manifest["fingerprint"] == corpus_fingerprint(documents)
    assert (tmp_path / "manifest.json").is_file()
    for document in documents:
        assert (tmp_path / f"{document.id}.txt").read_text(encoding="utf-8") == document.text


def test_config_rejects_non_positive_documents() -> None:
    with pytest.raises(ValueError):
        CorpusConfig(documents=0)


def test_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValueError):
        CorpusConfig(documents=1, unexpected=True)  # type: ignore[call-arg]
