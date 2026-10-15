"""Dataset import, validation and round-trip tests (offline, synthetic fixtures)."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag_workbench.domain.errors import DatasetError
from rag_workbench.evaluation.datasets import (
    EVIDENCE_MAPPING,
    DatasetStore,
    load_beir,
    load_dataset,
    parse_qrels,
    write_dataset,
)


def _write_beir(directory: Path) -> tuple[Path, Path, Path]:
    corpus = directory / "corpus.jsonl"
    corpus.write_text(
        '{"_id": "D1", "title": "Alpha", "text": "alpha beta gamma"}\n'
        '{"_id": "D2", "title": "Beta", "text": "delta epsilon"}\n',
        encoding="utf-8",
    )
    queries = directory / "queries.jsonl"
    queries.write_text(
        '{"_id": "Q1", "text": "alpha"}\n{"_id": "Q2", "text": "delta"}\n',
        encoding="utf-8",
    )
    qrels = directory / "qrels.tsv"
    qrels.write_text(
        "query-id\tcorpus-id\tscore\nQ1\tD1\t1\nQ2\tD2\t1\n",
        encoding="utf-8",
    )
    return corpus, queries, qrels


def test_beir_import_and_round_trip(tmp_path: Path) -> None:
    corpus, queries, qrels = _write_beir(tmp_path)
    dataset = load_beir(
        name="synthetic",
        corpus_path=corpus,
        queries_path=queries,
        qrels_path=qrels,
        source="synthetic-test",
        license="CC0",
    )
    assert dataset.manifest.corpus_count == 2
    assert dataset.manifest.query_count == 2
    assert dataset.manifest.qrel_count == 2
    assert dataset.relevance_for("Q1") == {"D1": 1}

    out = tmp_path / "stored"
    manifest = write_dataset(out, dataset)
    assert manifest.corpus_sha256
    reloaded = load_dataset(out)
    assert reloaded.manifest.checksum == manifest.checksum
    assert {d.id for d in reloaded.documents} == {"D1", "D2"}
    assert reloaded.qrels == dataset.qrels


def test_missing_qrels_file_raises(tmp_path: Path) -> None:
    corpus, queries, _ = _write_beir(tmp_path)
    with pytest.raises(DatasetError):
        load_beir(
            name="synthetic",
            corpus_path=corpus,
            queries_path=queries,
            qrels_path=tmp_path / "nope.tsv",
            source="x",
            license="x",
        )


@pytest.mark.parametrize(
    "body",
    [
        "query-id\tcorpus-id\tscore\nQ1\tD1\tnotanint\n",
        "query-id\tcorpus-id\tscore\nQ1\n",
        "",
    ],
)
def test_malformed_qrels_rejected(tmp_path: Path, body: str) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text(body, encoding="utf-8")
    with pytest.raises(DatasetError):
        parse_qrels(path)


def test_qrels_without_header_two_columns(tmp_path: Path) -> None:
    path = tmp_path / "qrels.tsv"
    path.write_text("Q1\tD1\nQ1\tD2\n", encoding="utf-8")
    assert parse_qrels(path) == {"Q1": {"D1": 1, "D2": 1}}


def test_qrels_referencing_unknown_query_rejected(tmp_path: Path) -> None:
    corpus, queries, _ = _write_beir(tmp_path)
    bad_qrels = tmp_path / "bad.tsv"
    bad_qrels.write_text("query-id\tcorpus-id\tscore\nGHOST\tD1\t1\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_beir(
            name="synthetic",
            corpus_path=corpus,
            queries_path=queries,
            qrels_path=bad_qrels,
            source="x",
            license="x",
        )


def test_qrels_referencing_unknown_unit_rejected(tmp_path: Path) -> None:
    corpus, queries, _ = _write_beir(tmp_path)
    bad_qrels = tmp_path / "bad.tsv"
    bad_qrels.write_text("query-id\tcorpus-id\tscore\nQ1\tD9\t1\n", encoding="utf-8")
    with pytest.raises(DatasetError):
        load_beir(
            name="synthetic",
            corpus_path=corpus,
            queries_path=queries,
            qrels_path=bad_qrels,
            source="x",
            license="x",
        )


def test_queries_without_judgments_are_allowed(tmp_path: Path) -> None:
    """A qrels subset is valid: queries with no judgments are excluded, not errors."""

    corpus, queries, _ = _write_beir(tmp_path)
    subset = tmp_path / "subset.tsv"
    subset.write_text("query-id\tcorpus-id\tscore\nQ1\tD1\t1\n", encoding="utf-8")
    dataset = load_beir(
        name="x",
        corpus_path=corpus,
        queries_path=queries,
        qrels_path=subset,
        source="x",
        license="x",
    )
    assert dataset.manifest.query_count == 2
    assert dataset.relevance_for("Q2") == {}


def test_evidence_mapping_requires_known_contract(tmp_path: Path) -> None:
    corpus = tmp_path / "corpus.jsonl"
    corpus.write_text('{"_id": "C1", "title": null, "text": "confidential"}\n', encoding="utf-8")
    queries = tmp_path / "queries.jsonl"
    queries.write_text('{"_id": "Q1", "text": "confidential"}\n', encoding="utf-8")
    qrels = tmp_path / "qrels.tsv"
    qrels.write_text("query-id\tcorpus-id\tscore\nQ1\tC1::1\t1\n", encoding="utf-8")
    evidence = tmp_path / "evidence.jsonl"
    evidence.write_text(
        '{"_id": "C1::1", "contract": "GHOST", "section": "1", "text": "confidential"}\n',
        encoding="utf-8",
    )
    with pytest.raises(DatasetError):
        load_beir(
            name="legal",
            corpus_path=corpus,
            queries_path=queries,
            qrels_path=qrels,
            evidence_path=evidence,
            source="x",
            license="x",
            relevance_mapping=EVIDENCE_MAPPING,
        )


def test_dataset_store_discovers_and_loads(tmp_path: Path) -> None:
    corpus, queries, qrels = _write_beir(tmp_path)
    dataset = load_beir(
        name="Synthetic Set",
        corpus_path=corpus,
        queries_path=queries,
        qrels_path=qrels,
        source="x",
        license="x",
    )
    root = tmp_path / "datasets"
    write_dataset(root / "synthetic-set", dataset)
    store = DatasetStore([root])
    assert store.list_names() == ["synthetic-set"]
    assert store.exists("Synthetic Set")
    loaded = store.load("synthetic-set")
    assert loaded.name == "Synthetic Set"
