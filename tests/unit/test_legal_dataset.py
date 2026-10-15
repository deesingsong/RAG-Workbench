"""Integrity checks for the synthetic legal-contract evaluation dataset.

These tests read the committed dataset under ``datasets/legal-contracts`` (no
network, no model) and assert its structural and provenance invariants, most
importantly that every evidence anchor is a verbatim substring of its contract.
"""

from __future__ import annotations

from pathlib import Path

from rag_workbench.evaluation.datasets import EVIDENCE_MAPPING, load_dataset

DATASET_DIR = Path(__file__).resolve().parents[2] / "datasets" / "legal-contracts"

# Queries chosen as pure paraphrases: their wording shares no guaranteed verbatim
# content words with the target clause. We assert their presence loosely.
PARAPHRASE_QUERY_IDS = {"lq-009", "lq-010", "lq-011"}


def _normalise(text: str) -> str:
    return " ".join(text.lower().split())


def test_legal_dataset_loads_with_expected_shape() -> None:
    dataset = load_dataset(DATASET_DIR)

    assert 6 <= len(dataset.documents) <= 8
    assert 40 <= len(dataset.queries) <= 48
    assert dataset.manifest.relevance_mapping == EVIDENCE_MAPPING
    assert dataset.manifest.corpus_count == len(dataset.documents)
    assert dataset.manifest.query_count == len(dataset.queries)
    assert dataset.manifest.evidence_count == len(dataset.evidence)
    assert dataset.manifest.qrel_count == sum(len(row) for row in dataset.qrels.values())


def test_every_query_has_relevant_units_that_exist() -> None:
    dataset = load_dataset(DATASET_DIR)

    query_ids = {query.id for query in dataset.queries}
    assert set(dataset.qrels) == query_ids

    for query_id in query_ids:
        judgments = dataset.relevance_for(query_id)
        assert judgments, f"query {query_id} has no relevant units"
        for unit_id in judgments:
            assert unit_id in dataset.evidence, f"qrels unit {unit_id} is missing from evidence"


def test_every_evidence_anchor_is_verbatim_in_its_contract() -> None:
    dataset = load_dataset(DATASET_DIR)

    documents = dataset.document_index()
    for unit in dataset.evidence.values():
        assert unit.id == f"{unit.contract}::{unit.section}"
        contract = documents[unit.contract]
        assert _normalise(unit.text) in _normalise(contract.text), (
            f"evidence {unit.id} is not a substring of contract {unit.contract}"
        )


def test_dataset_has_multi_unit_queries() -> None:
    dataset = load_dataset(DATASET_DIR)

    multi_unit = [qid for qid, rels in dataset.qrels.items() if len(rels) >= 2]
    assert multi_unit, "expected at least one query with two or more relevant units"
    # The primary unit of a multi-unit query is graded above the rest.
    for query_id in multi_unit:
        grades = sorted(dataset.relevance_for(query_id).values(), reverse=True)
        assert grades[0] == 2


def test_dataset_includes_pure_paraphrase_queries() -> None:
    dataset = load_dataset(DATASET_DIR)

    present = {query.id for query in dataset.queries}
    assert present >= PARAPHRASE_QUERY_IDS
