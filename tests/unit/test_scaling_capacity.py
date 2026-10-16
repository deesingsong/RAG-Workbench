"""Deterministic capacity-planning model."""

from __future__ import annotations

import pytest

from rag_workbench.scaling.capacity import (
    IngestionAssumptions,
    StorageAssumptions,
    estimate_ingestion,
    estimate_storage,
    human_bytes,
    storage_scenarios,
)


def _flat_storage() -> StorageAssumptions:
    """A fully-specified example with all overheads disabled, so the total is
    plain arithmetic the test computes by hand."""

    return StorageAssumptions(
        documents=1000,
        chunks_per_document=2,
        source_document_kb=10,
        chunk_text_kb=1,
        metadata_bytes_per_chunk=100,
        embedding_dimensions=4,
        bytes_per_dimension=4,
        vector_indexes=1,
        replication_factor=1,
        vector_index_overhead_factor=1.0,
        payload_index_overhead_factor=1.0,
        lexical_index_ratio=0.0,
        backup_copies=0,
        temp_reindex_factor=0.0,
        operational_headroom=0.0,
    )


def test_storage_arithmetic_is_exact_under_assumptions() -> None:
    estimate = estimate_storage(_flat_storage())

    # 10_240_000 source + 2_048_000 chunk text + 200_000 metadata + 32_000 vectors
    assert estimate.total_bytes == 12_520_000
    assert estimate.primary_bytes == 12_520_000
    assert estimate.replica_bytes == 0
    assert estimate.chunks_total == 2000


def test_storage_multipliers_stack() -> None:
    assumptions = _flat_storage().model_copy(
        update={
            "replication_factor": 2,
            "backup_copies": 1,
            "temp_reindex_factor": 1.0,
            "operational_headroom": 0.25,
        }
    )
    estimate = estimate_storage(assumptions)
    primary = estimate.primary_bytes

    assert estimate.replica_bytes == primary
    assert estimate.backup_bytes == primary
    assert estimate.reindex_bytes == primary
    subtotal = primary * 4
    assert estimate.headroom_bytes == int(subtotal * 0.25)
    assert estimate.total_bytes == subtotal + int(subtotal * 0.25)


def test_every_component_declares_a_basis() -> None:
    estimate = estimate_storage(StorageAssumptions(documents=1_000_000))
    assert estimate.components
    for component in estimate.components:
        assert component.basis in {"exact", "measured", "assumed", "documented", "unknown"}


def test_overhead_multipliers_are_assumptions_not_documented() -> None:
    # The +50% vector, +20% payload and 35% lexical ratios are illustrative
    # defaults with no identifiable external reference in this project, so they
    # must be labelled "assumed", never "documented" or "measured".
    estimate = estimate_storage(_flat_storage())
    by_key = {component.key: component for component in estimate.components}

    assert by_key["raw_vectors"].basis == "exact"
    assert by_key["vector_index_overhead"].basis == "assumed"
    assert by_key["payload_index"].basis == "assumed"
    assert by_key["lexical_index"].basis == "assumed"


def test_default_throughput_is_assumed_not_measured() -> None:
    assert IngestionAssumptions().throughput_basis == "assumed"
    estimate = estimate_ingestion(IngestionAssumptions())
    assert estimate.basis == "assumed"
    assert any("conditional projection" in note for note in estimate.notes)
    assert any("assumed" in note for note in estimate.notes)


def test_throughput_basis_is_caller_selectable() -> None:
    estimate = estimate_ingestion(IngestionAssumptions(throughput_basis="measured"))
    assert estimate.basis == "measured"


def test_storage_scenarios_change_chunk_count() -> None:
    scenarios = storage_scenarios(StorageAssumptions(documents=10_000_000))

    assert set(scenarios) == {"small", "typical", "large"}
    assert (
        scenarios["small"].chunks_per_document
        < scenarios["typical"].chunks_per_document
        < scenarios["large"].chunks_per_document
    )


def test_ingestion_keeps_up_when_capacity_exceeds_demand() -> None:
    estimate = estimate_ingestion(
        IngestionAssumptions(
            documents_per_day=1000,
            chunks_per_document=10,
            embedding_chunks_per_second_per_worker=10,
            indexing_chunks_per_second=200,
            workers=1,
            retry_overhead=0.0,
        )
    )

    assert estimate.chunks_per_day == 10_000
    assert estimate.bottleneck_stage == "embedding"
    assert estimate.keeps_up is True
    assert estimate.backlog_per_day == 0.0


def test_ingestion_reports_backlog_when_demand_exceeds_capacity() -> None:
    estimate = estimate_ingestion(
        IngestionAssumptions(
            documents_per_day=1_000_000,
            chunks_per_document=10,
            embedding_chunks_per_second_per_worker=1,
            indexing_chunks_per_second=1,
            workers=1,
            retry_overhead=0.0,
        )
    )

    assert estimate.keeps_up is False
    assert estimate.backlog_per_day > 0
    assert any("backlog" in note.lower() for note in estimate.notes)


def test_worker_scaling_is_non_linear() -> None:
    estimate = estimate_ingestion(
        IngestionAssumptions(workers=4, worker_scaling_efficiency=0.5)
    )
    # 1 + (4 - 1) * 0.5 = 2.5 effective workers, never 4.
    assert estimate.effective_workers == pytest.approx(2.5)


def test_human_bytes_units() -> None:
    assert human_bytes(0) == "0.0 B"
    assert human_bytes(2048) == "2.0 KB"
    assert human_bytes(1024**3) == "1.0 GB"


def test_assumptions_reject_negative_values() -> None:
    with pytest.raises(ValueError):
        StorageAssumptions(documents=0)


def test_assumptions_reject_non_finite_values() -> None:
    # Public calculator inputs must not accept NaN/Infinity, which would make the
    # arithmetic meaningless (and serialize as invalid JSON downstream).
    with pytest.raises(ValueError):
        StorageAssumptions(chunks_per_document=float("inf"))
    with pytest.raises(ValueError):
        StorageAssumptions(operational_headroom=float("nan"))
    with pytest.raises(ValueError):
        IngestionAssumptions(embedding_chunks_per_second_per_worker=float("inf"))
    with pytest.raises(ValueError):
        IngestionAssumptions(worker_scaling_efficiency=float("nan"))


def test_assumptions_reject_unreasonable_values() -> None:
    with pytest.raises(ValueError):
        StorageAssumptions(chunks_per_document=1_000_000_000)
    with pytest.raises(ValueError):
        IngestionAssumptions(embedding_chunks_per_second_per_worker=10_000_000_000)
