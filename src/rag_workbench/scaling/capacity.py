"""Deterministic capacity-planning model for a 10-million-document deployment.

This is deliberate arithmetic, not a simulation and not a benchmark. Given an
explicit set of assumptions it computes storage and ingestion requirements, and
labels every component with the *basis* for its number:

* ``exact``       — exact arithmetic under the stated assumptions;
* ``measured``    — genuinely derived from a throughput measured in this project;
* ``assumed``     — an illustrative default that has NOT been measured (for
  example the built-in embedding throughput or the HNSW/payload overhead
  multipliers); replacing it with a measured or documented figure is the point;
* ``documented``  — supported by an identifiable external reference;
* ``unknown``     — cannot be known without production benchmarking.

The distinction matters: a projection is only as trustworthy as its weakest
assumption, and this module never lets a hard-coded multiplier masquerade as a
measurement. Real measured values (local embedding throughput, on-disk index
sizes) are supplied by the *caller* — the harness records them, and the user
plugs them in — so the model itself stays pure and offline-testable.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Basis = Literal["exact", "measured", "assumed", "documented", "unknown"]

BYTES_PER_KB = 1024
BYTES_PER_GB = 1024**3
SECONDS_PER_DAY = 86400


class StorageAssumptions(BaseModel):
    """Inputs to the storage projection. All sizes are decimal-friendly averages.

    Numeric fields reject non-finite values (NaN/Infinity) and carry an upper
    bound so the public calculator cannot be fed an unreasonable number that
    produces a meaningless projection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents: int = Field(default=10_000_000, ge=1, le=1_000_000_000)
    chunks_per_document: float = Field(default=10.0, gt=0, le=100_000, allow_inf_nan=False)
    source_document_kb: float = Field(default=24.0, gt=0, le=10_000_000, allow_inf_nan=False)
    chunk_text_kb: float = Field(default=1.2, gt=0, le=1_000_000, allow_inf_nan=False)
    metadata_bytes_per_chunk: int = Field(default=640, ge=0, le=100_000_000)
    embedding_dimensions: int = Field(default=384, ge=1, le=16384)
    bytes_per_dimension: int = Field(default=4, ge=1, le=8)
    #: Number of distinct embedding models/indexes indexed over the same corpus.
    vector_indexes: int = Field(default=1, ge=1, le=16)
    replication_factor: int = Field(default=2, ge=1, le=10)
    #: HNSW/graph overhead multiplier *on top of* raw vectors (1.5 == +50%).
    vector_index_overhead_factor: float = Field(
        default=1.5, ge=1.0, le=5.0, allow_inf_nan=False
    )
    #: Payload-index overhead multiplier on metadata (1.2 == +20%).
    payload_index_overhead_factor: float = Field(
        default=1.2, ge=1.0, le=5.0, allow_inf_nan=False
    )
    #: Lexical index size as a fraction of source text size.
    lexical_index_ratio: float = Field(default=0.35, ge=0.0, le=5.0, allow_inf_nan=False)
    backup_copies: int = Field(default=1, ge=0, le=10)
    #: Fraction of the primary footprint reserved for a full reindex working set.
    temp_reindex_factor: float = Field(default=1.0, ge=0.0, le=5.0, allow_inf_nan=False)
    operational_headroom: float = Field(default=0.25, ge=0.0, le=5.0, allow_inf_nan=False)


class IngestionAssumptions(BaseModel):
    """Inputs to the ingestion-throughput projection.

    ``embedding_chunks_per_second_per_worker``, ``indexing_chunks_per_second`` and
    ``worker_scaling_efficiency`` are **illustrative assumptions** by default, so
    ``throughput_basis`` defaults to ``"assumed"``. Supply figures from your own
    measured run and set ``throughput_basis="measured"`` (or cite an external
    source and use ``"documented"``) before reading ``keeps_up`` as anything more
    than a conditional projection.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    documents_per_day: int = Field(default=100_000, ge=0, le=1_000_000_000_000)
    chunks_per_document: float = Field(default=10.0, gt=0, le=100_000, allow_inf_nan=False)
    embedding_chunks_per_second_per_worker: float = Field(
        default=20.0, gt=0, le=10_000_000, allow_inf_nan=False
    )
    indexing_chunks_per_second: float = Field(
        default=200.0, gt=0, le=1_000_000_000, allow_inf_nan=False
    )
    workers: int = Field(default=4, ge=1, le=1024)
    #: Fraction of ideal linear speed-up actually achieved when adding workers.
    #: 1.0 would be perfect scaling, which CPU ingestion rarely delivers.
    worker_scaling_efficiency: float = Field(
        default=0.75, ge=0.0, le=1.0, allow_inf_nan=False
    )
    available_hours_per_day: float = Field(default=24.0, gt=0, le=24, allow_inf_nan=False)
    retry_overhead: float = Field(default=0.10, ge=0.0, le=5.0, allow_inf_nan=False)
    #: ``"assumed"`` unless the caller supplies a genuinely measured throughput.
    throughput_basis: Basis = "assumed"


class StorageComponent(BaseModel):
    """One named line of the storage estimate, with its basis and a note."""

    model_config = ConfigDict(frozen=True)

    key: str
    label: str
    bytes: int
    basis: Basis
    note: str = ""


class StorageEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1"
    assumptions: StorageAssumptions
    chunks_total: float
    components: list[StorageComponent]
    primary_bytes: int
    replica_bytes: int
    backup_bytes: int
    reindex_bytes: int
    headroom_bytes: int
    total_bytes: int

    @property
    def total_gb(self) -> float:
        return self.total_bytes / BYTES_PER_GB


class IngestionEstimate(BaseModel):
    model_config = ConfigDict(frozen=True)

    schema_version: str = "1"
    assumptions: IngestionAssumptions
    chunks_per_day: float
    effective_workers: float
    embedding_capacity_per_day: float
    indexing_capacity_per_day: float
    bottleneck_per_day: float
    bottleneck_stage: str
    utilization: float
    keeps_up: bool
    backlog_per_day: float
    basis: Basis
    notes: list[str] = Field(default_factory=list)


#: Named storage scenarios required by the specification (small / typical / large
#: documents). Values are illustrative starting points a user can override.
STORAGE_SCENARIOS: dict[str, dict[str, float]] = {
    "small": {"chunks_per_document": 3.0, "source_document_kb": 8.0, "chunk_text_kb": 0.8},
    "typical": {"chunks_per_document": 10.0, "source_document_kb": 24.0, "chunk_text_kb": 1.2},
    "large": {"chunks_per_document": 48.0, "source_document_kb": 120.0, "chunk_text_kb": 1.5},
}


def storage_scenarios(base: StorageAssumptions | None = None) -> dict[str, StorageAssumptions]:
    """Return the three documented scenarios (small/typical/large) as assumptions."""

    base = base or StorageAssumptions()
    scenarios: dict[str, StorageAssumptions] = {}
    for name, overrides in STORAGE_SCENARIOS.items():
        scenarios[name] = base.model_copy(update=overrides)
    return scenarios


def estimate_storage(assumptions: StorageAssumptions) -> StorageEstimate:
    """Compute the storage estimate for one set of assumptions."""

    chunks = assumptions.documents * assumptions.chunks_per_document
    source_bytes = int(assumptions.documents * assumptions.source_document_kb * BYTES_PER_KB)
    chunk_text_bytes = int(chunks * assumptions.chunk_text_kb * BYTES_PER_KB)
    metadata_bytes = int(chunks * assumptions.metadata_bytes_per_chunk)
    raw_vector_bytes = int(
        chunks
        * assumptions.embedding_dimensions
        * assumptions.bytes_per_dimension
        * assumptions.vector_indexes
    )
    vector_index_overhead = int(
        raw_vector_bytes * (assumptions.vector_index_overhead_factor - 1.0)
    )
    payload_index_bytes = int(
        metadata_bytes * (assumptions.payload_index_overhead_factor - 1.0)
    )
    lexical_bytes = int(source_bytes * assumptions.lexical_index_ratio)

    components = [
        StorageComponent(
            key="source_documents",
            label="Original source documents",
            bytes=source_bytes,
            basis="exact",
            note="documents x average source size (object storage)",
        ),
        StorageComponent(
            key="chunk_text",
            label="Chunk text",
            bytes=chunk_text_bytes,
            basis="exact",
            note="chunks x average chunk text size",
        ),
        StorageComponent(
            key="chunk_metadata",
            label="Chunk metadata",
            bytes=metadata_bytes,
            basis="exact",
            note="chunks x metadata bytes per chunk",
        ),
        StorageComponent(
            key="raw_vectors",
            label="Raw embedding vectors",
            bytes=raw_vector_bytes,
            basis="exact",
            note="documents x chunks/doc x dimensions x bytes/dim x indexes",
        ),
        StorageComponent(
            key="vector_index_overhead",
            label="Vector index (HNSW/graph) overhead",
            bytes=vector_index_overhead,
            basis="assumed",
            note=(
                f"+{(assumptions.vector_index_overhead_factor - 1) * 100:.0f}% over raw "
                "vectors — an assumption to replace with the chosen engine's documented "
                "figure"
            ),
        ),
        StorageComponent(
            key="payload_index",
            label="Payload index overhead",
            bytes=payload_index_bytes,
            basis="assumed",
            note=(
                f"+{(assumptions.payload_index_overhead_factor - 1) * 100:.0f}% over "
                "metadata — an assumption, not a measured index size"
            ),
        ),
        StorageComponent(
            key="lexical_index",
            label="Lexical index",
            bytes=lexical_bytes,
            basis="assumed",
            note=(
                f"{assumptions.lexical_index_ratio:.0%} of source text — an assumption; "
                "the measured FTS5 prototype is a calibration point, not this ratio"
            ),
        ),
    ]
    primary = sum(component.bytes for component in components)
    replica_bytes = int(primary * (assumptions.replication_factor - 1))
    backup_bytes = int(primary * assumptions.backup_copies)
    reindex_bytes = int(primary * assumptions.temp_reindex_factor)
    subtotal = primary + replica_bytes + backup_bytes + reindex_bytes
    headroom_bytes = int(subtotal * assumptions.operational_headroom)

    return StorageEstimate(
        assumptions=assumptions,
        chunks_total=chunks,
        components=components,
        primary_bytes=primary,
        replica_bytes=replica_bytes,
        backup_bytes=backup_bytes,
        reindex_bytes=reindex_bytes,
        headroom_bytes=headroom_bytes,
        total_bytes=subtotal + headroom_bytes,
    )


def estimate_ingestion(assumptions: IngestionAssumptions) -> IngestionEstimate:
    """Compute whether a worker configuration keeps up with incoming documents."""

    chunks_per_day = assumptions.documents_per_day * assumptions.chunks_per_document
    # Non-linear scaling: only ``efficiency`` of each extra worker's ideal
    # capacity is realised. 1 worker -> 1.0 effective.
    effective_workers = 1.0 + (assumptions.workers - 1) * assumptions.worker_scaling_efficiency
    available_seconds = assumptions.available_hours_per_day * 3600
    embedding_capacity = (
        assumptions.embedding_chunks_per_second_per_worker
        * effective_workers
        * available_seconds
        / (1.0 + assumptions.retry_overhead)
    )
    indexing_capacity = assumptions.indexing_chunks_per_second * available_seconds
    bottleneck = min(embedding_capacity, indexing_capacity)
    bottleneck_stage = "embedding" if embedding_capacity <= indexing_capacity else "indexing"
    utilization = chunks_per_day / bottleneck if bottleneck > 0 else float("inf")
    backlog = max(0.0, chunks_per_day - bottleneck)

    notes = [
        "Effective workers assume non-linear CPU scaling "
        f"(efficiency {assumptions.worker_scaling_efficiency:.2f}); throughput is "
        "not assumed to grow linearly with worker count.",
        f"Throughput basis is '{assumptions.throughput_basis}'. Embedding throughput, "
        "indexing throughput and worker efficiency are illustrative assumptions "
        "unless you supplied measured values; this model does not invent or verify "
        "GPU/multi-core throughput.",
        "keeps_up is a conditional projection under these assumptions, not a "
        "demonstrated production capability.",
        "Initial bulk ingestion and continuous incremental updates should be "
        "planned as separate workloads sharing the same capacity.",
    ]
    if backlog > 0:
        notes.append(
            "Projected backlog: incoming chunks exceed processing capacity; "
            "documents would age beyond the freshness target."
        )

    return IngestionEstimate(
        assumptions=assumptions,
        chunks_per_day=chunks_per_day,
        effective_workers=effective_workers,
        embedding_capacity_per_day=embedding_capacity,
        indexing_capacity_per_day=indexing_capacity,
        bottleneck_per_day=bottleneck,
        bottleneck_stage=bottleneck_stage,
        utilization=utilization,
        keeps_up=utilization <= 1.0,
        backlog_per_day=backlog,
        basis=assumptions.throughput_basis,
        notes=notes,
    )


def human_bytes(num_bytes: float) -> str:
    """Render a byte count in binary units (B/KB/MB/GB/TB)."""

    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"  # pragma: no cover - unreachable


__all__ = [
    "BYTES_PER_GB",
    "STORAGE_SCENARIOS",
    "Basis",
    "IngestionAssumptions",
    "IngestionEstimate",
    "StorageAssumptions",
    "StorageComponent",
    "StorageEstimate",
    "estimate_ingestion",
    "estimate_storage",
    "human_bytes",
    "storage_scenarios",
]
