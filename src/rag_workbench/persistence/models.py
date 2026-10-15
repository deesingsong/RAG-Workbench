"""SQLAlchemy 2.0 ORM models for local metadata and sparse retrieval.

The relational tables hold documents, chunks and ingestion jobs. Keyword search
is served by an SQLite FTS5 virtual table that is kept synchronized with the
``chunks`` table by database triggers (see :mod:`rag_workbench.persistence.database`).
"""

from __future__ import annotations

from datetime import UTC, datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


def utcnow() -> datetime:
    """Timezone-aware UTC timestamp."""

    return datetime.now(tz=UTC)


class Base(DeclarativeBase):
    """Declarative base for all tables."""


class IndexProfile(Base):
    """A versioned, config-hashed chunking/index configuration.

    ``id`` is the profile key from :class:`~rag_workbench.config.IndexProfileSpec`,
    derived purely from behaviour-affecting settings so identical configurations
    share one index and different ones never mix.
    """

    __tablename__ = "index_profiles"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False)
    config_hash: Mapped[str] = mapped_column(String(64), nullable=False)
    parser_version: Mapped[str] = mapped_column(String(32), nullable=False)
    chunker_strategy: Mapped[str] = mapped_column(String(32), nullable=False)
    chunker_params: Mapped[dict] = mapped_column(JSON, nullable=False)
    embedding_model: Mapped[str | None] = mapped_column(String(160), nullable=True)
    embedding_dimension: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Revision, metric, normalization and preprocessing prefixes. Kept as JSON so
    # adding a dense identity field does not require a new column each time; the
    # profile *key* (not this blob) is what guarantees index isolation.
    embedding_params: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    index_version: Mapped[str] = mapped_column(String(32), nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    chunks: Mapped[list[Chunk]] = relationship(back_populates="profile")


class Document(Base):
    """A source document identified by its collection and content checksum."""

    __tablename__ = "documents"
    __table_args__ = (
        UniqueConstraint("collection", "source_checksum", name="uq_document_collection_checksum"),
    )

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    collection: Mapped[str] = mapped_column(String(120), nullable=False, index=True)
    source_name: Mapped[str] = mapped_column(String(400), nullable=False)
    source_checksum: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    # Original identifier of the source when it comes from an imported benchmark
    # dataset (e.g. a BEIR document ID). ``None`` for ordinary uploads; used by
    # document-level evaluation to map retrieved chunks back to benchmark IDs.
    external_id: Mapped[str | None] = mapped_column(String(256), nullable=True)
    mime_type: Mapped[str] = mapped_column(String(120), nullable=False)
    byte_size: Mapped[int] = mapped_column(Integer, nullable=False)
    imported_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="indexed")
    diagnostics: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    chunks: Mapped[list[Chunk]] = relationship(back_populates="document")


class Chunk(Base):
    """A persisted chunk belonging to one document under one index profile."""

    __tablename__ = "chunks"
    __table_args__ = (
        UniqueConstraint("document_id", "profile_id", "chunk_order", name="uq_chunk_identity"),
        Index("ix_chunks_profile", "profile_id"),
        Index("ix_chunks_document", "document_id"),
    )

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    chunk_id: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    document_id: Mapped[str] = mapped_column(
        ForeignKey("documents.id", ondelete="CASCADE"), nullable=False
    )
    profile_id: Mapped[str] = mapped_column(
        ForeignKey("index_profiles.id", ondelete="CASCADE"), nullable=False
    )
    chunk_order: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    token_count: Mapped[int] = mapped_column(Integer, nullable=False)
    section_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    # Clause-aware provenance (Phase 3). Nullable so Phase 1/2 databases migrate
    # additively; ``None`` means the value could not be derived deterministically.
    section_number: Mapped[str | None] = mapped_column(String(64), nullable=True)
    section_title: Mapped[str | None] = mapped_column(Text, nullable=True)
    parent_clause: Mapped[str | None] = mapped_column(Text, nullable=True)
    section_level: Mapped[int | None] = mapped_column(Integer, nullable=True)
    diagnostics: Mapped[list | None] = mapped_column(JSON, nullable=True)

    document: Mapped[Document] = relationship(back_populates="chunks")
    profile: Mapped[IndexProfile] = relationship(back_populates="chunks")


class IngestionJob(Base):
    """Lifecycle record for one (document, profile) ingestion attempt."""

    __tablename__ = "ingestion_jobs"

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    document_id: Mapped[str | None] = mapped_column(
        ForeignKey("documents.id", ondelete="SET NULL"), nullable=True, index=True
    )
    profile_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    collection: Mapped[str] = mapped_column(String(120), nullable=False)
    state: Mapped[str] = mapped_column(String(16), nullable=False)
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    chunk_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    duration_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    error_code: Mapped[str | None] = mapped_column(String(48), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    diagnostics: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    timings_ms: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)


class Experiment(Base):
    """A recorded evaluation experiment.

    Only compact, structured metadata lives here; the per-query rows live in
    :class:`ExperimentQuery`, and large retrieved-text objects are never
    duplicated (the trace keeps identifiers, scores and a short excerpt).
    """

    __tablename__ = "experiments"
    __table_args__ = (Index("ix_experiments_dataset", "dataset_name"),)

    id: Mapped[str] = mapped_column(String(64), primary_key=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    dataset_name: Mapped[str] = mapped_column(String(160), nullable=False)
    dataset_version: Mapped[str] = mapped_column(String(32), nullable=False, default="1")
    dataset_checksum: Mapped[str | None] = mapped_column(String(64), nullable=True)
    relevance_mapping: Mapped[str] = mapped_column(String(24), nullable=False)
    mode: Mapped[str] = mapped_column(String(24), nullable=False)
    profile_id: Mapped[str] = mapped_column(String(64), nullable=False)
    collection: Mapped[str] = mapped_column(String(120), nullable=False)
    rerank: Mapped[bool] = mapped_column(nullable=False, default=False)
    candidate_depth: Mapped[int] = mapped_column(Integer, nullable=False)
    top_k: Mapped[int] = mapped_column(Integer, nullable=False)
    label: Mapped[str | None] = mapped_column(String(200), nullable=True)
    subset: Mapped[bool] = mapped_column(nullable=False, default=False)
    git_commit: Mapped[str | None] = mapped_column(String(64), nullable=True)
    config: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    dataset: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    runtime: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    aggregate: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    candidate_metrics: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    counts: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    latency: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    notes: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    queries: Mapped[list[ExperimentQuery]] = relationship(
        back_populates="experiment", cascade="all, delete-orphan"
    )


class ExperimentQuery(Base):
    """Per-query metrics, diagnostics and a compact retrieval trace."""

    __tablename__ = "experiment_queries"
    __table_args__ = (Index("ix_experiment_queries_experiment", "experiment_id"),)

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.id", ondelete="CASCADE"), nullable=False
    )
    query_id: Mapped[str] = mapped_column(String(160), nullable=False)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    num_relevant: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ok")
    candidate_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    unique_documents: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    sufficient_candidates: Mapped[bool] = mapped_column(nullable=False, default=True)
    metrics: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    candidate_recall: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    timings_ms: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    diagnostics: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    trace: Mapped[list] = mapped_column(JSON, nullable=False, default=list)

    experiment: Mapped[Experiment] = relationship(back_populates="queries")
