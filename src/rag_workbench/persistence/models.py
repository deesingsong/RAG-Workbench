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
