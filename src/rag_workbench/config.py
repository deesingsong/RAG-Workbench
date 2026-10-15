"""Application configuration and index-profile identity.

Two ideas live here:

* :class:`Settings` — process-wide, environment-driven configuration.
* :class:`IndexProfileSpec` — the immutable, hashable description of a chunking
  configuration. A profile's *key* is derived only from its behaviour-affecting
  fields, so two differently named profiles with identical settings share one
  index, and two profiles with different chunking never contaminate each other.
"""

from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

#: Bumped whenever parser output (block/section extraction) changes in a way that
#: would change chunk text. Part of an index profile's identity.
DEFAULT_PARSER_VERSION = "1.0.0"

#: Bumped when the on-disk index layout changes.
INDEX_VERSION = "1"

ChunkStrategy = Literal["fixed_token", "paragraph"]


class ChunkerConfig(BaseModel):
    """Behaviour-affecting chunker settings.

    ``max_tokens`` and ``overlap_tokens`` count *tokens*, not characters. The
    token unit is a deterministic regex word tokenizer (Unicode word characters
    plus individual punctuation marks); see :mod:`rag_workbench.ingestion.chunkers`.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    strategy: ChunkStrategy = "paragraph"
    max_tokens: int = Field(default=220, ge=1, le=4096)
    overlap_tokens: int = Field(default=40, ge=0, le=4096)

    @model_validator(mode="after")
    def _validate_overlap(self) -> ChunkerConfig:
        if self.overlap_tokens >= self.max_tokens:
            raise ValueError("overlap_tokens must be smaller than max_tokens")
        return self


class IndexProfileSpec(BaseModel):
    """The full identity of an index profile.

    ``name`` is a human label and is deliberately excluded from the derived key,
    so that renaming a profile does not silently create a new index.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="paragraph-220", min_length=1, max_length=120)
    parser_version: str = DEFAULT_PARSER_VERSION
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    index_version: str = INDEX_VERSION

    def canonical(self) -> dict[str, object]:
        """Return the behaviour-affecting fields in a stable, JSON-safe form."""

        return {
            "parser_version": self.parser_version,
            "chunker": self.chunker.model_dump(),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "index_version": self.index_version,
        }

    @property
    def key(self) -> str:
        """Deterministic, short profile identifier derived from :meth:`canonical`."""

        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()
        return f"p-{digest[:16]}"

    @property
    def config_hash(self) -> str:
        """Full SHA-256 of the canonical configuration, for experiment records."""

        payload = json.dumps(self.canonical(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class Settings(BaseSettings):
    """Environment-driven process settings (prefix ``RAG_WORKBENCH_``)."""

    model_config = SettingsConfigDict(
        env_prefix="RAG_WORKBENCH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "RAG Workbench"
    data_dir: Path = Path("data")
    database_url: str | None = None
    max_upload_mb: int = Field(default=25, ge=1, le=2048)
    allowed_extensions: tuple[str, ...] = (".txt", ".md", ".markdown", ".pdf")
    log_level: str = "INFO"
    default_collection: str = "default"
    default_chunk_strategy: ChunkStrategy = "paragraph"
    default_max_tokens: int = Field(default=220, ge=1, le=4096)
    default_overlap_tokens: int = Field(default=40, ge=0, le=4096)

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = (self.data_dir / "rag_workbench.db").resolve()
        return f"sqlite:///{db_path.as_posix()}"

    def default_profile(self) -> IndexProfileSpec:
        """Build the configured default index profile."""

        chunker = ChunkerConfig(
            strategy=self.default_chunk_strategy,
            max_tokens=self.default_max_tokens,
            overlap_tokens=self.default_overlap_tokens,
        )
        return IndexProfileSpec(name=f"{chunker.strategy}-{chunker.max_tokens}", chunker=chunker)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings.

    Tests that need different values should construct :class:`Settings` directly
    (and call ``get_settings.cache_clear()`` if they rely on the cached accessor).
    """

    return Settings()
