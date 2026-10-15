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

#: Repository root, used to locate the small committed datasets and recorded
#: benchmark artifacts that ship with the source tree.
_REPO_ROOT = Path(__file__).resolve().parents[2]

#: Bumped whenever parser output (block/section extraction) changes in a way that
#: would change chunk text. Part of an index profile's identity.
DEFAULT_PARSER_VERSION = "1.0.0"

#: Bumped when the on-disk index layout changes.
INDEX_VERSION = "1"

ChunkStrategy = Literal["fixed_token", "paragraph", "clause_aware"]
SimilarityMetric = Literal["cosine"]

#: Known embedding dimensions for models whose dimension should not be guessed.
#: Supplying a dimension explicitly always wins; this table only fills the gap for
#: the default model.
DEFAULT_EMBEDDING_DIMENSIONS: dict[str, int] = {
    "sentence-transformers/all-MiniLM-L6-v2": 384,
}

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

#: Default local cross-encoder used for optional second-stage reranking.
#: Cross-encoder scores are relevance logits, not calibrated probabilities.
DEFAULT_RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"


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

    A profile is either *lexical-only* (``embedding_model is None``) or *dense*
    (an embedding model is configured). Only dense profiles carry the extra
    embedding identity fields. :meth:`canonical` therefore adds those fields
    **conditionally**, which keeps the derived key of every Phase 1 lexical
    profile byte-for-byte identical to the Phase 1 value — existing BM25 indexes
    and their profile rows stay valid — while ensuring that any change to a
    behaviour-affecting embedding setting produces a different key.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(default="paragraph-220", min_length=1, max_length=120)
    parser_version: str = DEFAULT_PARSER_VERSION
    chunker: ChunkerConfig = Field(default_factory=ChunkerConfig)
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_revision: str | None = None
    similarity_metric: SimilarityMetric = "cosine"
    normalize_embeddings: bool = True
    embedding_document_prefix: str = ""
    embedding_query_prefix: str = ""
    index_version: str = INDEX_VERSION

    @model_validator(mode="after")
    def _validate_dense(self) -> IndexProfileSpec:
        if self.embedding_model is not None and self.embedding_dimension is None:
            raise ValueError(
                "dense profiles require embedding_dimension "
                "(use resolve_embedding_dimension to derive it for known models)"
            )
        if self.embedding_model is None and self.embedding_dimension is not None:
            raise ValueError("embedding_dimension was set without an embedding_model")
        return self

    @property
    def is_dense(self) -> bool:
        """Whether this profile stores dense vectors."""

        return self.embedding_model is not None

    def canonical(self) -> dict[str, object]:
        """Return the behaviour-affecting fields in a stable, JSON-safe form."""

        payload: dict[str, object] = {
            "parser_version": self.parser_version,
            "chunker": self.chunker.model_dump(),
            "embedding_model": self.embedding_model,
            "embedding_dimension": self.embedding_dimension,
            "index_version": self.index_version,
        }
        if self.is_dense:
            # These only affect behaviour when vectors exist, so they are added
            # only for dense profiles rather than invalidating lexical ones.
            payload["embedding_revision"] = self.embedding_revision
            payload["similarity_metric"] = self.similarity_metric
            payload["normalize_embeddings"] = self.normalize_embeddings
            payload["embedding_document_prefix"] = self.embedding_document_prefix
            payload["embedding_query_prefix"] = self.embedding_query_prefix
        return payload

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

    @property
    def vector_collection(self) -> str:
        """Deterministic vector-collection name for this profile.

        Derived purely from the profile key, so an immutable configuration maps
        to exactly one collection and incompatible configurations never share.
        """

        return f"rw_{self.key}"


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

    # Dense retrieval. These defaults are only used when a dense profile is
    # created explicitly (CLI `--embed` / API `embed=true`); nothing downloads a
    # model or opens the vector store at startup.
    embedding_model: str = DEFAULT_EMBEDDING_MODEL
    embedding_revision: str | None = None
    embedding_dimension: int | None = None
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    embedding_normalize: bool = True
    vector_dir_name: str = "qdrant"
    dense_candidate_k: int = Field(default=50, ge=1, le=1000)
    rrf_k: int = Field(default=60, ge=1, le=10000)

    # Cross-encoder reranking (optional; only loaded when a search requests it).
    reranker_model: str = DEFAULT_RERANKER_MODEL
    reranker_revision: str | None = None
    reranker_batch_size: int = Field(default=32, ge=1, le=512)
    reranker_max_length: int | None = Field(default=None, ge=8, le=8192)
    rerank_candidates: int = Field(default=20, ge=1, le=500)

    # Evaluation. The optional dense/reranker extras are only needed to *run*
    # dense or reranked benchmarks locally; inspecting recorded experiments and
    # BM25 evaluation need nothing extra.
    default_eval_k: tuple[int, ...] = Field(default=(1, 3, 5, 10))
    default_candidate_depth: int = Field(default=100, ge=1, le=500)
    dataset_dir_name: str = "datasets"
    bundled_datasets_dir: Path = _REPO_ROOT / "datasets"
    recorded_experiments_dir: Path = _REPO_ROOT / "benchmarks" / "recorded"

    # Public read-only demonstration mode. When true, every mutating endpoint is
    # refused at the backend (not merely hidden in the UI) and the shared sample
    # corpus is initialised once by a trusted internal bootstrap.
    read_only: bool = False
    autoseed_sample: bool = False

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024

    @property
    def resolved_database_url(self) -> str:
        if self.database_url:
            return self.database_url
        db_path = (self.data_dir / "rag_workbench.db").resolve()
        return f"sqlite:///{db_path.as_posix()}"

    @property
    def vector_store_path(self) -> Path:
        """Persistent Qdrant local-mode storage directory."""

        return (self.data_dir / self.vector_dir_name).resolve()

    @property
    def datasets_dir(self) -> Path:
        """Writable dataset root (imported/BEIR datasets live here)."""

        return (self.data_dir / self.dataset_dir_name).resolve()

    def dataset_roots(self) -> list[Path]:
        """Dataset roots in search order: writable first, then the bundled set."""

        return [self.datasets_dir, Path(self.bundled_datasets_dir)]

    def default_profile(self) -> IndexProfileSpec:
        """Build the configured default *lexical* index profile."""

        chunker = ChunkerConfig(
            strategy=self.default_chunk_strategy,
            max_tokens=self.default_max_tokens,
            overlap_tokens=self.default_overlap_tokens,
        )
        return IndexProfileSpec(name=f"{chunker.strategy}-{chunker.max_tokens}", chunker=chunker)

    def dense_profile(
        self,
        *,
        name: str | None = None,
        chunker: ChunkerConfig | None = None,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        embedding_revision: str | None = None,
    ) -> IndexProfileSpec:
        """Build a dense index profile from the configured defaults."""

        chunker = chunker or ChunkerConfig(
            strategy=self.default_chunk_strategy,
            max_tokens=self.default_max_tokens,
            overlap_tokens=self.default_overlap_tokens,
        )
        model = embedding_model or self.embedding_model
        dimension = resolve_embedding_dimension(
            model,
            embedding_dimension if embedding_dimension is not None else self.embedding_dimension,
        )
        return IndexProfileSpec(
            name=name or f"dense-{chunker.strategy}-{chunker.max_tokens}",
            chunker=chunker,
            embedding_model=model,
            embedding_dimension=dimension,
            embedding_revision=(
                embedding_revision if embedding_revision is not None else self.embedding_revision
            ),
            normalize_embeddings=self.embedding_normalize,
        )

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)


def resolve_embedding_dimension(model_id: str, explicit: int | None) -> int:
    """Return a concrete embedding dimension, refusing to guess unknown models."""

    if explicit is not None:
        return explicit
    if model_id in DEFAULT_EMBEDDING_DIMENSIONS:
        return DEFAULT_EMBEDDING_DIMENSIONS[model_id]
    known = ", ".join(sorted(DEFAULT_EMBEDDING_DIMENSIONS))
    raise ValueError(
        f"unknown embedding dimension for '{model_id}'; pass an explicit "
        f"embedding_dimension. Known models: {known}"
    )



@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the cached process settings.

    Tests that need different values should construct :class:`Settings` directly
    (and call ``get_settings.cache_clear()`` if they rely on the cached accessor).
    """

    return Settings()
