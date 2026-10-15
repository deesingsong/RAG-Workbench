"""Local embedding providers.

The default provider wraps ``sentence-transformers`` and loads the model
**lazily** on the first embedding call, so nothing is downloaded at application
startup and BM25-only workflows never touch a model. ``sentence-transformers``
is an optional dependency; when it is missing the provider raises
:class:`~rag_workbench.domain.errors.DenseUnavailableError` with install
instructions instead of breaking unrelated functionality.

Model identity is explicit: ``model_id``, ``revision``, ``dimension``,
``normalize`` and the tokenizer ``max_tokens`` limit are all exposed so they can
be recorded in an index profile. Inputs longer than the model's supported
sequence length are counted and reported rather than silently truncated.
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Sequence

from rag_workbench.config import IndexProfileSpec, Settings
from rag_workbench.domain.errors import (
    DenseUnavailableError,
    DimensionMismatchError,
    ProfileNotDenseError,
)
from rag_workbench.domain.schemas import EmbeddingBatch

logger = logging.getLogger(__name__)

DENSE_INSTALL_HINT = "Install dense retrieval extras, then retry: uv sync --extra dense"

# One loaded model per (model_id, revision) per process; loading is expensive and
# the model is read-only. Guarded because the UI/API could call concurrently.
_MODEL_CACHE: dict[tuple[str, str | None], object] = {}
_MODEL_LOCK = threading.Lock()


def _embedding_dimension(model) -> int:  # noqa: ANN001
    """Read a model's output dimension across sentence-transformers versions."""

    getter = getattr(model, "get_embedding_dimension", None) or getattr(
        model, "get_sentence_embedding_dimension", None
    )
    if getter is None:
        raise DenseUnavailableError(
            "embedding model does not expose an embedding dimension"
        )
    return int(getter())


def _load_sentence_transformer(model_id: str, revision: str | None):  # noqa: ANN202
    """Import sentence-transformers lazily and load the model on CPU."""

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # optional dependency
        raise DenseUnavailableError(
            "sentence-transformers is not installed. " + DENSE_INSTALL_HINT
        ) from exc

    key = (model_id, revision)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            model = SentenceTransformer(model_id, revision=revision, device="cpu")
        except Exception as exc:  # noqa: BLE001 - network, HF, disk, config errors
            raise DenseUnavailableError(
                f"could not load embedding model '{model_id}' (revision={revision!r}): {exc}. "
                "The model must be downloadable once or already present in the local "
                "Hugging Face cache."
            ) from exc
        _MODEL_CACHE[key] = model
        logger.info(
            "embedding model loaded",
            extra={
                "model_id": model_id,
                "revision": revision,
                "dimension": _embedding_dimension(model),
                "max_seq_length": getattr(model, "max_seq_length", None),
            },
        )
        return model


def clear_model_cache() -> None:
    """Drop cached models (used by tests)."""

    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


class SentenceTransformerEmbeddingProvider:
    """CPU-first embedding provider backed by Sentence Transformers."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        normalize: bool = True,
        batch_size: int = 32,
        document_prefix: str = "",
        query_prefix: str = "",
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._normalize = normalize
        self._batch_size = batch_size
        self._document_prefix = document_prefix
        self._query_prefix = query_prefix
        self._model = None  # lazy

    # ---------------------------------------------------------------- identity

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def normalize(self) -> bool:
        return self._normalize

    def _ensure_model(self):  # noqa: ANN202
        if self._model is None:
            self._model = _load_sentence_transformer(self._model_id, self._revision)
        return self._model

    @property
    def dimension(self) -> int:
        return _embedding_dimension(self._ensure_model())

    @property
    def max_tokens(self) -> int | None:
        value = getattr(self._ensure_model(), "max_seq_length", None)
        return int(value) if value is not None else None

    # ---------------------------------------------------------------- encoding

    def embed_documents(self, texts: Sequence[str]) -> EmbeddingBatch:
        model = self._ensure_model()
        prepared = [f"{self._document_prefix}{text}" for text in texts]
        over_length, max_seen = self._length_stats(model, prepared)

        if not prepared:
            return EmbeddingBatch(
                vectors=[], dimension=self.dimension, diagnostics=[], max_tokens_seen=0
            )

        vectors = model.encode(
            list(prepared),
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        as_lists = [vector.tolist() for vector in vectors]

        diagnostics: list[str] = []
        if over_length:
            limit = self.max_tokens
            diagnostics.append(
                f"{over_length} input(s) exceeded the model sequence length ({limit} tokens "
                "including special tokens) and were truncated by the model"
            )
        dimension = len(as_lists[0]) if as_lists else self.dimension
        return EmbeddingBatch(
            vectors=as_lists,
            dimension=dimension,
            diagnostics=diagnostics,
            truncated_count=over_length,
            max_tokens_seen=max_seen,
        )

    def embed_query(self, text: str) -> list[float]:
        model = self._ensure_model()
        vector = model.encode(
            f"{self._query_prefix}{text}",
            batch_size=1,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=self._normalize,
        )
        return vector.tolist()

    # ---------------------------------------------------------------- helpers

    def _length_stats(self, model, texts: list[str]) -> tuple[int, int]:  # noqa: ANN001
        """Count inputs over the model's sequence limit without hijacking encode."""

        tokenizer = getattr(model, "tokenizer", None)
        max_length = getattr(model, "max_seq_length", None)
        if tokenizer is None or max_length is None:
            return 0, 0

        over = 0
        max_seen = 0
        for text in texts:
            try:
                tokens = tokenizer(text, truncation=False, add_special_tokens=True)["input_ids"]
            except Exception:  # noqa: BLE001 - tokenizer variation; skip diagnostics
                continue
            length = len(tokens)
            max_seen = max(max_seen, length)
            if length > max_length:
                over += 1
        return over, max_seen


def build_embedding_provider(
    spec: IndexProfileSpec, settings: Settings | None = None
) -> SentenceTransformerEmbeddingProvider:
    """Build the default provider for a dense index profile."""

    if not spec.is_dense:
        raise ProfileNotDenseError(
            f"profile '{spec.key}' has no embedding_model; dense retrieval requires a dense profile"
        )
    assert spec.embedding_model is not None  # narrowed by is_dense
    assert spec.embedding_dimension is not None
    batch_size = settings.embedding_batch_size if settings is not None else 32
    return SentenceTransformerEmbeddingProvider(
        spec.embedding_model,
        revision=spec.embedding_revision,
        normalize=spec.normalize_embeddings,
        batch_size=batch_size,
        document_prefix=spec.embedding_document_prefix,
        query_prefix=spec.embedding_query_prefix,
    )


def validate_dimension(expected: int, observed: int, *, context: str = "embedding output") -> None:
    """Raise a clear error when a vector dimension does not match the profile."""

    if expected != observed:
        raise DimensionMismatchError(
            f"{context} has dimension {observed} but the index profile expects {expected}"
        )


__all__ = [
    "DENSE_INSTALL_HINT",
    "SentenceTransformerEmbeddingProvider",
    "build_embedding_provider",
    "clear_model_cache",
    "validate_dimension",
]
