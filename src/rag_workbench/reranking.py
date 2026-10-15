"""Local cross-encoder reranking.

A cross-encoder scores a *query-passage pair* jointly, which is far more
expensive than the dual-encoder similarity used for dense retrieval but also much
more accurate at ordering a small candidate set. :class:`CrossEncoderReranker`
wraps Sentence Transformers' ``CrossEncoder`` and loads the model **lazily** on
the first scoring call, so BM25-only workflows never touch a model.

Important semantics:

* Scores are raw relevance logits: **larger is more relevant**, but they are
  *not* calibrated probabilities of relevance and are never described as such.
* Model initialization (including a possible one-off download) is measured
  separately from steady-state inference: :meth:`CrossEncoderReranker.ensure_loaded`
  returns the one-off initialization time, and ``score`` performs inference only.
* The reranker only ever *reorders* passages supplied by the retrieval layer; it
  never introduces new candidates.
"""

from __future__ import annotations

import logging
import threading
import time
from collections.abc import Sequence

from rag_workbench.config import DEFAULT_RERANKER_MODEL, Settings
from rag_workbench.domain.errors import RerankerUnavailableError
from rag_workbench.domain.schemas import RerankBatch

logger = logging.getLogger(__name__)

RERANKER_INSTALL_HINT = "Install dense/reranking extras, then retry: uv sync --extra dense"

# One loaded cross-encoder per (model_id, revision) per process. Loading is
# expensive and the model is read-only, so it is shared and never mutated.
_MODEL_CACHE: dict[tuple[str, str | None], object] = {}
_MODEL_LOCK = threading.Lock()


def _load_cross_encoder(model_id: str, revision: str | None):  # noqa: ANN202
    """Import sentence-transformers lazily and load the cross-encoder on CPU."""

    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # optional dependency
        raise RerankerUnavailableError(
            "sentence-transformers is not installed. " + RERANKER_INSTALL_HINT
        ) from exc

    key = (model_id, revision)
    with _MODEL_LOCK:
        cached = _MODEL_CACHE.get(key)
        if cached is not None:
            return cached
        try:
            model = CrossEncoder(model_id, revision=revision, device="cpu")
        except Exception as exc:  # noqa: BLE001 - network, HF, disk, config errors
            raise RerankerUnavailableError(
                f"could not load reranker model '{model_id}' (revision={revision!r}): {exc}. "
                "The model must be downloadable once or already present in the local "
                "Hugging Face cache."
            ) from exc
        _MODEL_CACHE[key] = model
        logger.info(
            "reranker model loaded",
            extra={
                "model_id": model_id,
                "revision": revision,
                "max_length": getattr(model, "max_length", None),
            },
        )
        return model


def clear_reranker_cache() -> None:
    """Drop cached reranker models (used by tests)."""

    with _MODEL_LOCK:
        _MODEL_CACHE.clear()


class CrossEncoderReranker:
    """CPU-first cross-encoder reranker backed by Sentence Transformers."""

    def __init__(
        self,
        model_id: str,
        *,
        revision: str | None = None,
        batch_size: int = 32,
        max_length: int | None = None,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._batch_size = batch_size
        self._max_length = max_length
        self._model = None  # lazy

    # ---------------------------------------------------------------- identity

    @property
    def model_id(self) -> str:
        return self._model_id

    @property
    def revision(self) -> str | None:
        return self._revision

    @property
    def max_length(self) -> int | None:
        if self._max_length is not None:
            return self._max_length
        if self._model is not None:
            value = getattr(self._model, "max_length", None)
            return int(value) if value is not None else None
        return None

    @property
    def batch_size(self) -> int:
        return self._batch_size

    @property
    def is_loaded(self) -> bool:
        return self._model is not None or (
            self._model_id,
            self._revision,
        ) in _MODEL_CACHE

    def ensure_loaded(self) -> float:
        """Load the model if needed; return one-off initialization time in ms."""

        if self._model is None:
            started = time.perf_counter()
            self._model = _load_cross_encoder(self._model_id, self._revision)
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return round(elapsed_ms, 3)
        return 0.0

    # ---------------------------------------------------------------- scoring

    def score(self, query: str, passages: Sequence[str]) -> RerankBatch:
        """Score query-passage pairs; scores are relevance logits (higher is better)."""

        prepared = list(passages)
        if not prepared:
            return RerankBatch(scores=[], max_length=self.max_length, diagnostics=[])

        model = self._ensure_model()
        pairs = [(query, passage) for passage in prepared]
        raw = model.predict(
            pairs,
            batch_size=self._batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        scores = [float(value) for value in _as_flat_list(raw)]

        over_length, max_seen = _pair_length_stats(model, pairs, self.max_length)
        diagnostics: list[str] = []
        if over_length:
            diagnostics.append(
                f"{over_length} query-passage pair(s) exceeded the model max length "
                f"({self.max_length} tokens) and were truncated by the model"
            )
        return RerankBatch(
            scores=scores,
            max_length=self.max_length,
            truncated_count=over_length,
            diagnostics=diagnostics,
        )

    # ---------------------------------------------------------------- helpers

    def _ensure_model(self):  # noqa: ANN202
        if self._model is None:
            self._model = _load_cross_encoder(self._model_id, self._revision)
        return self._model


def _as_flat_list(raw) -> list[float]:  # noqa: ANN001
    """Normalize ``CrossEncoder.predict`` output (numpy array or list) to floats."""

    try:
        return [float(value) for value in raw.reshape(-1).tolist()]  # type: ignore[attr-defined]
    except AttributeError:
        flattened: list[float] = []
        for item in raw:
            if isinstance(item, (list, tuple)):
                flattened.extend(float(value) for value in item)
            else:
                flattened.append(float(item))
        return flattened


def _pair_length_stats(
    model, pairs: Sequence[tuple[str, str]], max_length: int | None  # noqa: ANN001
) -> tuple[int, int]:
    """Count pairs longer than the model's max length without hijacking predict."""

    tokenizer = getattr(model, "tokenizer", None)
    if tokenizer is None or max_length is None:
        return 0, 0

    over = 0
    max_seen = 0
    for query, passage in pairs:
        try:
            encoded = tokenizer(query, passage, truncation=False)["input_ids"]
        except Exception:  # noqa: BLE001 - tokenizer variation; skip diagnostics
            continue
        length = len(encoded)
        max_seen = max(max_seen, length)
        if length > max_length:
            over += 1
    return over, max_seen


def build_reranker(model_id: str | None = None, settings: Settings | None = None):
    """Build the configured cross-encoder reranker (default: ms-marco MiniLM)."""

    resolved = model_id or (
        settings.reranker_model if settings is not None else DEFAULT_RERANKER_MODEL
    )
    return CrossEncoderReranker(
        resolved,
        revision=settings.reranker_revision if settings is not None else None,
        batch_size=settings.reranker_batch_size if settings is not None else 32,
        max_length=settings.reranker_max_length if settings is not None else None,
    )


__all__ = [
    "RERANKER_INSTALL_HINT",
    "CrossEncoderReranker",
    "build_reranker",
    "clear_reranker_cache",
]
