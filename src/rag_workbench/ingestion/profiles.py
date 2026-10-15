"""Build an :class:`IndexProfileSpec` from request/CLI options.

Both the document-ingestion endpoints and the evaluation commands need to turn a
loose set of strategy parameters into a validated, hashable profile. Keeping the
one implementation here means an evaluation profile is always exactly the profile
the API would have created for the same options.
"""

from __future__ import annotations

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.domain.errors import ValidationError


def build_profile(
    settings: Settings,
    *,
    strategy: str | None = None,
    max_tokens: int | None = None,
    overlap_tokens: int | None = None,
    name: str | None = None,
    embed: bool = False,
    embedding_model: str | None = None,
    embedding_dimension: int | None = None,
) -> IndexProfileSpec:
    """Return a validated profile, raising :class:`ValidationError` on bad input."""

    try:
        chunker = ChunkerConfig(
            strategy=strategy or settings.default_chunk_strategy,  # type: ignore[arg-type]
            max_tokens=max_tokens or settings.default_max_tokens,
            overlap_tokens=(
                overlap_tokens if overlap_tokens is not None else settings.default_overlap_tokens
            ),
        )
    except ValueError as exc:
        raise ValidationError(f"invalid chunking configuration: {exc}") from exc

    if embed:
        try:
            return settings.dense_profile(
                name=name or f"dense-{chunker.strategy}-{chunker.max_tokens}",
                chunker=chunker,
                embedding_model=embedding_model,
                embedding_dimension=embedding_dimension,
            )
        except ValueError as exc:
            raise ValidationError(f"invalid dense configuration: {exc}") from exc

    return IndexProfileSpec(
        name=name or f"{chunker.strategy}-{chunker.max_tokens}", chunker=chunker
    )


__all__ = ["build_profile"]
