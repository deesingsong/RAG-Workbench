"""Trusted internal bootstrap for the bundled demonstration corpus.

The public cloud demo runs read-only, so an anonymous visitor cannot ingest
anything. The shared sample contract is instead seeded once by the application
itself on startup, through the ordinary ingestion service, before the read-only
guard applies. This keeps the demo populated without opening a mutation path to
the public.
"""

from __future__ import annotations

from pathlib import Path

from rag_workbench.config import ChunkerConfig, IndexProfileSpec
from rag_workbench.ingestion.pipeline import IngestionService

_REPO_ROOT = Path(__file__).resolve().parents[3]
BUNDLED_SAMPLE = _REPO_ROOT / "tests" / "fixtures" / "sample_contract.txt"

#: Demonstration profiles the sample corpus is seeded under.
DEMO_STRATEGIES: tuple[str, ...] = ("paragraph", "clause_aware")


def profile_for_strategy(
    strategy: str, *, max_tokens: int = 220, overlap: int = 40
) -> IndexProfileSpec:
    chunker = ChunkerConfig(
        strategy=strategy, max_tokens=max_tokens, overlap_tokens=overlap
    )  # type: ignore[arg-type]
    return IndexProfileSpec(name=f"{strategy}-{max_tokens}", chunker=chunker)


def seed_bundled_sample(
    ingestion: IngestionService,
    *,
    collection: str = "contracts",
    strategies: tuple[str, ...] = DEMO_STRATEGIES,
    sample_path: Path | None = None,
) -> list[str]:
    """Seed whichever demonstration profiles are missing; return their names.

    Runs against the ingestion service directly (never over HTTP), so it is not
    subject to the public read-only guard. Idempotent: an already-present profile
    is skipped.
    """

    path = sample_path or BUNDLED_SAMPLE
    if not path.is_file():
        return []
    existing = {record.chunker_strategy for record in ingestion.list_profiles()}
    seeded: list[str] = []
    for strategy in strategies:
        if strategy in existing:
            continue
        profile = profile_for_strategy(strategy)
        result = ingestion.ingest_bytes(
            path.read_bytes(),
            source_name=path.name,
            collection=collection,
            profile=profile,
        )
        seeded.append(f"{strategy} ({result.chunk_count} chunks)")
    return seeded


__all__ = ["BUNDLED_SAMPLE", "DEMO_STRATEGIES", "profile_for_strategy", "seed_bundled_sample"]
