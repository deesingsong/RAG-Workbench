"""Runtime environment capture and latency statistics for experiment records.

Reproducibility needs more than metrics: a result is only meaningful alongside the
machine, interpreter and library versions that produced it, plus an honest summary
of the latency distribution. Everything here is best-effort and never fails a run.
"""

from __future__ import annotations

import importlib.metadata
import math
import platform
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path

_TRACKED_PACKAGES = (
    "rag-workbench",
    "torch",
    "sentence-transformers",
    "transformers",
    "qdrant-client",
    "sqlalchemy",
    "numpy",
)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def git_commit(root: Path | None = None) -> str | None:
    """Return the current git commit SHA, or ``None`` outside a repository."""

    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = result.stdout.strip()
    return sha or None


def git_worktree_dirty(root: Path | None = None) -> bool | None:
    """Return whether the working tree has uncommitted changes.

    ``True`` means there are staged or unstaged changes (so ``HEAD`` is *not* the
    exact executable source); ``False`` means the tree is clean; ``None`` means
    the state could not be determined (git is missing, or ``root`` is not a
    repository). A caller must never present ``None`` as "clean".
    """

    try:
        result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root) if root else None,
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return bool(result.stdout.strip())


def runtime_environment() -> dict[str, object]:
    """Best-effort description of the interpreter, host and key libraries."""

    packages = {name: version for name in _TRACKED_PACKAGES if (version := _package_version(name))}
    return {
        "python": sys.version.split()[0],
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": _cpu_count(),
        "packages": packages,
    }


def _cpu_count() -> int | None:
    import os

    try:
        return os.cpu_count()
    except Exception:  # noqa: BLE001 - never fail a benchmark over host introspection
        return None


def percentile(sorted_values: Sequence[float], fraction: float) -> float:
    """Nearest-rank percentile of an already-sorted, non-empty sequence."""

    if not sorted_values:
        raise ValueError("percentile of an empty sequence is undefined")
    if not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1]")
    rank = max(1, math.ceil(fraction * len(sorted_values)))
    return float(sorted_values[rank - 1])


def latency_stats(samples: Sequence[float]) -> dict[str, float | int | None]:
    """Summarise a list of millisecond timings."""

    clean = [float(value) for value in samples if value is not None]
    if not clean:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "min": None, "max": None}
    ordered = sorted(clean)
    return {
        "count": len(ordered),
        "mean": round(sum(ordered) / len(ordered), 3),
        "p50": round(percentile(ordered, 0.50), 3),
        "p95": round(percentile(ordered, 0.95), 3),
        "min": round(ordered[0], 3),
        "max": round(ordered[-1], 3),
    }


__all__ = [
    "git_commit",
    "git_worktree_dirty",
    "latency_stats",
    "percentile",
    "runtime_environment",
]
