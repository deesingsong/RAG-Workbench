"""Deployment revision identity.

A **package version** (``__version__``) is a human-facing release string that can
stay constant across many commits, so it is a poor way to tell whether a running
process is serving the code that is currently on disk. The **deployment
revision** is a concrete identity of a build of the running code. This module
resolves it, in order of preference:

1. Git through the ``git`` executable (``git rev-parse``);
2. the repository's ``.git`` metadata read directly, without a subprocess (a
   slim image may not ship ``git``);
3. a deterministic content hash of the package source (no repository at all).

The result is a short string plus the method that produced it, so a caller can
report *how* the revision was determined rather than presenting a hash or SHA as
more authoritative than it is.
"""

from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

_GIT_TIMEOUT_SECONDS = 3.0
_SHORT_LENGTH = 12


def deployment_revision(root: Path) -> tuple[str | None, str]:
    """Return ``(revision, source)`` for the code rooted at ``root``.

    ``source`` is one of ``git``, ``git-file``, ``source-hash`` or
    ``unavailable``. ``revision`` is ``None`` only when nothing could be read.
    """

    revision = _git_subprocess(root)
    if revision:
        return revision, "git"
    revision = _git_metadata(root)
    if revision:
        return revision, "git-file"
    revision = _source_hash(root / "src" / "rag_workbench")
    if revision:
        return revision, "source-hash"
    return None, "unavailable"


def _git_subprocess(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--short=12", "HEAD"],
            capture_output=True,
            text=True,
            timeout=_GIT_TIMEOUT_SECONDS,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip() or None


def _git_metadata(root: Path) -> str | None:
    """Read the commit from ``.git`` files directly (no ``git`` executable)."""

    head = root / ".git" / "HEAD"
    try:
        content = head.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not content:
        return None
    if content.startswith("ref:"):
        return _resolve_ref(root, content[4:].strip())
    # Detached HEAD: the file holds the SHA directly.
    return content[:_SHORT_LENGTH]


def _resolve_ref(root: Path, ref: str) -> str | None:
    ref_file = root / ".git" / ref
    try:
        value = ref_file.read_text(encoding="utf-8").strip()
        if value:
            return value[:_SHORT_LENGTH]
    except OSError:
        pass
    packed = root / ".git" / "packed-refs"
    try:
        for line in packed.read_text(encoding="utf-8").splitlines():
            if line and not line.startswith("#") and line.rsplit(" ", 1)[-1] == ref:
                return line.split(" ", 1)[0][:_SHORT_LENGTH]
    except OSError:
        pass
    return None


def _source_hash(directory: Path) -> str | None:
    """A stable fingerprint of the package source, used without a repository."""

    if not directory.is_dir():
        return None
    digest = hashlib.sha256()
    for path in sorted(directory.rglob("*.py")):
        digest.update(path.relative_to(directory).as_posix().encode("utf-8"))
        try:
            digest.update(path.read_bytes())
        except OSError:
            return None
    return digest.hexdigest()[:_SHORT_LENGTH]


__all__ = ["deployment_revision"]
