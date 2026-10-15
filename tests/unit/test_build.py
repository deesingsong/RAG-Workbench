"""Unit tests for deployment-revision resolution."""

from __future__ import annotations

from pathlib import Path

from rag_workbench import build


def _git_dir(tmp_path: Path) -> Path:
    git = tmp_path / ".git"
    git.mkdir()
    return git


def test_git_metadata_reads_ref(tmp_path: Path) -> None:
    git = _git_dir(tmp_path)
    (git / "refs" / "heads").mkdir(parents=True)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "refs" / "heads" / "main").write_text("a" * 40 + "\n", encoding="utf-8")
    assert build._git_metadata(tmp_path) == "a" * 12


def test_git_metadata_reads_packed_refs(tmp_path: Path) -> None:
    git = _git_dir(tmp_path)
    (git / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
    (git / "packed-refs").write_text("# pack-refs\n" + "b" * 40 + " refs/heads/main\n")
    assert build._git_metadata(tmp_path) == "b" * 12


def test_git_metadata_reads_detached_head(tmp_path: Path) -> None:
    git = _git_dir(tmp_path)
    (git / "HEAD").write_text("c" * 40 + "\n", encoding="utf-8")
    assert build._git_metadata(tmp_path) == "c" * 12


def test_git_metadata_missing_returns_none(tmp_path: Path) -> None:
    assert build._git_metadata(tmp_path) is None


def test_deployment_revision_prefers_git(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(build, "_git_subprocess", lambda root: "deadbeef1234")
    revision, source = build.deployment_revision(tmp_path)
    assert (revision, source) == ("deadbeef1234", "git")


def test_deployment_revision_uses_git_files_without_executable(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(build, "_git_subprocess", lambda root: None)
    git = _git_dir(tmp_path)
    (git / "HEAD").write_text("d" * 40 + "\n", encoding="utf-8")
    revision, source = build.deployment_revision(tmp_path)
    assert (revision, source) == ("d" * 12, "git-file")


def test_deployment_revision_falls_back_to_source_hash(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(build, "_git_subprocess", lambda root: None)
    package = tmp_path / "src" / "rag_workbench"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("x = 1\n", encoding="utf-8")
    revision, source = build.deployment_revision(tmp_path)
    assert source == "source-hash"
    assert revision is not None and len(revision) == 12


def test_source_hash_is_deterministic_and_content_sensitive(tmp_path: Path) -> None:
    package = tmp_path / "pkg"
    package.mkdir()
    module = package / "mod.py"
    module.write_text("a = 1\n", encoding="utf-8")
    first = build._source_hash(package)
    assert first == build._source_hash(package)
    module.write_text("a = 2\n", encoding="utf-8")
    assert build._source_hash(package) != first


def test_deployment_revision_unavailable_without_git_or_source(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(build, "_git_subprocess", lambda root: None)
    revision, source = build.deployment_revision(tmp_path)
    assert (revision, source) == (None, "unavailable")
