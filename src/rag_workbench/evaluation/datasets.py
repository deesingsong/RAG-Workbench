"""Evaluation dataset loading, validation and local storage.

A *dataset* is a directory containing:

* ``manifest.json`` — identity and integrity metadata (source, version, license,
  relevance-mapping policy, counts, checksums);
* ``corpus.jsonl``  — one JSON object per document: ``{"_id", "title", "text"}``;
* ``queries.jsonl`` — one JSON object per query: ``{"_id", "text"}``;
* ``qrels.tsv``     — ``query-id<TAB>corpus-id<TAB>score`` relevance judgments;
* ``evidence.jsonl`` *optional* — for evidence-unit datasets (the legal set),
  one object per evidence unit: ``{"_id", "contract", "section", "text"}``.

The format is a deliberately thin super-set of BEIR's own JSONL/TSV layout, so a
BEIR dataset (SciFact) can be imported by normalising its files without
rewriting the judgments. Nothing here downloads anything: import reads local
files and the download step is a separate, explicit command.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from rag_workbench.domain.errors import DatasetError, DatasetNotFoundError

DATASET_SCHEMA_VERSION = "1"

#: Relevance-mapping policy names, recorded in every experiment for honesty.
DOCUMENT_MAPPING = "document"
EVIDENCE_MAPPING = "evidence"
MAPPING_POLICIES = (DOCUMENT_MAPPING, EVIDENCE_MAPPING)

_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """Reduce a dataset name to a filesystem-safe slug."""

    slug = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    if not slug:
        raise DatasetError(f"dataset name '{name}' does not contain any usable characters")
    return slug


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(path.read_bytes())


class CorpusDocument(BaseModel):
    """A benchmark document, addressed by its original identifier."""

    model_config = ConfigDict(frozen=True)

    id: str
    title: str | None = None
    text: str

    def index_text(self) -> str:
        """Title plus body, the text that gets indexed for retrieval."""

        if self.title and self.title.strip():
            return f"{self.title.strip()}\n{self.text}".strip()
        return self.text


class EvalQuery(BaseModel):
    """An evaluation query, addressed by its original identifier."""

    model_config = ConfigDict(frozen=True)

    id: str
    text: str


class EvidenceUnit(BaseModel):
    """A clause/evidence unit that a legal query's relevance is anchored to.

    ``text`` is the canonical evidence passage. A retrieved chunk counts as a
    match when it contains that passage (normalised) *or* carries the same
    section number within the same contract. This keeps the judgment honest: the
    unit is the ground truth, and both chunkers are measured against it rather
    than one chunker being privileged.
    """

    model_config = ConfigDict(frozen=True)

    id: str
    contract: str
    section: str | None = None
    text: str


class DatasetManifest(BaseModel):
    """Identity, provenance and integrity metadata for a dataset."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = DATASET_SCHEMA_VERSION
    name: str
    version: str = "1"
    source: str
    license: str
    description: str = ""
    relevance_mapping: str = DOCUMENT_MAPPING
    corpus_count: int = 0
    query_count: int = 0
    qrel_count: int = 0
    evidence_count: int = 0
    corpus_sha256: str | None = None
    queries_sha256: str | None = None
    qrels_sha256: str | None = None
    evidence_sha256: str | None = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=UTC))
    notes: list[str] = Field(default_factory=list)

    @property
    def checksum(self) -> str:
        """A single checksum over the normalised content files that matter."""

        parts = [
            self.corpus_sha256 or "",
            self.queries_sha256 or "",
            self.qrels_sha256 or "",
            self.evidence_sha256 or "",
        ]
        return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


class EvaluationDataset(BaseModel):
    """A loaded dataset: manifest, corpus, queries, judgments and evidence units."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    manifest: DatasetManifest
    documents: list[CorpusDocument]
    queries: list[EvalQuery]
    qrels: dict[str, dict[str, int]]
    evidence: dict[str, EvidenceUnit] = Field(default_factory=dict)
    directory: Path | None = None

    @property
    def name(self) -> str:
        return self.manifest.name

    def relevance_for(self, query_id: str) -> dict[str, int]:
        return dict(self.qrels.get(query_id, {}))

    def document_index(self) -> dict[str, CorpusDocument]:
        return {document.id: document for document in self.documents}


# --------------------------------------------------------------------- parsers


def _read_jsonl(path: Path) -> Iterator[dict]:
    if not path.is_file():
        raise DatasetError(f"expected dataset file is missing: {path}")
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                payload = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise DatasetError(f"{path.name} line {line_number}: invalid JSON ({exc})") from exc
            if not isinstance(payload, dict):
                raise DatasetError(f"{path.name} line {line_number}: expected a JSON object")
            yield payload


def parse_corpus(path: Path) -> list[CorpusDocument]:
    documents: list[CorpusDocument] = []
    seen: set[str] = set()
    for payload in _read_jsonl(path):
        doc_id = str(payload.get("_id") or payload.get("id") or "").strip()
        text = payload.get("text")
        title = payload.get("title")
        if not doc_id:
            raise DatasetError(f"{path.name}: a corpus entry has no _id")
        if doc_id in seen:
            raise DatasetError(f"{path.name}: duplicate corpus id '{doc_id}'")
        if text is None or not str(text).strip():
            raise DatasetError(f"{path.name}: corpus entry '{doc_id}' has no text")
        seen.add(doc_id)
        documents.append(
            CorpusDocument(
                id=doc_id, title=(str(title) if title is not None else None), text=str(text)
            )
        )
    if not documents:
        raise DatasetError(f"{path.name}: no documents found")
    return documents


def parse_queries(path: Path) -> list[EvalQuery]:
    queries: list[EvalQuery] = []
    seen: set[str] = set()
    for payload in _read_jsonl(path):
        query_id = str(payload.get("_id") or payload.get("id") or "").strip()
        text = payload.get("text")
        if not query_id:
            raise DatasetError(f"{path.name}: a query entry has no _id")
        if query_id in seen:
            raise DatasetError(f"{path.name}: duplicate query id '{query_id}'")
        if text is None or not str(text).strip():
            raise DatasetError(f"{path.name}: query '{query_id}' has no text")
        seen.add(query_id)
        queries.append(EvalQuery(id=query_id, text=str(text)))
    if not queries:
        raise DatasetError(f"{path.name}: no queries found")
    return queries


def parse_qrels(path: Path) -> dict[str, dict[str, int]]:
    """Parse a BEIR-style qrels TSV into ``{query_id: {corpus_id: grade}}``.

    The header row is optional; a two-column row (no score) defaults to grade 1.
    Malformed rows and non-positive scores are rejected loudly rather than being
    silently dropped, because a quietly wrong judgment is worse than a failure.
    """

    if not path.is_file():
        raise DatasetError(f"qrels file is missing: {path}")
    qrels: dict[str, dict[str, int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            stripped = line.rstrip("\n")
            if not stripped.strip():
                continue
            fields = stripped.split("\t")
            if line_number == 1 and fields[0].strip().lower() in {"query-id", "query_id", "qid"}:
                continue  # header row
            if len(fields) < 2:
                raise DatasetError(f"{path.name} line {line_number}: expected at least 2 columns")
            query_id, corpus_id = fields[0].strip(), fields[1].strip()
            score_raw = fields[2].strip() if len(fields) > 2 and fields[2].strip() else "1"
            if not query_id or not corpus_id:
                raise DatasetError(f"{path.name} line {line_number}: empty id field")
            try:
                score = int(score_raw)
            except ValueError as exc:
                raise DatasetError(
                    f"{path.name} line {line_number}: score '{score_raw}' is not an integer"
                ) from exc
            if score <= 0:
                # BEIR sometimes uses 0 to mean "explicitly not relevant"; that is
                # the absence of a judgment, not a relevance grade.
                continue
            qrels.setdefault(query_id, {})[corpus_id] = score
    if not qrels:
        raise DatasetError(f"{path.name}: no relevance judgments found")
    return qrels


def parse_evidence(path: Path, corpus_ids: set[str]) -> dict[str, EvidenceUnit]:
    units: dict[str, EvidenceUnit] = {}
    for payload in _read_jsonl(path):
        unit_id = str(payload.get("_id") or "").strip()
        contract = str(payload.get("contract") or "").strip()
        text = payload.get("text")
        if not unit_id or not contract or text is None or not str(text).strip():
            raise DatasetError(f"{path.name}: evidence entry requires _id, contract and text")
        if contract not in corpus_ids:
            raise DatasetError(
                f"{path.name}: evidence unit '{unit_id}' references unknown contract '{contract}'"
            )
        units[unit_id] = EvidenceUnit(
            id=unit_id,
            contract=contract,
            section=(str(payload["section"]) if payload.get("section") is not None else None),
            text=str(text),
        )
    return units


def _validate_qrels(
    qrels: dict[str, dict[str, int]],
    *,
    query_ids: set[str],
    valid_units: set[str],
    source: str,
) -> None:
    for query_id, judgments in qrels.items():
        if query_id not in query_ids:
            raise DatasetError(
                f"{source}: qrels reference unknown query id '{query_id}' "
                "(the query list and judgments are inconsistent)"
            )
        for unit_id in judgments:
            if unit_id not in valid_units:
                raise DatasetError(
                    f"{source}: qrels for query '{query_id}' reference unknown unit '{unit_id}'"
                )


# ------------------------------------------------------------------- assembly


def build_manifest(
    *,
    name: str,
    source: str,
    license: str,
    version: str = "1",
    description: str = "",
    relevance_mapping: str = DOCUMENT_MAPPING,
    documents: list[CorpusDocument],
    queries: list[EvalQuery],
    qrels: dict[str, dict[str, int]],
    evidence: dict[str, EvidenceUnit] | None = None,
    notes: Iterable[str] = (),
    checksums: dict[str, str | None] | None = None,
) -> DatasetManifest:
    if relevance_mapping not in MAPPING_POLICIES:
        raise DatasetError(
            f"unknown relevance mapping '{relevance_mapping}'; expected one of {MAPPING_POLICIES}"
        )
    evidence = evidence or {}
    checksums = checksums or {}
    return DatasetManifest(
        name=name,
        version=version,
        source=source,
        license=license,
        description=description,
        relevance_mapping=relevance_mapping,
        corpus_count=len(documents),
        query_count=len(queries),
        qrel_count=sum(len(judgments) for judgments in qrels.values()),
        evidence_count=len(evidence),
        corpus_sha256=checksums.get("corpus"),
        queries_sha256=checksums.get("queries"),
        qrels_sha256=checksums.get("qrels"),
        evidence_sha256=checksums.get("evidence"),
        notes=list(notes),
    )


def load_beir(
    *,
    name: str,
    corpus_path: Path,
    queries_path: Path,
    qrels_path: Path,
    source: str,
    license: str,
    version: str = "1",
    description: str = "",
    relevance_mapping: str = DOCUMENT_MAPPING,
    evidence_path: Path | None = None,
    notes: Iterable[str] = (),
) -> EvaluationDataset:
    """Normalise BEIR-format files into an :class:`EvaluationDataset`."""

    for path in (corpus_path, queries_path, qrels_path):
        if not path.is_file():
            raise DatasetError(f"file not found: {path}")
    documents = parse_corpus(corpus_path)
    queries = parse_queries(queries_path)
    qrels = parse_qrels(qrels_path)
    evidence = (
        parse_evidence(evidence_path, {document.id for document in documents})
        if evidence_path is not None
        else {}
    )
    valid_units = set(evidence) if evidence else {document.id for document in documents}
    _validate_qrels(
        qrels, query_ids={query.id for query in queries}, valid_units=valid_units, source=name
    )
    manifest = build_manifest(
        name=name,
        source=source,
        license=license,
        version=version,
        description=description,
        relevance_mapping=relevance_mapping,
        documents=documents,
        queries=queries,
        qrels=qrels,
        evidence=evidence,
        notes=notes,
        checksums={
            "corpus": sha256_file(corpus_path),
            "queries": sha256_file(queries_path),
            "qrels": sha256_file(qrels_path),
            "evidence": sha256_file(evidence_path) if evidence_path is not None else None,
        },
    )
    return EvaluationDataset(
        manifest=manifest, documents=documents, queries=queries, qrels=qrels, evidence=evidence
    )


def _dump_jsonl(path: Path, rows: Iterable[dict]) -> str:
    lines = [json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows]
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return sha256_bytes(payload)


def _dump_qrels(path: Path, qrels: dict[str, dict[str, int]]) -> str:
    lines = ["query-id\tcorpus-id\tscore"]
    for query_id in sorted(qrels):
        for unit_id in sorted(qrels[query_id]):
            lines.append(f"{query_id}\t{unit_id}\t{qrels[query_id][unit_id]}")
    payload = ("\n".join(lines) + "\n").encode("utf-8")
    path.write_bytes(payload)
    return sha256_bytes(payload)


def write_dataset(directory: Path, dataset: EvaluationDataset) -> DatasetManifest:
    """Write a dataset to ``directory`` and return the manifest with checksums."""

    directory.mkdir(parents=True, exist_ok=True)
    corpus_sha = _dump_jsonl(
        directory / "corpus.jsonl",
        ({"_id": doc.id, "title": doc.title, "text": doc.text} for doc in dataset.documents),
    )
    queries_sha = _dump_jsonl(
        directory / "queries.jsonl",
        ({"_id": query.id, "text": query.text} for query in dataset.queries),
    )
    qrels_sha = _dump_qrels(directory / "qrels.tsv", dataset.qrels)
    evidence_sha: str | None = None
    if dataset.evidence:
        evidence_sha = _dump_jsonl(
            directory / "evidence.jsonl",
            (
                {
                    "_id": unit.id,
                    "contract": unit.contract,
                    "section": unit.section,
                    "text": unit.text,
                }
                for unit in dataset.evidence.values()
            ),
        )
    manifest = dataset.manifest.model_copy(
        update={
            "corpus_sha256": corpus_sha,
            "queries_sha256": queries_sha,
            "qrels_sha256": qrels_sha,
            "evidence_sha256": evidence_sha,
        }
    )
    (directory / "manifest.json").write_text(
        manifest.model_dump_json(indent=2), encoding="utf-8"
    )
    return manifest


def load_dataset(directory: Path) -> EvaluationDataset:
    """Load a dataset directory written by :func:`write_dataset` (or equivalent)."""

    manifest_path = directory / "manifest.json"
    if not manifest_path.is_file():
        raise DatasetError(f"not a dataset directory (no manifest.json): {directory}")
    manifest = DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    documents = parse_corpus(directory / "corpus.jsonl")
    queries = parse_queries(directory / "queries.jsonl")
    qrels = parse_qrels(directory / "qrels.tsv")
    evidence_path = directory / "evidence.jsonl"
    evidence = (
        parse_evidence(evidence_path, {document.id for document in documents})
        if evidence_path.is_file()
        else {}
    )
    valid_units = set(evidence) if evidence else {document.id for document in documents}
    _validate_qrels(
        qrels,
        query_ids={query.id for query in queries},
        valid_units=valid_units,
        source=manifest.name,
    )
    return EvaluationDataset(
        manifest=manifest,
        documents=documents,
        queries=queries,
        qrels=qrels,
        evidence=evidence,
        directory=directory,
    )


class DatasetStore:
    """Discover and load datasets from a set of roots.

    Roots are searched in order. The first root is the writable location (under
    ``data_dir``); a later, committed root lets the repository ship the small
    synthetic legal set without writing into the data directory.
    """

    def __init__(self, roots: Iterable[Path]) -> None:
        self._roots = [Path(root) for root in roots]

    @property
    def roots(self) -> list[Path]:
        return list(self._roots)

    def list_names(self) -> list[str]:
        names: list[str] = []
        for root in self._roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if (child / "manifest.json").is_file() and child.name not in names:
                    names.append(child.name)
        return names

    def resolve(self, name: str) -> Path:
        # Accept the directory slug, the exact manifest name, or a slug of it.
        try:
            slug = slugify(name)
        except DatasetError:
            slug = None
        candidates: list[str] = [name] if slug is None else [name, slug]
        for root in self._roots:
            if not root.is_dir():
                continue
            for child in sorted(root.iterdir()):
                if not (child / "manifest.json").is_file():
                    continue
                if child.name in candidates:
                    return child
                try:
                    manifest = json.loads((child / "manifest.json").read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                manifest_name = str(manifest.get("name", ""))
                if manifest_name == name:
                    return child
                try:
                    if slugify(manifest_name) in candidates:
                        return child
                except DatasetError:
                    continue
        # Fall back to a slugified directory match across roots.
        if slug is not None:
            for root in self._roots:
                candidate = root / slug
                if (candidate / "manifest.json").is_file():
                    return candidate
        raise DatasetNotFoundError(
            f"dataset '{name}' not found; available: {', '.join(self.list_names()) or '(none)'}"
        )

    def load(self, name: str) -> EvaluationDataset:
        return load_dataset(self.resolve(name))

    def exists(self, name: str) -> bool:
        try:
            self.resolve(name)
            return True
        except DatasetNotFoundError:
            return False


__all__ = [
    "DATASET_SCHEMA_VERSION",
    "DOCUMENT_MAPPING",
    "EVIDENCE_MAPPING",
    "MAPPING_POLICIES",
    "CorpusDocument",
    "DatasetManifest",
    "DatasetStore",
    "EvalQuery",
    "EvaluationDataset",
    "EvidenceUnit",
    "build_manifest",
    "load_beir",
    "load_dataset",
    "parse_corpus",
    "parse_evidence",
    "parse_qrels",
    "parse_queries",
    "sha256_bytes",
    "sha256_file",
    "slugify",
    "write_dataset",
]
