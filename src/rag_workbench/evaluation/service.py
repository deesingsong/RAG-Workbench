"""Application-facing evaluation service.

This is the single entry point the API and CLI use for evaluation: listing and
importing datasets, ingesting a corpus through the ordinary ingestion path,
running a configuration and persisting the experiment, and merging live and
recorded experiments. Streamlit calls the same behaviour over HTTP, so it never
re-implements any of it.
"""

from __future__ import annotations

import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

from rag_workbench.config import IndexProfileSpec, Settings
from rag_workbench.domain.errors import DatasetError, ExperimentNotFoundError
from rag_workbench.evaluation.datasets import (
    DOCUMENT_MAPPING,
    DatasetStore,
    EvaluationDataset,
    load_beir,
    sha256_file,
    slugify,
    write_dataset,
)
from rag_workbench.evaluation.experiments import (
    ExperimentStore,
    find_recorded,
    load_recorded,
    load_recorded_summary,
)
from rag_workbench.evaluation.ingest import CorpusImportReport, import_corpus
from rag_workbench.evaluation.runner import EvalRunConfig, EvaluationRunner, ExperimentResult
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.profiles import build_profile
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService

#: The BEIR SciFact archive (public). Download is only ever triggered explicitly.
SCIFACT_URL = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/scifact.zip"
SCIFACT_SOURCE = "BEIR SciFact (https://github.com/allenai/scifact)"
SCIFACT_LICENSE = "CC BY 4.0 (SciFact release)"


def human_size(num_bytes: int) -> str:
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} GB"


class EvaluationService:
    """Datasets, corpus ingestion, runs and experiment records."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        ingestion_service: IngestionService,
        search_service: SearchService,
        *,
        git_root: Path | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._ingestion = ingestion_service
        self._search = search_service
        self._datasets = DatasetStore(settings.dataset_roots())
        self._experiments = ExperimentStore(database)
        self._recorded_dir = Path(settings.recorded_experiments_dir)
        self._git_root = git_root

    # ----------------------------------------------------------------- datasets

    def list_datasets(self) -> list[dict]:
        summaries: list[dict] = []
        for slug in self._datasets.list_names():
            dataset = self._datasets.load(slug)
            summaries.append(self._dataset_summary(dataset, slug))
        return summaries

    def dataset_summary(self, name: str) -> dict:
        dataset = self._datasets.load(name)
        return self._dataset_summary(dataset, self._datasets.resolve(name).name)

    def load_dataset(self, name: str) -> EvaluationDataset:
        return self._datasets.load(name)

    def import_beir(
        self,
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
        overwrite: bool = True,
    ) -> dict:
        dataset = load_beir(
            name=name,
            corpus_path=corpus_path,
            queries_path=queries_path,
            qrels_path=qrels_path,
            source=source,
            license=license,
            version=version,
            description=description,
            relevance_mapping=relevance_mapping,
            evidence_path=evidence_path,
            notes=notes,
        )
        slug = slugify(name)
        target = self._settings.datasets_dir / slug
        if target.exists() and not overwrite:
            raise DatasetError(f"dataset '{name}' already exists at {target}")
        manifest = write_dataset(target, dataset)
        dataset.manifest = manifest
        dataset.directory = target
        return self._dataset_summary(dataset, slug)

    def download_and_import_scifact(self, *, name: str = "scifact", version: str = "beir") -> dict:
        """Download the SciFact archive and import it (explicit command only)."""

        with tempfile.TemporaryDirectory(prefix="scifact_") as workdir:
            archive = Path(workdir) / "scifact.zip"
            self._download(SCIFACT_URL, archive)
            extracted = Path(workdir) / "scifact"
            with zipfile.ZipFile(archive) as zip_file:
                zip_file.extractall(workdir)
            if not extracted.is_dir():
                raise DatasetError("SciFact archive did not contain a 'scifact' directory")
            qrels_dir = extracted / "qrels"
            tsv_files = sorted(qrels_dir.glob("*.tsv")) if qrels_dir.is_dir() else []
            if not tsv_files:
                raise DatasetError("SciFact archive did not contain qrels/*.tsv")
            # SciFact ships train/dev/test judgments; the *test* split is the
            # benchmark. Pick it explicitly rather than by alphabetical order
            # (which would select "train.tsv").
            preferred = [path for path in tsv_files if path.stem.lower() == "test"]
            qrels_path = preferred[0] if preferred else tsv_files[-1]
            return self.import_beir(
                name=name,
                corpus_path=extracted / "corpus.jsonl",
                queries_path=extracted / "queries.jsonl",
                qrels_path=qrels_path,
                source=SCIFACT_SOURCE,
                license=SCIFACT_LICENSE,
                version=version,
                description="BEIR SciFact scientific-claim verification retrieval benchmark.",
                notes=[
                    "Document-level qrels; chunk retrieval is mapped to document IDs "
                    "before evaluation.",
                    "Retrieval depth is a chunk count, so K unique documents may require "
                    "a deeper chunk pool.",
                ],
            )

    @staticmethod
    def _download(url: str, destination: Path) -> None:
        import httpx

        try:
            with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
                response.raise_for_status()
                with destination.open("wb") as handle:
                    for chunk in response.iter_bytes():
                        handle.write(chunk)
        except Exception as exc:  # noqa: BLE001 - surface a single actionable error
            raise DatasetError(f"could not download {url}: {exc}") from exc

    def ingest_dataset(
        self,
        name: str,
        *,
        collection: str | None = None,
        strategy: str | None = None,
        max_tokens: int | None = None,
        overlap_tokens: int | None = None,
        profile_name: str | None = None,
        embed: bool = False,
        embedding_model: str | None = None,
        embedding_dimension: int | None = None,
        limit: int | None = None,
        profile: IndexProfileSpec | None = None,
    ) -> CorpusImportReport:
        dataset = self._datasets.load(name)
        resolved_profile = profile or build_profile(
            self._settings,
            strategy=strategy,
            max_tokens=max_tokens,
            overlap_tokens=overlap_tokens,
            name=profile_name,
            embed=embed,
            embedding_model=embedding_model,
            embedding_dimension=embedding_dimension,
        )
        resolved_collection = collection or self._datasets.resolve(name).name
        return import_corpus(
            self._ingestion,
            dataset,
            collection=resolved_collection,
            profile=resolved_profile,
            limit=limit,
        )

    # -------------------------------------------------------------- experiments

    def run(self, config: EvalRunConfig) -> dict:
        dataset = self._datasets.load(config.dataset)
        runner = EvaluationRunner(
            self._db, self._settings, self._search, git_root=self._git_root
        )
        result: ExperimentResult = runner.run(dataset, config)
        experiment_id = self._experiments.save(result)
        result.experiment_id = experiment_id
        return self._experiments.get(experiment_id)

    def list_experiments(
        self, *, dataset: str | None = None, include_recorded: bool = True
    ) -> list[dict]:
        summaries = self._experiments.list_experiments(dataset=dataset)
        seen = {item["experiment_id"] for item in summaries}
        if include_recorded:
            for record in load_recorded(self._recorded_dir):
                if dataset is not None and record.get("dataset", {}).get("name") != dataset:
                    continue
                # A locally-run experiment that was also committed as an artifact
                # is listed once, from its live row.
                if record.get("experiment_id") in seen:
                    continue
                summaries.append(load_recorded_summary(record))
        summaries.sort(key=lambda item: str(item.get("created_at", "")), reverse=True)
        return summaries

    def get_experiment(self, experiment_id: str, *, include_recorded: bool = True) -> dict:
        try:
            return self._experiments.get(experiment_id)
        except ExperimentNotFoundError:
            if include_recorded:
                recorded = find_recorded(self._recorded_dir, experiment_id)
                if recorded is not None:
                    return recorded
            raise

    def get_query(self, experiment_id: str, query_id: str) -> dict:
        try:
            return self._experiments.get_query(experiment_id, query_id)
        except ExperimentNotFoundError:
            recorded = find_recorded(self._recorded_dir, experiment_id)
            if recorded is not None:
                for query in recorded.get("per_query", []):
                    if query.get("query_id") == query_id:
                        return query
            raise

    def export_json(self, experiment_id: str) -> dict:
        return self.get_experiment(experiment_id)

    def export_csv(self, experiment_id: str) -> str:
        from rag_workbench.evaluation.experiments import record_to_csv

        return record_to_csv(self.get_experiment(experiment_id))

    def recorded_dir(self) -> Path:
        return self._recorded_dir

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _dataset_summary(dataset: EvaluationDataset, slug: str) -> dict:
        manifest = dataset.manifest
        return {
            "name": manifest.name,
            "slug": slug,
            "version": manifest.version,
            "source": manifest.source,
            "license": manifest.license,
            "description": manifest.description,
            "relevance_mapping": manifest.relevance_mapping,
            "corpus_count": manifest.corpus_count,
            "query_count": manifest.query_count,
            "qrel_count": manifest.qrel_count,
            "evidence_count": manifest.evidence_count,
            "checksum": manifest.checksum,
            "created_at": manifest.created_at.isoformat(),
            "notes": list(manifest.notes),
            "directory": str(dataset.directory) if dataset.directory else None,
        }


def copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def checksum(path: Path) -> str:
    return sha256_file(path)


__all__ = [
    "SCIFACT_LICENSE",
    "SCIFACT_SOURCE",
    "SCIFACT_URL",
    "EvaluationService",
    "human_size",
]
