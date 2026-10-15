"""End-to-end evaluation tests: real SQLite, real BM25, real persistence.

No models, no network: the dense/rerank paths are covered elsewhere with fakes,
and these tests exercise the full evaluation pipeline on tiny synthetic datasets.
"""

from __future__ import annotations

import json
from pathlib import Path

from rag_workbench.config import ChunkerConfig, IndexProfileSpec, Settings
from rag_workbench.evaluation.datasets import (
    EVIDENCE_MAPPING,
    CorpusDocument,
    EvalQuery,
    EvaluationDataset,
    EvidenceUnit,
    build_manifest,
    write_dataset,
)
from rag_workbench.evaluation.experiments import ExperimentStore
from rag_workbench.evaluation.ingest import import_corpus
from rag_workbench.evaluation.runner import EvalRunConfig, EvaluationRunner
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService

_DOCS = [
    CorpusDocument(
        id="D1", title="Alpha", text="The alpha system encodes text into latent vectors."
    ),
    CorpusDocument(
        id="D2", title="Beta", text="The beta system performs keyword search over FTS5."
    ),
    CorpusDocument(
        id="D3", title="Gamma", text="The gamma system fuses rankings with reciprocal rank fusion."
    ),
]
_QUERIES = [
    EvalQuery(id="Q1", text="latent vectors"),
    EvalQuery(id="Q2", text="keyword search"),
    EvalQuery(id="Q3", text="reciprocal rank fusion"),
]
_QRELS = {"Q1": {"D1": 1}, "Q2": {"D2": 1}, "Q3": {"D3": 1}}


def _dataset(directory: Path) -> EvaluationDataset:
    manifest = build_manifest(
        name="eval-small",
        source="synthetic",
        license="CC0",
        documents=_DOCS,
        queries=_QUERIES,
        qrels=_QRELS,
    )
    dataset = EvaluationDataset(
        manifest=manifest, documents=_DOCS, queries=_QUERIES, qrels=_QRELS
    )
    write_dataset(directory, dataset)
    return EvaluationDataset(
        manifest=manifest,
        documents=_DOCS,
        queries=_QUERIES,
        qrels=_QRELS,
        directory=directory,
    )


def _services(settings: Settings, database: Database):
    ingestion = IngestionService(database, settings)
    search = SearchService(database, settings)
    runner = EvaluationRunner(database, settings, search)
    return ingestion, runner


def test_document_level_evaluation_end_to_end(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    dataset = _dataset(tmp_path / "datasets" / "eval-small")
    ingestion, runner = _services(settings, database)
    profile = IndexProfileSpec(name="paragraph-220", chunker=ChunkerConfig())
    report = import_corpus(
        ingestion, dataset, collection="eval-small", profile=profile
    )
    assert report.documents == 3
    assert report.chunks >= 3
    assert report.diagnostics == []

    config = EvalRunConfig(
        dataset="eval-small",
        profile_id=profile.key,
        collection="eval-small",
        mode="bm25",
        candidate_depth=10,
        top_k=3,
        k_values=(1, 3),
    )
    result = runner.run(dataset, config)
    assert result.counts["queries_run"] == 3
    assert result.counts["queries_evaluated"] == 3
    # Every query's single relevant document is rank 1 with these tiny fixtures.
    assert result.aggregate["recall@1"] == 1.0
    assert result.aggregate["mrr@1"] == 1.0
    assert result.aggregate["ndcg@3"] == 1.0
    assert result.aggregate["map@3"] == 1.0
    assert result.candidate_metrics["candidate_recall@10"] == 1.0

    trace = result.per_query[0].trace
    assert trace and trace[0].key == "D1"
    assert trace[0].external_document_id == "D1"
    assert trace[0].relevant is True


class _CountingSearch:
    """Wrap a real search service and record the queries it was asked to run."""

    def __init__(self, inner: SearchService) -> None:
        self._inner = inner
        self.queries: list[str] = []

    def search(self, **kwargs):  # noqa: ANN003, ANN201 - transparent passthrough
        self.queries.append(kwargs["query"])
        return self._inner.search(**kwargs)


def test_unjudged_queries_are_skipped_before_retrieval(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    queries = [
        EvalQuery(id="Q1", text="system"),
        EvalQuery(id="QX", text="a query with no judgments at all"),
    ]
    qrels = {"Q1": {"D1": 1}}
    manifest = build_manifest(
        name="eval-skip",
        source="synthetic",
        license="CC0",
        documents=_DOCS,
        queries=queries,
        qrels=qrels,
    )
    directory = tmp_path / "datasets" / "eval-skip"
    write_dataset(
        directory,
        EvaluationDataset(
            manifest=manifest, documents=_DOCS, queries=queries, qrels=qrels
        ),
    )
    dataset = EvaluationDataset(
        manifest=manifest,
        documents=_DOCS,
        queries=queries,
        qrels=qrels,
        directory=directory,
    )

    ingestion = IngestionService(database, settings)
    profile = IndexProfileSpec(name="paragraph-220", chunker=ChunkerConfig())
    import_corpus(ingestion, dataset, collection="eval-skip", profile=profile)

    counting = _CountingSearch(SearchService(database, settings))
    runner = EvaluationRunner(database, settings, counting)
    result = runner.run(
        dataset,
        EvalRunConfig(
            dataset="eval-skip",
            profile_id=profile.key,
            collection="eval-skip",
            top_k=3,
            k_values=(1, 3),
        ),
    )

    # The unjudged query was never retrieved; only the judged one was.
    assert counting.queries == ["system"]
    assert result.counts["queries_retrieved"] == 1
    assert result.counts["queries_excluded_no_qrels"] == 1
    # An excluded query must not be counted as an insufficient-candidate run.
    assert result.counts["queries_insufficient_candidates"] == 0
    # Latency covers only the query that actually ran.
    assert result.latency["queries"] == 1
    assert result.latency["retrieval_ms"]["count"] == 1
    assert result.latency["total_ms"]["count"] == 1

    excluded = next(outcome for outcome in result.per_query if outcome.query_id == "QX")
    assert excluded.metrics is None
    assert excluded.timings_ms == {}
    assert excluded.trace == []
    assert result.notes and "skipped before retrieval" in result.notes[0]


def test_experiment_persistence_and_export_round_trip(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    dataset = _dataset(tmp_path / "datasets" / "eval-small")
    ingestion, runner = _services(settings, database)
    profile = IndexProfileSpec(name="paragraph-220", chunker=ChunkerConfig())
    import_corpus(ingestion, dataset, collection="eval-small", profile=profile)

    config = EvalRunConfig(
        dataset="eval-small",
        profile_id=profile.key,
        collection="eval-small",
        top_k=3,
        k_values=(1, 3),
    )
    result = runner.run(dataset, config)
    store = ExperimentStore(database)
    experiment_id = store.save(result)

    summaries = store.list_experiments(dataset="eval-small")
    assert [item["experiment_id"] for item in summaries] == [experiment_id]

    record = store.get(experiment_id)
    assert record["aggregate"] == result.aggregate
    assert len(record["per_query"]) == 3
    assert record["source"] == "live"

    exported = store.export_json(experiment_id)
    assert json.loads(json.dumps(exported))["experiment_id"] == experiment_id

    csv_text = store.export_csv(experiment_id)
    lines = csv_text.strip().splitlines()
    assert lines[0].startswith("experiment_id,query_id")
    assert len(lines) == 4  # header + 3 queries
    assert "recall@1" in lines[0]

    # A per-query lookup drives the failure explorer.
    single = store.get_query(experiment_id, "Q1")
    assert single["query_id"] == "Q1"
    assert single["trace"][0]["key"] == "D1"


def test_evidence_mapping_only_counts_matching_clause(
    settings: Settings, database: Database, tmp_path: Path
) -> None:
    contract = (
        "1. Definitions\n"
        '"Confidential Information" means non-public technical data.\n\n'
        "2. Payment Terms\n"
        "Invoices are payable within thirty days of receipt.\n"
    )
    documents = [CorpusDocument(id="C1", title="Contract", text=contract)]
    queries = [
        EvalQuery(id="QP", text="payable within thirty days"),
        EvalQuery(id="QC", text="non-public technical data"),
    ]
    qrels = {"QP": {"C1::2": 1}, "QC": {"C1::1": 1}}
    evidence = {
        "C1::1": EvidenceUnit(
            id="C1::1",
            contract="C1",
            section="1",
            text='"Confidential Information" means non-public technical data.',
        ),
        "C1::2": EvidenceUnit(
            id="C1::2",
            contract="C1",
            section="2",
            text="Invoices are payable within thirty days of receipt.",
        ),
    }
    manifest = build_manifest(
        name="legal-small",
        source="synthetic",
        license="CC0",
        relevance_mapping=EVIDENCE_MAPPING,
        documents=documents,
        queries=queries,
        qrels=qrels,
        evidence=evidence,
    )
    directory = tmp_path / "datasets" / "legal-small"
    write_dataset(
        directory,
        EvaluationDataset(
            manifest=manifest,
            documents=documents,
            queries=queries,
            qrels=qrels,
            evidence=evidence,
        ),
    )
    dataset = EvaluationDataset(
        manifest=manifest,
        documents=documents,
        queries=queries,
        qrels=qrels,
        evidence=evidence,
        directory=directory,
    )

    ingestion, runner = _services(settings, database)
    profile = IndexProfileSpec(
        name="clause", chunker=ChunkerConfig(strategy="clause_aware")
    )
    import_corpus(ingestion, dataset, collection="legal-small", profile=profile)
    config = EvalRunConfig(
        dataset="legal-small",
        profile_id=profile.key,
        collection="legal-small",
        mode="bm25",
        candidate_depth=10,
        top_k=2,
        k_values=(1, 2),
    )
    result = runner.run(dataset, config)
    assert result.config["relevance_mapping"] == EVIDENCE_MAPPING
    # The payment query must retrieve the payment clause, not the definitions one.
    payment = next(q for q in result.per_query if q.query_id == "QP")
    assert payment.status == "ok"
    top = payment.trace[0]
    assert top.key == "C1::2"
    assert top.relevant is True
    assert payment.metrics is not None
    assert payment.metrics["recall@1"] == 1.0
