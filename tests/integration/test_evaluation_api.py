"""API-level tests for evaluation endpoints and read-only enforcement.

These drive the real FastAPI app over an isolated SQLite database and a tiny
synthetic dataset written into the test dataset root. No network, no models.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_workbench.api.app import create_app
from rag_workbench.config import Settings
from rag_workbench.evaluation.datasets import (
    CorpusDocument,
    EvalQuery,
    EvaluationDataset,
    build_manifest,
    write_dataset,
)

_DOCS = [
    CorpusDocument(id="D1", title="Alpha", text="Alpha encodes text into latent vectors."),
    CorpusDocument(id="D2", title="Beta", text="Beta performs keyword search over FTS5."),
    CorpusDocument(
        id="D3", title="Gamma", text="Gamma fuses rankings with reciprocal rank fusion."
    ),
]
_QUERIES = [
    EvalQuery(id="Q1", text="latent vectors"),
    EvalQuery(id="Q2", text="keyword search"),
    EvalQuery(id="Q3", text="reciprocal rank fusion"),
]
_QRELS = {"Q1": {"D1": 1}, "Q2": {"D2": 1}, "Q3": {"D3": 1}}


def _write_small_dataset(root: Path) -> None:
    manifest = build_manifest(
        name="eval-small",
        source="synthetic",
        license="CC0",
        documents=_DOCS,
        queries=_QUERIES,
        qrels=_QRELS,
    )
    write_dataset(
        root / "eval-small",
        EvaluationDataset(
            manifest=manifest, documents=_DOCS, queries=_QUERIES, qrels=_QRELS
        ),
    )


def test_dataset_listing_and_detail(client: TestClient, settings: Settings) -> None:
    _write_small_dataset(settings.datasets_dir)

    listing = client.get("/v1/datasets")
    assert listing.status_code == 200
    slugs = [item["slug"] for item in listing.json()["datasets"]]
    assert "eval-small" in slugs

    detail = client.get("/v1/datasets/eval-small")
    assert detail.status_code == 200
    body = detail.json()
    assert body["corpus_count"] == 3
    assert body["query_count"] == 3
    assert body["relevance_mapping"] == "document"

    missing = client.get("/v1/datasets/does-not-exist")
    assert missing.status_code == 404
    assert missing.json()["error"] == "dataset_not_found"


def test_ingest_run_inspect_export_round_trip(
    client: TestClient, settings: Settings
) -> None:
    _write_small_dataset(settings.datasets_dir)

    ingest = client.post(
        "/v1/datasets/ingest",
        json={"dataset": "eval-small", "strategy": "paragraph"},
    )
    assert ingest.status_code == 200, ingest.text
    ingest_body = ingest.json()
    assert ingest_body["documents"] == 3
    profile_id = ingest_body["profile_id"]

    run = client.post(
        "/v1/evaluations",
        json={
            "dataset": "eval-small",
            "profile_id": profile_id,
            "collection": "eval-small",
            "mode": "bm25",
            "candidate_depth": 10,
            "top_k": 3,
            "k_values": [1, 3],
        },
    )
    assert run.status_code == 200, run.text
    record = run.json()
    experiment_id = record["experiment_id"]
    assert record["aggregate"]["recall@1"] == 1.0
    assert record["config"]["query_mode"] == "or"

    listed = client.get("/v1/experiments", params={"dataset": "eval-small"})
    assert listed.status_code == 200
    ids = [item["experiment_id"] for item in listed.json()["experiments"]]
    assert experiment_id in ids

    fetched = client.get(f"/v1/experiments/{experiment_id}")
    assert fetched.status_code == 200
    assert fetched.json()["experiment_id"] == experiment_id

    single = client.get(f"/v1/experiments/{experiment_id}/queries/Q1")
    assert single.status_code == 200
    assert single.json()["trace"][0]["key"] == "D1"

    as_json = client.get(f"/v1/experiments/{experiment_id}/export", params={"format": "json"})
    assert as_json.status_code == 200
    assert as_json.json()["experiment_id"] == experiment_id

    as_csv = client.get(f"/v1/experiments/{experiment_id}/export", params={"format": "csv"})
    assert as_csv.status_code == 200
    assert as_csv.headers["content-type"].startswith("text/csv")
    assert as_csv.text.splitlines()[0].startswith("experiment_id,query_id")

    missing = client.get("/v1/experiments/exp-does-not-exist")
    assert missing.status_code == 404


def test_invalid_candidate_pool_is_rejected(client: TestClient, settings: Settings) -> None:
    _write_small_dataset(settings.datasets_dir)
    ingest = client.post(
        "/v1/datasets/ingest", json={"dataset": "eval-small", "strategy": "paragraph"}
    )
    profile_id = ingest.json()["profile_id"]
    bad = client.post(
        "/v1/evaluations",
        json={
            "dataset": "eval-small",
            "profile_id": profile_id,
            "collection": "eval-small",
            "top_k": 10,
            "rerank": True,
            "rerank_candidates": 3,
        },
    )
    assert bad.status_code == 422
    assert bad.json()["error"] == "invalid_candidates"


def test_config_endpoint_reports_capabilities(client: TestClient) -> None:
    response = client.get("/v1/config")
    assert response.status_code == 200
    body = response.json()
    assert body["read_only"] is False
    assert body["autoseed_sample"] is False
    assert "bm25" in body["supported_modes"]
    assert body["default_candidate_depth"] >= 1
    # The deployment revision is distinct from the package version and reports
    # how it was resolved.
    assert body["version"]
    assert body["revision_source"] in {"git", "git-file", "source-hash", "unavailable"}


@pytest.mark.parametrize(
    "method,path,payload",
    [
        ("post", "/v1/documents", None),
        ("post", "/v1/datasets/ingest", {"dataset": "x"}),
        ("post", "/v1/evaluations", {"dataset": "x", "profile_id": "y"}),
        ("post", "/v1/vectors/rebuild", {"all_profiles": True}),
    ],
)
def test_read_only_refuses_mutations(
    settings: Settings, method: str, path: str, payload: dict | None
) -> None:
    read_only = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        read_only=True,
        _env_file=None,
    )
    app = create_app(read_only)
    with TestClient(app) as client:
        call = getattr(client, method)
        response = call(path) if payload is None else call(path, json=payload)
    assert response.status_code == 403
    assert response.json()["error"] == "read_only_mode"


def test_read_only_still_serves_reads(settings: Settings) -> None:
    _write_small_dataset(settings.datasets_dir)
    read_only = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        read_only=True,
        _env_file=None,
    )
    app = create_app(read_only)
    with TestClient(app) as client:
        assert client.get("/v1/datasets").status_code == 200
        assert client.get("/v1/config").status_code == 200
        assert client.get("/health").status_code == 200


def test_autoseed_sample_populates_demo_corpus(settings: Settings) -> None:
    seeded = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        autoseed_sample=True,
        _env_file=None,
    )
    app = create_app(seeded)
    with TestClient(app) as client:
        documents = client.get("/v1/documents")
        assert documents.status_code == 200
        assert len(documents.json()["documents"]) >= 1
