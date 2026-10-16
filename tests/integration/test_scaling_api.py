"""Scalability API endpoints: recorded benchmarks, capacity and architecture."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from rag_workbench.api.app import create_app
from rag_workbench.config import Settings


def _artifact(benchmark_id: str = "scale-it") -> dict:
    return {
        "schema_version": "1",
        "benchmark_id": benchmark_id,
        "created_at": "2026-10-16T10:00:00+00:00",
        "git_commit": "abc123",
        "environment": {"python": "3.12.0", "cpu_count": 8},
        "corpus": {"documents": 100, "config": {"seed": 0}},
        "workloads": [
            {
                "documents": 100,
                "chunks": 1000,
                "ingestion": {"documents_per_second": 42.0},
                "index": {"db_bytes": 4096, "fts_bytes": 1024},
                "retrieval": [
                    {
                        "mode": "bm25",
                        "candidate_depth": 10,
                        "latency_ms": {"p50": 1.0, "p95": 2.0},
                    }
                ],
            }
        ],
        "limitations": ["single machine"],
    }


@pytest.fixture
def scaling_client(tmp_path: Path, settings: Settings) -> TestClient:
    recorded = tmp_path / "recorded_scaling"
    recorded.mkdir()
    (recorded / "scale-it.json").write_text(json.dumps(_artifact()), encoding="utf-8")
    tuned = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        recorded_scaling_dir=recorded,
        _env_file=None,
    )
    app = create_app(tuned)
    with TestClient(app) as client:
        yield client


def test_list_scaling_benchmarks(scaling_client: TestClient) -> None:
    response = scaling_client.get("/v1/scaling-benchmarks")

    assert response.status_code == 200
    body = response.json()
    assert body["total"] == 1
    summary = body["benchmarks"][0]
    assert summary["benchmark_id"] == "scale-it"
    assert summary["documents_max"] == 100
    assert summary["modes"] == ["bm25"]


def test_get_scaling_benchmark_detail(scaling_client: TestClient) -> None:
    response = scaling_client.get("/v1/scaling-benchmarks/scale-it")

    assert response.status_code == 200
    body = response.json()
    assert body["benchmark_id"] == "scale-it"
    assert body["workloads"][0]["documents"] == 100


def test_unknown_scaling_benchmark_returns_404(scaling_client: TestClient) -> None:
    response = scaling_client.get("/v1/scaling-benchmarks/does-not-exist")

    assert response.status_code == 404


def test_capacity_endpoint_computes_storage_and_ingestion(scaling_client: TestClient) -> None:
    response = scaling_client.post(
        "/v1/scaling/capacity",
        json={
            "storage": {"documents": 1_000_000, "chunks_per_document": 5},
            "ingestion": {
                "documents_per_day": 1000,
                "chunks_per_document": 5,
                "embedding_chunks_per_second_per_worker": 10,
                "workers": 1,
            },
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["storage"]["chunks_total"] == 5_000_000
    assert body["storage"]["total_bytes"] > 0
    assert body["ingestion"]["keeps_up"] is True
    # Scenarios are always returned alongside the explicit estimate.
    assert set(body["scenarios"]) == {"small", "typical", "large"}


def test_capacity_rejects_invalid_assumptions(scaling_client: TestClient) -> None:
    response = scaling_client.post(
        "/v1/scaling/capacity", json={"storage": {"documents": 0}}
    )

    assert response.status_code == 422


def test_scenarios_endpoint(scaling_client: TestClient) -> None:
    response = scaling_client.get("/v1/scaling/scenarios")

    assert response.status_code == 200
    assert set(response.json()["scenarios"]) == {"small", "typical", "large"}


def test_architecture_endpoint(scaling_client: TestClient) -> None:
    response = scaling_client.get("/v1/scaling/architecture")

    assert response.status_code == 200
    body = response.json()
    assert "current_diagram" in body
    assert any(row["area"] for row in body["rows"])


def test_capacity_works_in_read_only_mode(tmp_path: Path, settings: Settings) -> None:
    """The capacity calculator is pure arithmetic and must not be gated."""

    read_only = Settings(
        data_dir=settings.data_dir,
        log_level="WARNING",
        read_only=True,
        _env_file=None,
    )
    app = create_app(read_only)
    with TestClient(app) as client:
        response = client.post(
            "/v1/scaling/capacity", json={"storage": {"documents": 10_000_000}}
        )

    assert response.status_code == 200
    assert response.json()["storage"]["total_bytes"] > 0


def test_config_reports_scaling_artifacts(scaling_client: TestClient) -> None:
    response = scaling_client.get("/v1/config")

    assert response.status_code == 200
    body = response.json()
    assert body["recorded_scaling_benchmarks"] == 1
    assert body["recorded_scaling_exists"] is True
    assert body["scaling_max_documents"] > 0
