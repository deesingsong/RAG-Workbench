"""Optional Locust load-test definition for a locally running RAG Workbench API.

This mirrors the built-in ``rag-workbench scaling load-test`` workloads but gives
Locust's richer reporting. Locust is intentionally **not** a project dependency:
run it through ``uv`` so nothing is added to the lockfile or the core install::

    uv run --with locust locust -f benchmarks/load/locustfile.py --host http://127.0.0.1:8765

Only read-only endpoints are exercised. Never point this at the public Streamlit
Community Cloud deployment.
"""

from __future__ import annotations

from locust import HttpUser, between, task


class ReadOnlyUser(HttpUser):
    """Search, health and recorded-artifact reads against a local API."""

    wait_time = between(0.05, 0.25)

    @task(5)
    def search(self) -> None:
        self.client.post(
            "/v1/search",
            json={"query": "termination for convenience", "mode": "bm25", "top_k": 10},
            name="/v1/search [bm25]",
        )

    @task(3)
    def health(self) -> None:
        self.client.get("/health")

    @task(1)
    def experiments(self) -> None:
        self.client.get("/v1/experiments")

    @task(1)
    def recorded_artifacts(self) -> None:
        self.client.get("/v1/scaling-benchmarks")
        self.client.get("/v1/answer-demos")
