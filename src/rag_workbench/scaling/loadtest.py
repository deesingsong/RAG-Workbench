"""Lightweight local load-testing harness.

This drives a *locally controlled* FastAPI server over HTTP and reports
throughput and tail latency for read-only endpoints. It uses ``httpx`` (already a
dependency) so load testing works without installing anything extra; Locust is
supported as an optional alternative for richer test shapes (see
``benchmarks/load/locustfile.py`` and the README).

It deliberately measures three different things that must not be conflated:

1. HTTP/API throughput (requests per second across endpoints);
2. retrieval throughput (BM25 search latency under concurrency);
3. model-inference throughput (only if a generation workload is explicitly added).

Never point this at the public Streamlit Community Cloud deployment: it targets a
locally controlled server, bounded by configured concurrency and duration.
"""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass, field

import httpx

from rag_workbench.evaluation.runtime import latency_stats


@dataclass(frozen=True)
class LoadStep:
    """One request shape in a workload."""

    name: str
    method: str
    path: str
    json: dict | None = None
    params: dict | None = None
    weight: int = 1


@dataclass
class LoadResult:
    name: str
    users: int
    duration_s: float
    requests: int = 0
    errors: int = 0
    status_counts: dict[str, int] = field(default_factory=dict)
    latencies_ms: list[float] = field(default_factory=list)

    def summary(self) -> dict[str, object]:
        stats = latency_stats(self.latencies_ms)
        elapsed = max(self.duration_s, 1e-9)
        return {
            "workload": self.name,
            "concurrent_users": self.users,
            "duration_s": round(self.duration_s, 3),
            "requests": self.requests,
            "errors": self.errors,
            "error_rate": round(self.errors / self.requests, 4) if self.requests else None,
            "requests_per_second": round(self.requests / elapsed, 3),
            "status_counts": dict(self.status_counts),
            "latency_ms": stats,
        }


#: Read-only workloads. All endpoints exist in the current API and mutate nothing.
READ_ONLY_WORKLOADS: dict[str, list[LoadStep]] = {
    "health": [LoadStep("health", "GET", "/health")],
    "bm25_search": [
        LoadStep(
            "search",
            "POST",
            "/v1/search",
            json={"query": "termination for convenience", "mode": "bm25", "top_k": 10},
        )
    ],
    "experiments": [LoadStep("experiments", "GET", "/v1/experiments")],
    "recorded_artifacts": [
        LoadStep("answer_demos", "GET", "/v1/answer-demos"),
        LoadStep("scaling_benchmarks", "GET", "/v1/scaling-benchmarks"),
    ],
    "mixed": [
        LoadStep("health", "GET", "/health", weight=3),
        LoadStep(
            "search",
            "POST",
            "/v1/search",
            json={"query": "confidential information", "mode": "bm25", "top_k": 5},
            weight=5,
        ),
        LoadStep("experiments", "GET", "/v1/experiments", weight=1),
        LoadStep("answer_demos", "GET", "/v1/answer-demos", weight=1),
        LoadStep("scaling_benchmarks", "GET", "/v1/scaling-benchmarks", weight=1),
    ],
}


def _weighted_pick(steps: list[LoadStep], rng: random.Random) -> LoadStep:
    total = sum(step.weight for step in steps)
    draw = rng.uniform(0, total)
    cumulative = 0.0
    for step in steps:
        cumulative += step.weight
        if draw <= cumulative:
            return step
    return steps[-1]


async def _worker(
    client: httpx.AsyncClient,
    steps: list[LoadStep],
    result: LoadResult,
    stop_at: float,
    rng: random.Random,
) -> None:
    while time.perf_counter() < stop_at:
        step = _weighted_pick(steps, rng)
        started = time.perf_counter()
        try:
            response = await client.request(
                step.method, step.path, json=step.json, params=step.params
            )
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            result.requests += 1
            result.latencies_ms.append(elapsed_ms)
            key = str(response.status_code)
            result.status_counts[key] = result.status_counts.get(key, 0) + 1
            if response.status_code >= 400:
                result.errors += 1
        except httpx.HTTPError:
            result.requests += 1
            result.errors += 1
            result.status_counts["transport_error"] = (
                result.status_counts.get("transport_error", 0) + 1
            )


async def run_load_test(
    base_url: str,
    *,
    workload: str = "mixed",
    users: int = 4,
    duration_s: float = 10.0,
    timeout_s: float = 30.0,
    steps: list[LoadStep] | None = None,
    seed: int = 0,
) -> dict[str, object]:
    """Run one bounded workload and return an aggregate result dict."""

    selected = steps or READ_ONLY_WORKLOADS.get(workload)
    if not selected:
        raise ValueError(f"unknown load workload '{workload}'")
    if users < 1 or users > 64:
        raise ValueError("users must be between 1 and 64 (bounded local load)")
    if duration_s <= 0 or duration_s > 600:
        raise ValueError("duration_s must be in (0, 600]")

    result = LoadResult(name=workload, users=users, duration_s=0.0)
    rng = random.Random(seed)
    started = time.perf_counter()
    stop_at = started + duration_s
    limits = httpx.Limits(max_connections=users + 2, max_keepalive_connections=users + 2)
    async with httpx.AsyncClient(base_url=base_url, timeout=timeout_s, limits=limits) as client:
        await asyncio.gather(
            *[_worker(client, selected, result, stop_at, rng) for _ in range(users)]
        )
    result.duration_s = time.perf_counter() - started
    summary = result.summary()
    summary["base_url"] = base_url
    summary["notes"] = [
        "Locally controlled server only; never run this against the public demo.",
        "HTTP throughput, retrieval throughput and model inference are distinct "
        "measurements and must not be combined into one number.",
    ]
    return summary


def run_load_test_sync(base_url: str, **kwargs: object) -> dict[str, object]:
    """Blocking wrapper for the CLI."""

    return asyncio.run(run_load_test(base_url, **kwargs))  # type: ignore[arg-type]


__all__ = [
    "READ_ONLY_WORKLOADS",
    "LoadResult",
    "LoadStep",
    "run_load_test",
    "run_load_test_sync",
]
