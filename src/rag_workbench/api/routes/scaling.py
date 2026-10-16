"""Scalability endpoints: recorded benchmarks, capacity calculator and design.

Everything here is read-only or pure arithmetic: the capacity calculator runs an
inexpensive deterministic model and mutates nothing, so it is deliberately *not*
gated by ``require_writable`` and works in the public read-only deployment. No
endpoint starts indexing or load testing.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from rag_workbench.api.deps import get_scaling_service
from rag_workbench.scaling.service import ScalingService

router = APIRouter()


class CapacityRequest(BaseModel):
    """Explicit assumptions for the capacity model.

    Both sub-objects are optional and validated against their own models; an
    invalid value returns 422 rather than a silent default.
    """

    storage: dict | None = None
    ingestion: dict | None = Field(default=None)


@router.get("/scaling-benchmarks", tags=["scaling"])
def list_scaling_benchmarks(
    service: ScalingService = Depends(get_scaling_service),
) -> dict:
    benchmarks = service.list_benchmarks()
    return {"total": len(benchmarks), "benchmarks": benchmarks}


@router.get("/scaling-benchmarks/{benchmark_id}", tags=["scaling"])
def get_scaling_benchmark(
    benchmark_id: str,
    service: ScalingService = Depends(get_scaling_service),
) -> dict:
    return service.get_benchmark(benchmark_id)


@router.get("/scaling/scenarios", tags=["scaling"])
def scaling_scenarios(
    service: ScalingService = Depends(get_scaling_service),
) -> dict:
    return {"scenarios": service.scenarios()}


@router.post("/scaling/capacity", tags=["scaling"])
def scaling_capacity(
    request: CapacityRequest,
    service: ScalingService = Depends(get_scaling_service),
) -> dict:
    return service.capacity(storage=request.storage, ingestion=request.ingestion)


@router.get("/scaling/architecture", tags=["scaling"])
def scaling_architecture(
    service: ScalingService = Depends(get_scaling_service),
) -> dict:
    return service.architecture()
