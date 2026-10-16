"""Application-facing scalability service.

Mirrors :class:`~rag_workbench.evaluation.service.EvaluationService`: the API,
CLI and Streamlit all reach scalability data through here, so the read-only demo
and the local harness cannot drift. Recorded artifacts are parsed once and cached
because, in a read-only deployment, they are immutable for the life of the
process.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import ValidationError as PydanticValidationError

from rag_workbench.config import Settings
from rag_workbench.domain.errors import ScalingBenchmarkNotFoundError, ValidationError
from rag_workbench.scaling import capacity as capacity_module
from rag_workbench.scaling.architecture import architecture_payload
from rag_workbench.scaling.recorded import (
    find_scaling_benchmark,
    load_scaling_benchmarks,
    scaling_summary,
)


class ScalingService:
    """List and read recorded scalability benchmarks; expose design data."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._recorded_dir = Path(settings.recorded_scaling_dir)
        self._cache: list[dict] | None = None

    def recorded_dir(self) -> Path:
        return self._recorded_dir

    def _recorded(self) -> list[dict]:
        if self._cache is None:
            self._cache = load_scaling_benchmarks(self._recorded_dir)
        return self._cache

    def list_benchmarks(self) -> list[dict]:
        return [scaling_summary(record) for record in self._recorded()]

    def get_benchmark(self, benchmark_id: str) -> dict:
        record = find_scaling_benchmark(self._recorded_dir, benchmark_id)
        if record is None:
            raise ScalingBenchmarkNotFoundError(
                f"scaling benchmark not found: {benchmark_id}"
            )
        return record

    # --------------------------------------------------------------- capacity

    def capacity(
        self,
        *,
        storage: dict | None = None,
        ingestion: dict | None = None,
    ) -> dict:
        """Compute storage and ingestion estimates from explicit assumptions."""

        try:
            storage_model = capacity_module.StorageAssumptions.model_validate(storage or {})
        except PydanticValidationError as exc:
            raise ValidationError(
                "invalid storage assumptions: "
                + "; ".join(
                    error["msg"] for error in exc.errors(include_url=False, include_input=False)
                )
            ) from exc
        storage_estimate = capacity_module.estimate_storage(storage_model)
        payload: dict[str, object] = {
            "storage": json.loads(storage_estimate.model_dump_json()),
            "scenarios": {
                name: json.loads(capacity_module.estimate_storage(assumptions).model_dump_json())
                for name, assumptions in capacity_module.storage_scenarios(storage_model).items()
            },
        }
        if ingestion is not None:
            try:
                ingestion_model = capacity_module.IngestionAssumptions.model_validate(ingestion)
            except PydanticValidationError as exc:
                raise ValidationError(
                    "invalid ingestion assumptions: "
                    + "; ".join(
                        error["msg"]
                        for error in exc.errors(include_url=False, include_input=False)
                    )
                ) from exc
            payload["ingestion"] = json.loads(
                capacity_module.estimate_ingestion(ingestion_model).model_dump_json()
            )
        return payload

    def scenarios(self) -> dict:
        base = capacity_module.StorageAssumptions()
        return {
            name: json.loads(
                capacity_module.estimate_storage(assumptions).model_dump_json()
            )
            for name, assumptions in capacity_module.storage_scenarios(base).items()
        }

    def architecture(self) -> dict:
        return architecture_payload()

    # ------------------------------------------------------------------ write

    def write_artifact(
        self, record: dict, *, directory: Path | None = None, filename: str | None = None
    ) -> Path:
        target_dir = Path(directory) if directory else self._recorded_dir
        target_dir.mkdir(parents=True, exist_ok=True)
        benchmark_id = str(record.get("benchmark_id") or "scaling")
        name = filename or f"{benchmark_id}.json"
        path = target_dir / name
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        return path


__all__ = ["ScalingService"]
