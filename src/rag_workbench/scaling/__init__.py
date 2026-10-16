"""Scalability engineering: measured baselines, capacity planning and design.

Phase 6 makes the leap from a measured local prototype to a defensible design for
roughly ten million documents without pretending the prototype can serve that
load. The package is deliberately small and pure where it can be:

* :mod:`rag_workbench.scaling.corpus` — deterministic synthetic corpus generator;
* :mod:`rag_workbench.scaling.harness` — bounded local performance measurements;
* :mod:`rag_workbench.scaling.capacity` — deterministic capacity-planning model;
* :mod:`rag_workbench.scaling.loadtest` — bounded local API load testing;
* :mod:`rag_workbench.scaling.architecture` — current-vs-proposed design data;
* :mod:`rag_workbench.scaling.recorded` — recorded-artifact loading/validation;
* :mod:`rag_workbench.scaling.service` — the single entry point for API/CLI/UI.
"""

from __future__ import annotations

from rag_workbench.scaling.capacity import (
    IngestionAssumptions,
    StorageAssumptions,
    estimate_ingestion,
    estimate_storage,
)
from rag_workbench.scaling.corpus import CorpusConfig, generate_corpus

__all__ = [
    "CorpusConfig",
    "IngestionAssumptions",
    "StorageAssumptions",
    "estimate_ingestion",
    "estimate_storage",
    "generate_corpus",
]
