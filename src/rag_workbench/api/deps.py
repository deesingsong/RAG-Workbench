"""FastAPI dependency providers.

Services are constructed once in ``app.state`` by the application factory and
handed to routes through these dependencies, which keeps routes free of global
state and makes tests able to build an app with isolated settings.
"""

from __future__ import annotations

from fastapi import Depends, Request

from rag_workbench.config import Settings
from rag_workbench.domain.errors import ReadOnlyModeError
from rag_workbench.evaluation.service import EvaluationService
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def require_writable(settings: Settings = Depends(get_settings)) -> None:
    """Dependency that refuses mutating operations in read-only demo mode.

    Enforcement lives in the backend, not only in the UI, so a public deployment
    cannot be modified through the documented API either.
    """

    if settings.read_only:
        raise ReadOnlyModeError(
            "this deployment is read-only; ingestion and evaluation runs are disabled"
        )


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_evaluation_service(request: Request) -> EvaluationService:
    return request.app.state.evaluation_service


def get_vector_indexer(request: Request) -> VectorIndexer:
    return request.app.state.vector_indexer
