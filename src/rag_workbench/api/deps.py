"""FastAPI dependency providers.

Services are constructed once in ``app.state`` by the application factory and
handed to routes through these dependencies, which keeps routes free of global
state and makes tests able to build an app with isolated settings.
"""

from __future__ import annotations

from fastapi import Request

from rag_workbench.config import Settings
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService


def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_database(request: Request) -> Database:
    return request.app.state.database


def get_ingestion_service(request: Request) -> IngestionService:
    return request.app.state.ingestion_service


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_vector_indexer(request: Request) -> VectorIndexer:
    return request.app.state.vector_indexer
