"""FastAPI application factory.

Create the app with :func:`create_app`. Services are built once and stored on
``app.state``; there is no import-time side effect, so tests can point the app at
an isolated database and the CLI/uvicorn use the factory form:

    uvicorn --factory rag_workbench.api.app:create_app
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from rag_workbench import __version__
from rag_workbench.api.routes import documents, health, search, vectors
from rag_workbench.config import Settings, get_settings
from rag_workbench.domain.errors import (
    DenseUnavailableError,
    DimensionMismatchError,
    FileTooLargeError,
    IngestionError,
    NotFoundError,
    ParsingError,
    RagWorkbenchError,
    RerankError,
    RerankerUnavailableError,
    SearchError,
    UnsupportedFileTypeError,
    ValidationError,
    VectorStoreError,
)
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.logging_config import configure_logging
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.vector_store import close_vector_stores
from rag_workbench.retrieval.service import SearchService

logger = logging.getLogger(__name__)

_STATUS_BY_ERROR: dict[type[RagWorkbenchError], int] = {
    ValidationError: 422,
    UnsupportedFileTypeError: 415,
    FileTooLargeError: 413,
    ParsingError: 400,
    IngestionError: 400,
    SearchError: 400,
    NotFoundError: 404,
    DimensionMismatchError: 422,
    DenseUnavailableError: 503,
    VectorStoreError: 503,
    RerankerUnavailableError: 503,
    RerankError: 400,
}


def _status_for(error: RagWorkbenchError) -> int:
    for error_type, status in _STATUS_BY_ERROR.items():
        if isinstance(error, error_type):
            return status
    return 400


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build and configure the FastAPI application."""

    settings = settings or get_settings()
    configure_logging(settings.log_level)
    settings.ensure_dirs()

    database = Database(settings.resolved_database_url)
    database.create_schema()

    app = FastAPI(
        title=settings.app_name,
        version=__version__,
        summary="Local-first RAG retrieval engineering laboratory (BM25, dense, hybrid).",
    )
    app.state.settings = settings
    app.state.database = database
    vector_indexer = VectorIndexer(database, settings)
    app.state.vector_indexer = vector_indexer
    app.state.ingestion_service = IngestionService(
        database, settings, vector_indexer=vector_indexer
    )
    app.state.search_service = SearchService(database, settings)

    @app.exception_handler(RagWorkbenchError)
    async def _handle_workbench_error(_request: Request, exc: RagWorkbenchError) -> JSONResponse:
        return JSONResponse(
            status_code=_status_for(exc),
            content={"error": exc.code, "message": exc.message},
        )

    @app.on_event("shutdown")
    def _shutdown() -> None:
        database.dispose()
        close_vector_stores()

    app.include_router(health.router)
    app.include_router(documents.router, prefix="/v1")
    app.include_router(search.router, prefix="/v1")
    app.include_router(vectors.router, prefix="/v1")
    return app


__all__ = ["create_app"]
