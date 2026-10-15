"""Health endpoint."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from sqlalchemy import func, select

from rag_workbench import __version__
from rag_workbench.api.deps import get_database, get_settings
from rag_workbench.api.schemas import HealthResponse
from rag_workbench.config import Settings
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Chunk, Document, IndexProfile

router = APIRouter()


@router.get("/health", response_model=HealthResponse, tags=["system"])
def health(
    settings: Settings = Depends(get_settings),
    database: Database = Depends(get_database),
) -> HealthResponse:
    document_count = chunk_count = profile_count = 0
    database_status = "ok"
    try:
        with database.session() as session:
            document_count = int(session.execute(select(func.count(Document.id))).scalar() or 0)
            chunk_count = int(session.execute(select(func.count(Chunk.id))).scalar() or 0)
            profile_count = int(session.execute(select(func.count(IndexProfile.id))).scalar() or 0)
    except Exception as exc:  # noqa: BLE001 - health must report, not crash
        database_status = f"error: {exc}"

    return HealthResponse(
        status="ok" if database_status == "ok" else "degraded",
        app_name=settings.app_name,
        version=__version__,
        database=database_status,
        document_count=document_count,
        chunk_count=chunk_count,
        profile_count=profile_count,
    )
