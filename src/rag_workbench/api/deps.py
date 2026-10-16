"""FastAPI dependency providers.

Services are constructed once in ``app.state`` by the application factory and
handed to routes through these dependencies, which keeps routes free of global
state and makes tests able to build an app with isolated settings.
"""

from __future__ import annotations

import hmac

from fastapi import Depends, Request

from rag_workbench.answering.pipeline import AnswerService
from rag_workbench.auth.owner_auth import OwnerAuthService
from rag_workbench.config import Settings
from rag_workbench.domain.errors import (
    BootstrapDisabledError,
    GeneratorUnavailableError,
    OwnerAuthError,
    OwnerEvaluationsDisabledError,
    ReadOnlyModeError,
)
from rag_workbench.evaluation.owner import OwnerEvaluationService
from rag_workbench.evaluation.service import EvaluationService
from rag_workbench.ingestion.pipeline import IngestionService
from rag_workbench.ingestion.vector_index import VectorIndexer
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import SearchService
from rag_workbench.scaling.service import ScalingService


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


def get_owner_evaluation_service(request: Request) -> OwnerEvaluationService:
    return request.app.state.owner_evaluation_service


def get_vector_indexer(request: Request) -> VectorIndexer:
    return request.app.state.vector_indexer


def get_answer_service(request: Request) -> AnswerService:
    return request.app.state.answer_service


def get_scaling_service(request: Request) -> ScalingService:
    return request.app.state.scaling_service


def require_generation(settings: Settings = Depends(get_settings)) -> None:
    """Refuse live generation unless it is deliberately enabled and permitted.

    Two independent gates, both enforced in the backend:

    * a read-only deployment never runs generation (403), so the public demo
      cannot be turned into a model endpoint; and
    * live generation is opt-in (``RAG_WORKBENCH_GENERATION_ENABLED``); when it is
      off the endpoint reports 503 with the exact variable to set, and every other
      feature keeps working.
    """

    if settings.read_only:
        raise ReadOnlyModeError(
            "this deployment is read-only; live answer generation is disabled. "
            "Recorded answer demonstrations are available instead."
        )
    if not settings.generation_enabled:
        raise GeneratorUnavailableError(
            "live answer generation is disabled; set "
            "RAG_WORKBENCH_GENERATION_ENABLED=true and run Ollama to enable it"
        )


def get_owner_auth_service(request: Request) -> OwnerAuthService:
    return request.app.state.owner_auth_service


#: Header carrying the backend-issued owner session token. The token is sent by
#: the Streamlit Python process (never by page JavaScript) over the internal HTTP
#: connection and is kept out of URLs so it cannot land in access logs or history.
_OWNER_SESSION_HEADER = "Authorization"

#: Header carrying the one-time bootstrap enrollment secret. Distinct from the
#: session bearer: it authorizes first-time passkey registration only.
_BOOTSTRAP_HEADER = "X-Bootstrap-Token"


def _bearer_token(request: Request) -> str:
    raw = request.headers.get(_OWNER_SESSION_HEADER, "")
    if raw.lower().startswith("bearer "):
        return raw[7:].strip()
    return ""


def require_owner(
    request: Request,
    settings: Settings = Depends(get_settings),
    auth: OwnerAuthService = Depends(get_owner_auth_service),
) -> None:
    """Authorize a bounded owner evaluation, independently of any UI state.

    Two backends-enforced gates, in order:

    * the deployment must explicitly enable hosted owner evaluations (default
      disabled) as an emergency kill switch, otherwise 403; and
    * a valid owner session (issued only after a verified passkey assertion) must
      be presented, otherwise 401.
    """

    if not settings.owner_evaluations_enabled:
        raise OwnerEvaluationsDisabledError(
            "hosted owner evaluations are disabled; set "
            "RAG_WORKBENCH_OWNER_EVALUATIONS_ENABLED=true to enable them"
        )
    if auth.session_info(_bearer_token(request)) is None:
        raise OwnerAuthError("a valid owner session is required")


def require_bootstrap(
    request: Request,
    settings: Settings = Depends(get_settings),
) -> None:
    """Authorize one-time owner passkey enrollment with the bootstrap secret.

    Fails closed when bootstrap is disabled or has no secret, and compares the
    supplied secret in constant time. The secret is never echoed back.
    """

    if not settings.webauthn_bootstrap_ready:
        raise BootstrapDisabledError(
            "owner passkey bootstrap enrollment is disabled; set "
            "RAG_WORKBENCH_WEBAUTHN_BOOTSTRAP_TOKEN to enroll once"
        )
    expected = settings.webauthn_bootstrap_token or ""
    supplied = request.headers.get(_BOOTSTRAP_HEADER, "")
    if not supplied or not hmac.compare_digest(supplied, expected):
        raise BootstrapDisabledError("invalid or missing bootstrap secret")
