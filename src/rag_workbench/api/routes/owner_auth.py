"""Owner Mode authentication endpoints (WebAuthn / passkey).

Four ceremony steps plus session inspection and revocation:

* ``register/options`` and ``register/verify`` — one-time bootstrap enrollment,
  authorized by the temporary bootstrap secret, never by an ordinary visitor;
* ``login/options`` and ``login/verify`` — recurring passkey authentication;
* ``session`` — the visitor's own authentication status (never a privileged
  resource, and never reporting another visitor's status); and
* ``logout`` — explicit, idempotent revocation of the caller's own session.

Authorization for privileged resources (bounded owner evaluations) lives in
:mod:`rag_workbench.api.routes.owner` and is enforced by the backend, not the UI.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request

from rag_workbench.api.deps import (
    _bearer_token,
    get_owner_auth_service,
    require_bootstrap,
)
from rag_workbench.api.schemas import (
    OwnerSessionResponse,
    WebAuthnCredentialResponse,
    WebAuthnLoginVerifyRequest,
    WebAuthnOptionsResponse,
    WebAuthnRegisterVerifyRequest,
)
from rag_workbench.auth.owner_auth import OwnerAuthService

router = APIRouter()


@router.post(
    "/owner/webauthn/register/options",
    tags=["owner-auth"],
    dependencies=[Depends(require_bootstrap)],
)
def registration_options(
    auth: OwnerAuthService = Depends(get_owner_auth_service),
) -> WebAuthnOptionsResponse:
    """Begin one-time owner passkey registration (bootstrap-authorized)."""

    handle, options = auth.registration_options()
    return WebAuthnOptionsResponse(handle=handle, options=options)


@router.post(
    "/owner/webauthn/register/verify",
    tags=["owner-auth"],
    dependencies=[Depends(require_bootstrap)],
)
def registration_verify(
    request: WebAuthnRegisterVerifyRequest,
    auth: OwnerAuthService = Depends(get_owner_auth_service),
) -> WebAuthnCredentialResponse:
    """Verify a registration, returning the public record to persist."""

    record = auth.verify_registration(
        handle=request.handle, credential=request.credential, label=request.label
    )
    return WebAuthnCredentialResponse(credential=record)


@router.post("/owner/webauthn/login/options", tags=["owner-auth"])
def authentication_options(
    auth: OwnerAuthService = Depends(get_owner_auth_service),
) -> WebAuthnOptionsResponse:
    """Begin owner passkey authentication with fresh, single-use options."""

    handle, options = auth.authentication_options()
    return WebAuthnOptionsResponse(handle=handle, options=options)


@router.post("/owner/webauthn/login/verify", tags=["owner-auth"])
def authentication_verify(
    request: WebAuthnLoginVerifyRequest,
    auth: OwnerAuthService = Depends(get_owner_auth_service),
) -> OwnerSessionResponse:
    """Verify an assertion and issue a short-lived owner session."""

    result = auth.verify_authentication(
        handle=request.handle, credential=request.credential
    )
    return OwnerSessionResponse(
        authenticated=True,
        session_token=result["session_token"],
        credential_label=result.get("credential_label"),
        issued_at=result.get("issued_at"),
        expires_at=result.get("expires_at"),
        absolute_expires_at=result.get("absolute_expires_at"),
    )


@router.get("/owner/webauthn/session", tags=["owner-auth"])
def session_status(
    request: Request,
    auth: OwnerAuthService = Depends(get_owner_auth_service),
) -> OwnerSessionResponse:
    """Report only the caller's own owner-authentication status."""

    session = auth.session_info(_bearer_token(request))
    if session is None:
        return OwnerSessionResponse(authenticated=False)
    return OwnerSessionResponse(
        authenticated=True,
        credential_label=session.get("credential_label"),
        issued_at=session.get("issued_at"),
        expires_at=session.get("expires_at"),
        absolute_expires_at=session.get("absolute_expires_at"),
    )


@router.post("/owner/webauthn/logout", tags=["owner-auth"])
def logout(
    request: Request,
    auth: OwnerAuthService = Depends(get_owner_auth_service),
) -> OwnerSessionResponse:
    """Revoke the caller's own owner session (idempotent)."""

    auth.revoke_session(_bearer_token(request))
    return OwnerSessionResponse(authenticated=False)
