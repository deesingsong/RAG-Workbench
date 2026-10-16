"""Owner Mode passkey UI (WebAuthn) for the Streamlit client.

The browser performs the WebAuthn ceremony itself. Streamlit 1.66 can run trusted
same-page JavaScript through ``st.components.v2`` (no cross-origin iframe), which
is what browser WebAuthn requires. The component bridges one value -- the
authenticator response -- back to Python; Python then forwards it to the
in-process FastAPI backend over the ordinary internal HTTP connection, exactly
like every other request in this module. The backend-issued session token is kept
in ``st.session_state`` and sent only as an ``Authorization`` header: it is never
handed to page JavaScript.

Nothing here is a security boundary. The backend re-validates every session and
re-authorizes every privileged request; this module only drives the ceremony and
renders status.
"""

from __future__ import annotations

import json

import httpx
import streamlit as st

_TIMEOUT = 30.0

# Trusted, static JavaScript. It interpolates no user data; the options it acts on
# come from the backend and are read from ``component.data`` at runtime.
_PASSKEY_JS = r"""
export default function (component) {
    const { data, setTriggerValue, parentElement } = component;

    const b64urlToBuf = (value) => {
        const normalized = value.replace(/-/g, '+').replace(/_/g, '/');
        const padded = normalized + '='.repeat((4 - (normalized.length % 4)) % 4);
        const binary = atob(padded);
        const bytes = new Uint8Array(binary.length);
        for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
        return bytes.buffer;
    };
    const bufToB64url = (buffer) => {
        const bytes = new Uint8Array(buffer);
        let binary = '';
        for (let i = 0; i < bytes.length; i++) binary += String.fromCharCode(bytes[i]);
        return btoa(binary).replace(/\+/g, '-').replace(/\//g, '_').replace(/=+$/, '');
    };

    parentElement.innerHTML = '';
    const wrap = document.createElement('div');
    const button = document.createElement('button');
    button.textContent = data.button_label || 'Continue';
    button.style.cssText =
        'padding:0.45rem 1rem;border-radius:0.5rem;border:1px solid ' +
        'rgba(128,128,128,0.35);background:transparent;color:inherit;' +
        'font:inherit;cursor:pointer;';
    const status = document.createElement('div');
    status.style.cssText = 'margin-top:0.4rem;font-size:0.85rem;opacity:0.75;';
    wrap.appendChild(button);
    wrap.appendChild(status);
    parentElement.appendChild(wrap);

    const fail = (message) => {
        setTriggerValue('result', JSON.stringify({ nonce: data.nonce, error: message }));
    };

    button.onclick = async () => {
        try {
            if (!window.PublicKeyCredential || !navigator.credentials) {
                throw new Error('This browser does not support passkeys (WebAuthn).');
            }
            const options = JSON.parse(data.options);
            let payload;
            if (data.mode === 'register') {
                const publicKey = {
                    rp: options.rp,
                    user: {
                        id: b64urlToBuf(options.user.id),
                        name: options.user.name,
                        displayName: options.user.displayName,
                    },
                    challenge: b64urlToBuf(options.challenge),
                    pubKeyCredParams: options.pubKeyCredParams,
                    timeout: options.timeout,
                    attestation: options.attestation,
                    authenticatorSelection: options.authenticatorSelection,
                    excludeCredentials: (options.excludeCredentials || []).map((c) => ({
                        id: b64urlToBuf(c.id), type: c.type, transports: c.transports,
                    })),
                };
                const credential = await navigator.credentials.create({ publicKey });
                payload = {
                    nonce: data.nonce,
                    credential: {
                        id: credential.id,
                        rawId: bufToB64url(credential.rawId),
                        type: credential.type,
                        authenticatorAttachment: credential.authenticatorAttachment,
                        clientExtensionResults: credential.getClientExtensionResults(),
                        response: {
                            clientDataJSON: bufToB64url(credential.response.clientDataJSON),
                            attestationObject: bufToB64url(credential.response.attestationObject),
                            transports: credential.response.getTransports
                                ? credential.response.getTransports() : [],
                        },
                    },
                };
            } else {
                const publicKey = {
                    challenge: b64urlToBuf(options.challenge),
                    timeout: options.timeout,
                    rpId: options.rpId,
                    allowCredentials: (options.allowCredentials || []).map((c) => ({
                        id: b64urlToBuf(c.id), type: c.type, transports: c.transports,
                    })),
                    userVerification: options.userVerification,
                };
                const credential = await navigator.credentials.get({ publicKey });
                payload = {
                    nonce: data.nonce,
                    credential: {
                        id: credential.id,
                        rawId: bufToB64url(credential.rawId),
                        type: credential.type,
                        authenticatorAttachment: credential.authenticatorAttachment,
                        clientExtensionResults: credential.getClientExtensionResults(),
                        response: {
                            clientDataJSON: bufToB64url(credential.response.clientDataJSON),
                            authenticatorData: bufToB64url(credential.response.authenticatorData),
                            signature: bufToB64url(credential.response.signature),
                            userHandle: credential.response.userHandle
                                ? bufToB64url(credential.response.userHandle) : null,
                        },
                    },
                };
            }
            button.disabled = true;
            status.textContent = 'Verifying\u2026';
            setTriggerValue('result', JSON.stringify(payload));
        } catch (error) {
            fail(String((error && error.message) || error));
        }
    };
}
"""


def _build_component():
    """Register the v2 component once, tolerating an older Streamlit without v2."""

    try:
        from streamlit.components.v2 import component as _v2_component

        return (
            _v2_component("owner_passkey", js=_PASSKEY_JS, isolate_styles=True),
            None,
        )
    except Exception as exc:  # noqa: BLE001 - degrade to a clear message
        return None, str(exc)


_PASSKEY_COMPONENT, _PASSKEY_ERROR = _build_component()


def _headers(*, token: str | None = None, bootstrap: str | None = None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    if bootstrap:
        headers["X-Bootstrap-Token"] = bootstrap
    return headers


def _session_token() -> str:
    value = st.session_state.get("owner_session_token")
    return value if isinstance(value, str) else ""


def _fetch_session(api_url: str) -> dict:
    """Ask the backend for this visitor's own authentication status."""

    try:
        response = httpx.get(
            f"{api_url}/v1/owner/webauthn/session",
            headers=_headers(token=_session_token()),
            timeout=_TIMEOUT,
        )
        body = response.json()
    except (httpx.HTTPError, ValueError):
        return {"authenticated": False}
    if not isinstance(body, dict):
        return {"authenticated": False}
    if body.get("authenticated"):
        return body
    # A token that is no longer valid must not linger in the session.
    st.session_state.pop("owner_session_token", None)
    return {"authenticated": False}


def _post(api_url: str, path: str, *, json_body=None, token=None, bootstrap=None):
    try:
        response = httpx.post(
            f"{api_url}{path}",
            headers=_headers(token=token, bootstrap=bootstrap),
            json=json_body,
            timeout=_TIMEOUT,
        )
        try:
            body = response.json()
        except ValueError:
            body = {"message": response.text}
        return response.status_code, body
    except httpx.HTTPError as exc:
        return 0, {"error": "connection_error", "message": str(exc)}


def _mount(*, mode: str, options: str, handle: str, label: str, key: str):
    return _PASSKEY_COMPONENT(
        data={"mode": mode, "options": options, "nonce": handle, "button_label": label},
        key=key,
        on_result_change=lambda: None,
    )


def _read_result(result, expected_handle: str) -> dict | None:
    raw = getattr(result, "result", None)
    if not isinstance(raw, str) or not raw:
        return None
    try:
        payload = json.loads(raw)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("nonce") != expected_handle:
        return None
    return payload


def owner_session(api_url: str) -> dict:
    """Public helper: this visitor's current owner-authentication status."""

    return _fetch_session(api_url)


def render_owner_mode(api_url: str, config: dict) -> dict:
    """Render the Owner Mode section and return the current session status.

    The panel appears only when Owner Mode is actionable on this deployment
    (passkey authentication is configured, or this visitor is already signed in),
    so an ordinary read-only deployment without owner auth stays uncluttered.
    """

    session = _fetch_session(api_url)
    passkey = (config.get("passkey") or {}) if isinstance(config, dict) else {}
    if not (session.get("authenticated") or passkey.get("authentication_available")):
        return session
    with st.sidebar.expander("Owner Mode", expanded=False):
        st.caption(
            "Sign in with a WebAuthn passkey (Windows Hello, Touch ID, a security "
            "key). The private key never leaves your device; the backend only "
            "receives a signature it verifies. This does not change the "
            "deployment's permissions."
        )
        if _PASSKEY_COMPONENT is None:
            st.warning(
                "Passkey sign-in needs Streamlit with st.components.v2 "
                f"(unavailable: {_PASSKEY_ERROR})."
            )
            return session
        if not api_url:
            return session

        if session.get("authenticated"):
            _render_unlocked(api_url, session)
        elif passkey.get("credential_configured"):
            _render_login(api_url)
        elif passkey.get("registration_enabled"):
            st.caption("No owner passkey is registered yet.")
            _render_registration(api_url)
        else:
            st.caption(
                "No owner passkey is registered and bootstrap enrollment is "
                "disabled, so Owner Mode cannot be unlocked."
            )
    return session


def _render_unlocked(api_url: str, session: dict) -> None:
    st.success(f"Unlocked with {session.get('credential_label') or 'passkey'}")
    if session.get("absolute_expires_at"):
        st.caption(
            f"Session expires {session.get('expires_at')} "
            f"(at the latest {session.get('absolute_expires_at')})."
        )
    if st.button("Lock Owner Mode", key="owner_lock"):
        _post(api_url, "/v1/owner/webauthn/logout", token=_session_token())
        st.session_state.pop("owner_session_token", None)
        st.rerun()


def _render_login(api_url: str) -> None:
    pending = st.session_state.get("owner_login_pending")
    if not isinstance(pending, dict) or not pending.get("handle"):
        status, body = _post(api_url, "/v1/owner/webauthn/login/options")
        if status != 200 or not isinstance(body, dict) or not body.get("handle"):
            _render_backend_error(body, "Could not start passkey authentication")
            return
        pending = {"handle": body["handle"], "options": body.get("options", "")}
        st.session_state["owner_login_pending"] = pending

    result = _mount(
        mode="authenticate",
        options=pending["options"],
        handle=pending["handle"],
        label="Unlock with passkey",
        key="owner_passkey_login",
    )
    payload = _read_result(result, pending["handle"])
    if payload is None:
        return
    st.session_state.pop("owner_login_pending", None)
    if payload.get("error"):
        st.error(f"Passkey authentication failed: {payload['error']}")
        return
    status, body = _post(
        api_url,
        "/v1/owner/webauthn/login/verify",
        json_body={"handle": pending["handle"], "credential": payload.get("credential")},
    )
    if status == 200 and isinstance(body, dict) and body.get("session_token"):
        st.session_state["owner_session_token"] = body["session_token"]
        st.rerun()
    else:
        _render_backend_error(body, "The passkey could not be verified")


def _render_registration(api_url: str) -> None:
    st.caption(
        "One-time enrollment. Enter the bootstrap secret configured for this "
        "deployment; it authorizes registration only and is not used to sign in."
    )
    bootstrap = st.text_input(
        "Bootstrap secret", type="password", key="owner_bootstrap_token"
    )
    pending = st.session_state.get("owner_register_pending")
    if not isinstance(pending, dict) or not pending.get("handle"):
        if st.button("Begin enrollment", key="owner_begin_enrollment"):
            status, body = _post(
                api_url,
                "/v1/owner/webauthn/register/options",
                bootstrap=bootstrap.strip(),
            )
            if status == 200 and isinstance(body, dict) and body.get("handle"):
                st.session_state["owner_register_pending"] = {
                    "handle": body["handle"],
                    "options": body.get("options", ""),
                    "bootstrap": bootstrap.strip(),
                }
                st.rerun()
            else:
                _render_backend_error(body, "Enrollment could not start")
        return

    result = _mount(
        mode="register",
        options=pending["options"],
        handle=pending["handle"],
        label="Register passkey",
        key="owner_passkey_register",
    )
    payload = _read_result(result, pending["handle"])
    if payload is None:
        return
    st.session_state.pop("owner_register_pending", None)
    if payload.get("error"):
        st.error(f"Registration failed: {payload['error']}")
        return
    status, body = _post(
        api_url,
        "/v1/owner/webauthn/register/verify",
        json_body={
            "handle": pending["handle"],
            "credential": payload.get("credential"),
            "label": "owner passkey",
        },
        bootstrap=pending.get("bootstrap", ""),
    )
    if status == 200 and isinstance(body, dict) and body.get("credential"):
        st.success("Passkey verified.")
        st.caption(
            "Save this record into your deployment's trusted configuration as a "
            "one-item JSON array in RAG_WORKBENCH_WEBAUTHN_CREDENTIALS, then "
            "disable bootstrap enrollment. It contains only public data."
        )
        st.code(json.dumps([body["credential"]], indent=2), language="json")
    else:
        _render_backend_error(body, "The registration could not be verified")


def _render_backend_error(body: object, prefix: str) -> None:
    if isinstance(body, dict):
        detail = body.get("message") or body.get("error") or body
        st.error(f"{prefix}: {detail}")
    else:
        st.error(f"{prefix}: {body}")
