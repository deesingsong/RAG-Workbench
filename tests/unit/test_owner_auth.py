"""Unit tests for the owner WebAuthn/session stores and service.

These exercise the server-side primitives in isolation from FastAPI: credential
parsing fails closed, challenges are single-use and age-limited, sessions honour
idle and absolute lifetimes, and attempts are throttled. The HTTP ceremony is
covered end-to-end in ``tests/integration/test_laboratory_controls_api.py``.
"""

from __future__ import annotations

import pytest

from rag_workbench.auth import owner_auth
from rag_workbench.auth.owner_auth import (
    AttemptLimiter,
    ChallengeStore,
    OwnerAuthService,
    OwnerSessionStore,
    parse_credentials,
)
from rag_workbench.config import Settings


def _service(tmp_path, **overrides) -> OwnerAuthService:
    settings = Settings(
        data_dir=tmp_path,
        log_level="WARNING",
        webauthn_rp_id="owner.example.test",
        webauthn_origins=("https://owner.example.test",),
        _env_file=None,
        **overrides,
    )
    return OwnerAuthService(settings)


# -- credential parsing ------------------------------------------------------


def test_parse_credentials_rejects_garbage_and_fails_closed() -> None:
    assert parse_credentials("not json") == []
    assert parse_credentials("{}") == []
    assert parse_credentials("[]") == []
    assert parse_credentials('[{"credential_id": "a"}]') == []  # no public key
    assert parse_credentials("[1, 2, 3]") == []


def test_parse_credentials_reads_a_valid_record() -> None:
    raw = (
        '[{"credential_id": "abc", "public_key": "def", "sign_count": 3,'
        ' "label": "laptop", "transports": ["internal", 5], "backed_up": true}]'
    )
    (record,) = parse_credentials(raw)
    assert record.credential_id == "abc"
    assert record.public_key == "def"
    assert record.sign_count == 3
    assert record.transports == ("internal",)
    assert record.backed_up is True


# -- challenge store ---------------------------------------------------------


def test_challenges_are_single_use() -> None:
    store = ChallengeStore(ttl_s=100, max_pending=4)
    handle = store.create("registration", b"challenge-bytes")
    assert store.consume(handle, "registration") == b"challenge-bytes"
    assert store.consume(handle, "registration") is None


def test_challenges_reject_a_kind_mismatch() -> None:
    store = ChallengeStore(ttl_s=100, max_pending=4)
    handle = store.create("registration", b"c")
    assert store.consume(handle, "authentication") is None
    # A mismatched consume still destroys the challenge.
    assert store.consume(handle, "registration") is None


def test_challenges_expire_and_are_bounded(monkeypatch) -> None:
    clock = {"t": 1000.0}
    monkeypatch.setattr(owner_auth.time, "monotonic", lambda: clock["t"])
    store = ChallengeStore(ttl_s=10, max_pending=2)

    stale = store.create("registration", b"old")
    clock["t"] += 11  # past the TTL
    assert store.consume(stale, "registration") is None

    # The newest two survive; the third evicts the oldest.
    a = store.create("registration", b"a")
    clock["t"] += 1
    b = store.create("registration", b"b")
    clock["t"] += 1
    c = store.create("registration", b"c")
    assert store.consume(a, "registration") is None
    assert store.consume(b, "registration") == b"b"
    assert store.consume(c, "registration") == b"c"


# -- session store -----------------------------------------------------------


def test_sessions_validate_and_revoke(monkeypatch) -> None:
    monkeypatch.setattr(owner_auth.time, "monotonic", lambda: 500.0)
    store = OwnerSessionStore(idle_ttl_s=60, absolute_ttl_s=600, max_sessions=4)
    session = store.issue(credential_id="cred", label="laptop")
    assert store.validate(session.token) is not None
    assert store.revoke(session.token) is True
    assert store.validate(session.token) is None


def test_sessions_honour_idle_and_absolute_lifetimes(monkeypatch) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(owner_auth.time, "monotonic", lambda: clock["t"])
    store = OwnerSessionStore(idle_ttl_s=10, absolute_ttl_s=25, max_sessions=4)
    session = store.issue(credential_id="cred", label=None)

    clock["t"] = 9.0
    assert store.validate(session.token) is not None  # sliding refresh

    clock["t"] = 18.0  # within idle of the refresh at 9
    assert store.validate(session.token) is not None

    clock["t"] = 26.0  # past the absolute deadline despite recent activity
    assert store.validate(session.token) is None


def test_session_store_is_bounded(monkeypatch) -> None:
    monkeypatch.setattr(owner_auth.time, "monotonic", lambda: 0.0)
    store = OwnerSessionStore(idle_ttl_s=60, absolute_ttl_s=600, max_sessions=2)
    first = store.issue(credential_id="a", label=None)
    store.issue(credential_id="b", label=None)
    store.issue(credential_id="c", label=None)
    assert store.validate(first.token) is None


# -- attempt limiter ---------------------------------------------------------


def test_attempt_limiter_enforces_a_window(monkeypatch) -> None:
    clock = {"t": 0.0}
    monkeypatch.setattr(owner_auth.time, "monotonic", lambda: clock["t"])
    limiter = AttemptLimiter(max_attempts=2, window_s=10)
    assert limiter.allow() is True
    assert limiter.allow() is True
    assert limiter.allow() is False
    clock["t"] = 11.0  # the window slides
    assert limiter.allow() is True


# -- service capabilities ----------------------------------------------------


def test_service_is_unavailable_without_configuration(tmp_path) -> None:
    service = OwnerAuthService(
        Settings(data_dir=tmp_path, log_level="WARNING", _env_file=None)
    )
    assert service.available is False
    assert service.credential_count == 0
    assert service.registration_enabled is False
    summary = service.public_summary()
    assert summary["authentication_available"] is False
    assert summary["credential_configured"] is False


def test_service_requires_bootstrap_to_register(tmp_path) -> None:
    service = _service(tmp_path)  # RP configured, but bootstrap disabled
    assert service.available is True
    assert service.registration_enabled is False
    with pytest.raises(owner_auth.BootstrapDisabledError):
        service.registration_options()


def test_service_registration_ready_with_bootstrap(tmp_path) -> None:
    service = _service(
        tmp_path, webauthn_bootstrap_enabled=True, webauthn_bootstrap_token="secret"
    )
    assert service.registration_enabled is True
    handle, options = service.registration_options()
    assert handle
    assert "challenge" in options
    assert "secret" not in options


def test_service_authentication_requires_a_credential(tmp_path) -> None:
    service = _service(tmp_path)
    with pytest.raises(owner_auth.WebAuthnNotConfiguredError):
        service.authentication_options()


def test_public_summary_never_leaks_secrets(tmp_path) -> None:
    service = _service(
        tmp_path,
        webauthn_credentials='[{"credential_id": "cid", "public_key": "pub"}]',
    )
    summary = service.public_summary()
    assert summary["credential_configured"] is True
    text = repr(summary)
    assert "pub" not in text
    assert "cid" not in text
