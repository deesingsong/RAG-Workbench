"""API tests for Laboratory Controls (availability, permission, owner mode).

Covers the backend half of the three-state model -- *available* vs *permitted* --
plus the passkey-gated owner boundary. The per-session toggle is Streamlit state
and is deliberately absent from these responses: there is no global switch a
request could flip. Owner authentication is never a frontend boolean: the
privileged endpoint demands a server-validated session issued by a verified
WebAuthn assertion.
"""

from __future__ import annotations

import json

from fastapi.testclient import TestClient

from rag_workbench.answering.ollama import probe_generation
from rag_workbench.api.app import create_app
from rag_workbench.config import Settings
from tests.soft_authenticator import SoftAuthenticator

RP_ID = "owner.example.test"
ORIGIN = "https://owner.example.test"
BOOTSTRAP = "bootstrap-secret-not-a-real-secret"
BOOTSTRAP_HEADER = {"X-Bootstrap-Token": BOOTSTRAP}


def _settings(tmp_path, *, credentials=None, **overrides) -> Settings:
    return Settings(
        data_dir=tmp_path,
        log_level="WARNING",
        webauthn_rp_id=RP_ID,
        webauthn_origins=(ORIGIN,),
        webauthn_credentials=json.dumps(credentials or []),
        _env_file=None,
        **overrides,
    )


def _client(tmp_path, *, credentials=None, **overrides) -> TestClient:
    return TestClient(create_app(_settings(tmp_path, credentials=credentials, **overrides)))


def _register(tmp_path) -> tuple[dict, SoftAuthenticator]:
    """Enroll one passkey via the bootstrap flow and return its public record."""

    authenticator = SoftAuthenticator(rp_id=RP_ID, origin=ORIGIN)
    settings = _settings(
        tmp_path, webauthn_bootstrap_enabled=True, webauthn_bootstrap_token=BOOTSTRAP
    )
    with TestClient(create_app(settings)) as client:
        options = client.post(
            "/v1/owner/webauthn/register/options", headers=BOOTSTRAP_HEADER
        ).json()
        credential = authenticator.register(options["options"])
        verified = client.post(
            "/v1/owner/webauthn/register/verify",
            headers=BOOTSTRAP_HEADER,
            json={
                "handle": options["handle"],
                "credential": credential,
                "label": "test passkey",
            },
        )
        assert verified.status_code == 200, verified.text
        record = verified.json()["credential"]
    return record, authenticator


def _login(client: TestClient, authenticator: SoftAuthenticator) -> str:
    options = client.post("/v1/owner/webauthn/login/options").json()
    assertion = authenticator.authenticate(options["options"])
    response = client.post(
        "/v1/owner/webauthn/login/verify",
        json={"handle": options["handle"], "credential": assertion},
    )
    assert response.status_code == 200, response.text
    return response.json()["session_token"]


def test_config_reports_independent_capabilities(tmp_path) -> None:
    with _client(tmp_path) as client:
        cfg = client.get("/v1/config").json()
    lab = cfg["laboratory"]
    # Locally writable: evaluation and ingestion are permitted.
    assert lab["evaluation"]["permitted"] is True
    assert lab["ingestion"]["permitted"] is True
    # Generation is off by default and explains the required configuration.
    assert lab["generation"]["available"] is False
    assert lab["generation"]["permitted"] is False
    assert "RAG_WORKBENCH_GENERATION_ENABLED" in lab["generation"]["reason"]
    # Owner mode has no credential configured, so it is unavailable.
    assert lab["owner_evaluation"]["permitted"] is False
    assert lab["owner_evaluation"]["available"] is False
    # Session-enabled is a frontend concept and must never appear in the payload.
    assert "session_enabled" not in cfg
    assert "session" not in {key.lower() for key in lab["evaluation"]}


def test_read_only_disables_evaluation_and_ingestion_permission(tmp_path) -> None:
    with _client(tmp_path, read_only=True) as client:
        cfg = client.get("/v1/config").json()
    assert cfg["read_only"] is True
    assert cfg["laboratory"]["evaluation"]["permitted"] is False
    assert cfg["laboratory"]["ingestion"]["permitted"] is False
    assert "public visitors" in cfg["laboratory"]["evaluation"]["reason"]


def test_generation_probe_disabled_and_unreachable(tmp_path) -> None:
    disabled = Settings(data_dir=tmp_path, log_level="WARNING", _env_file=None)
    assert probe_generation(disabled) == {
        "available": False,
        "detail": "live generation is disabled",
    }

    # Enabled but no Ollama listening: a probe must report unavailable, not raise.
    unreachable = Settings(
        data_dir=tmp_path,
        log_level="WARNING",
        generation_enabled=True,
        ollama_base_url="http://127.0.0.1:1",
        _env_file=None,
    )
    result = probe_generation(unreachable, timeout_s=1.0)
    assert result["available"] is False
    assert result["detail"]


def test_anonymous_mutations_refused_in_read_only(tmp_path) -> None:
    with _client(tmp_path, read_only=True) as client:
        payload = {"dataset": "legal-contracts", "profile_id": "p-does-not-exist"}
        assert client.post("/v1/evaluations", json=payload).status_code == 403
        assert client.post("/v1/owner/evaluations", json={}).status_code == 403
        assert (
            client.post(
                "/v1/documents",
                files={"file": ("a.txt", b"hello world")},
                data={"collection": "c"},
            ).status_code
            == 403
        )
        assert client.post("/v1/answer", json={"question": "q"}).status_code == 403


# -- owner passkey authentication -------------------------------------------


def test_registration_requires_bootstrap(tmp_path) -> None:
    with _client(tmp_path) as client:
        # Bootstrap is disabled by default, so enrollment is refused outright.
        assert client.post("/v1/owner/webauthn/register/options").status_code == 403
    with _client(
        tmp_path, webauthn_bootstrap_enabled=True, webauthn_bootstrap_token=BOOTSTRAP
    ) as client:
        # Enabled, but a wrong secret is refused and never echoed.
        wrong = client.post(
            "/v1/owner/webauthn/register/options",
            headers={"X-Bootstrap-Token": "wrong"},
        )
        assert wrong.status_code == 403
        assert BOOTSTRAP not in wrong.text


def test_full_registration_and_authentication_round_trip(tmp_path) -> None:
    record, authenticator = _register(tmp_path)
    assert record["credential_id"]
    assert record["public_key"]
    assert record["label"] == "test passkey"

    with _client(tmp_path, credentials=[record]) as client:
        assert client.get("/v1/owner/webauthn/session").json()["authenticated"] is False

        token = _login(client, authenticator)
        assert token

        status = client.get(
            "/v1/owner/webauthn/session", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert status["authenticated"] is True
        assert status["credential_label"] == "test passkey"
        assert "session_token" not in status or status["session_token"] is None

        revoked = client.post(
            "/v1/owner/webauthn/logout", headers={"Authorization": f"Bearer {token}"}
        ).json()
        assert revoked["authenticated"] is False
        # The same token no longer authenticates.
        assert (
            client.get(
                "/v1/owner/webauthn/session",
                headers={"Authorization": f"Bearer {token}"},
            ).json()["authenticated"]
            is False
        )


def test_authentication_requires_a_configured_credential(tmp_path) -> None:
    with _client(tmp_path) as client:
        response = client.post("/v1/owner/webauthn/login/options")
    assert response.status_code == 403
    assert response.json()["error"] == "webauthn_not_configured"


def test_authentication_rejects_a_challenge_replay(tmp_path) -> None:
    record, authenticator = _register(tmp_path)
    with _client(tmp_path, credentials=[record]) as client:
        options = client.post("/v1/owner/webauthn/login/options").json()
        assertion = authenticator.authenticate(options["options"])
        payload = {"handle": options["handle"], "credential": assertion}
        assert client.post("/v1/owner/webauthn/login/verify", json=payload).status_code == 200
        # The same single-use challenge cannot be replayed.
        replay = client.post("/v1/owner/webauthn/login/verify", json=payload)
        assert replay.status_code == 400
        assert replay.json()["error"] == "webauthn_verification_failed"


def test_authentication_rejects_a_wrong_origin(tmp_path) -> None:
    record, _ = _register(tmp_path)
    attacker = SoftAuthenticator(rp_id=RP_ID, origin="https://evil.example.test")
    with _client(tmp_path, credentials=[record]) as client:
        options = client.post("/v1/owner/webauthn/login/options").json()
        assertion = attacker.authenticate(options["options"])
        response = client.post(
            "/v1/owner/webauthn/login/verify",
            json={"handle": options["handle"], "credential": assertion},
        )
    assert response.status_code == 400
    assert response.json()["error"] == "webauthn_verification_failed"


# -- bounded owner evaluation (the privileged resource) ----------------------


def test_owner_disabled_refuses_even_a_valid_session(tmp_path) -> None:
    record, authenticator = _register(tmp_path)
    with _client(tmp_path, credentials=[record]) as client:
        token = _login(client, authenticator)
        response = client.post(
            "/v1/owner/evaluations",
            json={},
            headers={"Authorization": f"Bearer {token}"},
        )
    assert response.status_code == 403
    assert response.json()["error"] == "owner_evaluations_disabled"


def test_owner_enabled_requires_a_valid_session(tmp_path) -> None:
    record, _ = _register(tmp_path)
    with _client(
        tmp_path, credentials=[record], owner_evaluations_enabled=True
    ) as client:
        assert client.post("/v1/owner/evaluations", json={}).status_code == 401
        bogus = client.post(
            "/v1/owner/evaluations",
            json={},
            headers={"Authorization": "Bearer not-a-real-token"},
        )
        assert bogus.status_code == 401
        assert bogus.json()["error"] == "owner_unauthorized"


def test_owner_evaluation_runs_bounded_bm25_with_a_session(tmp_path) -> None:
    record, authenticator = _register(tmp_path)
    with _client(
        tmp_path,
        credentials=[record],
        owner_evaluations_enabled=True,
        owner_eval_max_queries=20,
        owner_eval_max_top_k=10,
    ) as client:
        header = {"Authorization": f"Bearer {_login(client, authenticator)}"}
        capabilities = client.get("/v1/owner/capabilities", headers=header)
        assert capabilities.status_code == 200
        assert capabilities.json()["policy"]["modes"] == ["bm25"]

        response = client.post(
            "/v1/owner/evaluations", json={"max_queries": 5}, headers=header
        )
    assert response.status_code == 200
    body = response.json()
    assert body["experiment_id"]
    config = body["config"]
    assert config["mode"] == "bm25"
    assert config["rerank"] is False
    assert config["max_queries"] == 5
    assert config["max_runtime_s"] > 0
    assert body["counts"]["queries_run"] == 5


def test_owner_evaluation_rejects_inputs_over_the_bounds(tmp_path) -> None:
    record, authenticator = _register(tmp_path)
    with _client(
        tmp_path,
        credentials=[record],
        owner_evaluations_enabled=True,
        owner_eval_max_queries=5,
    ) as client:
        header = {"Authorization": f"Bearer {_login(client, authenticator)}"}

        over_queries = client.post(
            "/v1/owner/evaluations", json={"max_queries": 6}, headers=header
        )
        assert over_queries.status_code == 422
        assert over_queries.json()["error"] == "owner_limit_exceeded"

        schema_over = client.post(
            "/v1/owner/evaluations", json={"max_queries": 999}, headers=header
        )
        assert schema_over.status_code == 422

        disallowed = client.post(
            "/v1/owner/evaluations", json={"dataset": "scifact"}, headers=header
        )
        assert disallowed.status_code == 422
        assert disallowed.json()["error"] == "owner_limit_exceeded"


def test_owner_evaluation_works_while_deployment_is_read_only(tmp_path) -> None:
    record, authenticator = _register(tmp_path)
    with _client(
        tmp_path,
        credentials=[record],
        read_only=True,
        owner_evaluations_enabled=True,
    ) as client:
        header = {"Authorization": f"Bearer {_login(client, authenticator)}"}
        # Anonymous still refused ...
        assert client.post("/v1/evaluations", json={}).status_code == 403
        # ... while the authenticated owner may run a bounded evaluation.
        response = client.post(
            "/v1/owner/evaluations", json={"max_queries": 3}, headers=header
        )
        assert response.status_code == 200
        # ... and the deployment's read-only permission is unchanged.
        assert client.get("/v1/config").json()["read_only"] is True


def test_requests_never_mutate_deployment_permissions(tmp_path) -> None:
    """No request flips a process-wide toggle: read-only stays read-only."""

    with _client(tmp_path, read_only=True) as client:
        before = client.get("/v1/config").json()
        client.post("/v1/evaluations", json={"dataset": "x", "profile_id": "y"})
        client.post("/v1/owner/evaluations", json={})
        client.post("/v1/documents", files={"file": ("a.txt", b"x")}, data={"collection": "c"})
        after = client.get("/v1/config").json()
    assert before["read_only"] is after["read_only"] is True
    assert before["laboratory"] == after["laboratory"]


def test_config_never_exposes_owner_secrets(tmp_path) -> None:
    record, _ = _register(tmp_path)
    with _client(
        tmp_path, credentials=[record], owner_evaluations_enabled=True
    ) as client:
        config = client.get("/v1/config")
    assert config.status_code == 200
    # The bootstrap secret, the public key and any key material are never exposed.
    assert BOOTSTRAP not in config.text
    assert record["public_key"] not in config.text
    assert "session_token" not in config.text
    assert config.json()["passkey"]["credential_configured"] is True
    assert "owner_token_configured" not in config.json()


def test_public_config_omits_infrastructure_details(tmp_path) -> None:
    """The public config must not disclose filesystem layout or endpoints."""

    sensitive = {
        "repo_root",
        "bundled_datasets_dir",
        "recorded_experiments_dir",
        "recorded_answers_dir",
        "recorded_scaling_dir",
        "ollama_base_url",
        "ollama_model",
        "generation_available_detail",
    }
    with _client(tmp_path) as client:
        cfg = client.get("/v1/config").json()
        # A writable (operator/local) deployment can still read the locations...
        diag = client.get("/v1/config/diagnostics")
        assert diag.status_code == 200
        assert "repo_root" in diag.json()
    # ...but the public payload exposes capabilities, never paths or endpoints.
    assert sensitive.isdisjoint(cfg)
    assert cfg["bundled_datasets_exists"] in {True, False}
    assert cfg["recorded_scaling_exists"] in {True, False}


def test_config_diagnostics_refused_on_read_only_deployment(tmp_path) -> None:
    """The public read-only demo cannot read resolved paths or endpoints."""

    with _client(tmp_path, read_only=True) as client:
        response = client.get("/v1/config/diagnostics")
    assert response.status_code == 403
    assert response.json()["error"] == "read_only_mode"
