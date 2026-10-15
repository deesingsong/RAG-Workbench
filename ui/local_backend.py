"""Self-contained backend for single-process deployments.

Locally the UI is a thin HTTP client of a separately-run API (README §4). A
single-process host such as Streamlit Community Cloud cannot run a second
process, so :func:`resolve_api_url` starts the *same* FastAPI app with uvicorn on
a loopback port in a background thread and returns its URL.

Nothing about retrieval moves into the UI: the Streamlit code still speaks HTTP
to ``/v1/...`` exactly as it does against a separately-run server. The only
difference is who owns the server process. An explicit ``RAG_WORKBENCH_API_URL``
always wins, so local development is unchanged.
"""

from __future__ import annotations

import os
import socket
import sys
import threading
import time
from pathlib import Path

import httpx
import streamlit as st

# Make the package importable even if a host installs only third-party
# requirements and does not install the project itself.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC = _REPO_ROOT / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765
_SAMPLE_CONTRACT = _REPO_ROOT / "tests" / "fixtures" / "sample_contract.txt"


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.25)
        return sock.connect_ex((host, port)) == 0


def _health_ok(base_url: str) -> bool:
    try:
        return httpx.get(f"{base_url}/health", timeout=1.0).status_code == 200
    except httpx.HTTPError:
        return False


def _launch(host: str, port: int) -> None:
    import uvicorn

    from rag_workbench.api.app import create_app

    config = uvicorn.Config(create_app(), host=host, port=port, log_level="warning")
    threading.Thread(
        target=uvicorn.Server(config).run, name="rag-workbench-api", daemon=True
    ).start()


@st.cache_resource(show_spinner="Starting the in-process API …")
def _inproc_api_url(host: str, port: int) -> str:
    base_url = f"http://{host}:{port}"
    if not _port_open(host, port):
        _launch(host, port)
    deadline = time.monotonic() + 60.0
    while time.monotonic() < deadline:
        if _health_ok(base_url):
            return base_url
        time.sleep(0.25)
    raise RuntimeError(f"in-process API did not become healthy at {base_url}")


def resolve_api_url() -> str:
    """Return the API base URL, starting the in-process backend when needed."""

    configured = os.environ.get("RAG_WORKBENCH_API_URL", "").strip()
    if configured:
        return configured
    host = os.environ.get("RAG_WORKBENCH_INPROC_HOST", DEFAULT_HOST)
    port = int(os.environ.get("RAG_WORKBENCH_INPROC_PORT", str(DEFAULT_PORT)))
    return _inproc_api_url(host, port)


def seed_sample_if_empty(api_url: str) -> str | None:
    """Ingest the bundled sample contract once, on a fresh (empty) index.

    A deployed instance starts with empty ephemeral storage; this gives it
    something to search *and* compare. It follows the public API like any other
    client and only acts when the index is empty. Two profiles of the same
    document are seeded so the Chunking Laboratory has something to contrast:
    a `paragraph` profile and a `clause_aware` profile. Disable with
    ``RAG_WORKBENCH_AUTOSEED=0``.
    """

    if os.environ.get("RAG_WORKBENCH_AUTOSEED", "1").strip().lower() in {"0", "false", "no"}:
        return None
    if not _SAMPLE_CONTRACT.is_file():
        return None
    try:
        health = httpx.get(f"{api_url}/health", timeout=10.0)
        if health.status_code != 200 or health.json().get("document_count", 0) > 0:
            return None
        seeded: list[str] = []
        for strategy in ("paragraph", "clause_aware"):
            response = httpx.post(
                f"{api_url}/v1/documents",
                files={"file": (_SAMPLE_CONTRACT.name, _SAMPLE_CONTRACT.read_bytes())},
                data={"collection": "contracts", "strategy": strategy},
                timeout=120.0,
            )
            if response.status_code == 201:
                chunks = response.json().get("chunk_count")
                seeded.append(f"{strategy} ({chunks} chunks)")
    except httpx.HTTPError as exc:  # pragma: no cover - deployment convenience
        return f"Sample seed skipped: {exc}"
    if seeded:
        return "Seeded the bundled sample contract: " + ", ".join(seeded) + "."
    return None
