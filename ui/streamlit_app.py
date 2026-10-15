"""Streamlit "Retrieval Laboratory" — a thin HTTP client of the FastAPI backend.

This module performs no retrieval or ingestion logic itself: every action is an
HTTP request to the API, so the UI and the API can never disagree about how the
system works.
"""

from __future__ import annotations

import os

import httpx
import streamlit as st

API_URL = os.environ.get("RAG_WORKBENCH_API_URL", "http://127.0.0.1:8000")
TIMEOUT = 60.0

st.set_page_config(page_title="RAG Workbench", page_icon="🔎", layout="wide")


def api_get(path: str, **params) -> tuple[int, object]:
    try:
        response = httpx.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
        return response.status_code, _decode(response)
    except httpx.HTTPError as exc:
        return 0, {"error": "connection_error", "message": str(exc)}


def api_post(path: str, **kwargs) -> tuple[int, object]:
    try:
        response = httpx.post(f"{API_URL}{path}", timeout=TIMEOUT, **kwargs)
        return response.status_code, _decode(response)
    except httpx.HTTPError as exc:
        return 0, {"error": "connection_error", "message": str(exc)}


def api_delete(path: str, **params) -> tuple[int, object]:
    try:
        response = httpx.delete(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
        return response.status_code, _decode(response)
    except httpx.HTTPError as exc:
        return 0, {"error": "connection_error", "message": str(exc)}


def _decode(response: httpx.Response) -> object:
    try:
        return response.json()
    except ValueError:
        return {"message": response.text}


def render_status() -> None:
    st.header("System status")
    status, body = api_get("/health")
    if status != 200:
        st.error(f"Backend unreachable at {API_URL}: {body}")
        st.info("Start it with:  uv run rag-workbench serve")
        return
    assert isinstance(body, dict)
    cols = st.columns(4)
    cols[0].metric("Status", body.get("status", "?"))
    cols[1].metric("Documents", body.get("document_count", 0))
    cols[2].metric("Chunks", body.get("chunk_count", 0))
    cols[3].metric("Profiles", body.get("profile_count", 0))
    st.caption(f"API: {API_URL} • database: {body.get('database')} • version {body.get('version')}")


def render_corpus() -> None:
    st.header("Corpus manager")
    with st.form("upload", clear_on_submit=True):
        uploaded = st.file_uploader("Upload a TXT, MD or text PDF", type=["txt", "md", "markdown", "pdf"])
        collection = st.text_input("Collection", value="default")
        strategy = st.selectbox("Chunking strategy", ["paragraph", "fixed_token"])
        max_tokens = st.number_input("Max tokens", min_value=16, max_value=2048, value=220)
        overlap = st.number_input("Overlap tokens", min_value=0, max_value=1024, value=40)
        submitted = st.form_submit_button("Ingest")

    if submitted and uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue())}
        form = {
            "collection": collection,
            "strategy": strategy,
            "max_tokens": str(int(max_tokens)),
            "overlap_tokens": str(int(overlap)),
        }
        status, body = api_post("/v1/documents", files=files, data=form)
        if status == 201:
            assert isinstance(body, dict)
            kind = "duplicate" if body.get("is_duplicate") else "ingested"
            st.success(f"{kind}: {body.get('source_name')} → {body.get('chunk_count')} chunks")
        else:
            st.error(f"Ingestion failed ({status}): {body}")

    status, body = api_get("/v1/documents", limit=100)
    if status == 200 and isinstance(body, dict):
        documents = body.get("documents", [])
        st.subheader(f"Indexed documents ({body.get('total', 0)})")
        if documents:
            st.dataframe(documents, use_container_width=True)
        else:
            st.caption("No documents yet.")

    status, body = api_get("/v1/profiles")
    if status == 200 and isinstance(body, dict):
        st.subheader("Index profiles")
        profiles = body.get("profiles", [])
        if profiles:
            st.dataframe(profiles, use_container_width=True)
        else:
            st.caption("No profiles yet.")


def render_search() -> None:
    st.header("Search playground (BM25)")
    status, body = api_get("/v1/profiles")
    profiles = body.get("profiles", []) if status == 200 and isinstance(body, dict) else []
    if not profiles:
        st.warning("No index profiles yet — ingest a document first.")
        return

    labels = {f"{p['profile_id']} • {p['name']}": p["profile_id"] for p in profiles}
    query = st.text_input("Query", value="")
    col1, col2 = st.columns(2)
    selected = col1.selectbox("Profile", list(labels.keys()))
    top_k = col2.slider("Top K", min_value=1, max_value=20, value=5)

    if st.button("Search") and query.strip():
        status, body = api_post(
            "/v1/search",
            json={"query": query, "profile_id": labels[selected], "top_k": int(top_k), "mode": "bm25"},
        )
        if status != 200:
            st.error(f"Search failed ({status}): {body}")
            return
        assert isinstance(body, dict)
        st.caption(
            f"{len(body.get('hits', []))} hit(s) • profile {body.get('profile_id')} "
            f"• config {body.get('config_hash', '')[:12]} • retrieval {body.get('timings_ms', {}).get('retrieval')} ms"
        )
        for hit in body.get("hits", []):
            where = hit.get("source_name", "?")
            if hit.get("page_start") is not None:
                where += f" p.{hit['page_start']}"
            if hit.get("section_path"):
                where += f" | {hit['section_path']}"
            with st.expander(f"#{hit['rank']}  score={hit['score']:.4f}  {where}"):
                st.write(hit["text"])
                st.caption(f"chunk_id={hit['chunk_id']} document_id={hit['document_id']}")


PAGES = {
    "Status": render_status,
    "Corpus Manager": render_corpus,
    "Search Playground": render_search,
}


def main() -> None:
    st.sidebar.title("RAG Workbench")
    st.sidebar.caption("Phase 0-1: ingestion + BM25 baseline")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.title("Retrieval Laboratory")
    PAGES[page]()


main()
