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
        embed = st.checkbox("Build a dense profile (Sentence Transformers embeddings)")
        embedding_model = st.text_input(
            "Embedding model (optional)", value="", disabled=not embed
        )
        submitted = st.form_submit_button("Ingest")

    if submitted and uploaded is not None:
        files = {"file": (uploaded.name, uploaded.getvalue())}
        form = {
            "collection": collection,
            "strategy": strategy,
            "max_tokens": str(int(max_tokens)),
            "overlap_tokens": str(int(overlap)),
            "embed": "true" if embed else "false",
        }
        if embed and embedding_model.strip():
            form["embedding_model"] = embedding_model.strip()
        status, body = api_post("/v1/documents", files=files, data=form)
        if status == 201:
            assert isinstance(body, dict)
            if body.get("is_duplicate"):
                kind = "duplicate"
            else:
                kind = "ingested" if body.get("state") == "completed" else body.get("state")
            st.success(f"{kind}: {body.get('source_name')} → {body.get('chunk_count')} chunks")
            for note in body.get("diagnostics", []):
                st.info(note)
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

    render_vector_status()


def render_vector_status() -> None:
    status, body = api_get("/v1/vectors/status")
    if status != 200 or not isinstance(body, dict):
        return
    statuses = body.get("statuses", [])
    if not statuses:
        return
    st.subheader("Vector index health")
    for entry in statuses:
        flag = "in sync" if entry.get("in_sync") else "OUT OF SYNC"
        st.caption(
            f"{entry.get('profile_id')} • {entry.get('collection')} • {flag} • "
            f"chunks={entry.get('chunk_count')} points={entry.get('point_count')} "
            f"missing={entry.get('missing_points')} orphans={entry.get('orphan_points')}"
        )
    if any(not entry.get("in_sync") for entry in statuses) and st.button(
        "Rebuild dense indexes now"
    ):
        status, body = api_post("/v1/vectors/rebuild", json={"all_profiles": True})
        if status == 200:
            st.success(f"Rebuilt: {body}")
        else:
            st.error(f"Rebuild failed ({status}): {body}")


def _render_hits(body: dict) -> None:
    for hit in body.get("hits", []):
        where = hit.get("source_name", "?")
        if hit.get("page_start") is not None:
            where += f" p.{hit['page_start']}"
        if hit.get("section_path"):
            where += f" | {hit['section_path']}"
        components = ""
        if hit.get("bm25_rank") is not None or hit.get("dense_rank") is not None:
            components = f"  bm25_rank={hit.get('bm25_rank')} dense_rank={hit.get('dense_rank')}"
        with st.expander(
            f"#{hit['rank']}  {hit.get('score_kind', 'bm25')}={hit['score']:.4f}  {where}"
        ):
            st.write(hit["text"])
            st.caption(
                f"chunk_id={hit['chunk_id']} document_id={hit['document_id']}"
                f"  bm25_score={hit.get('bm25_score')} dense_score={hit.get('dense_score')}"
                f"{components}"
            )


def render_search() -> None:
    st.header("Search playground")
    status, body = api_get("/v1/profiles")
    profiles = body.get("profiles", []) if status == 200 and isinstance(body, dict) else []
    if not profiles:
        st.warning("No index profiles yet — ingest a document first.")
        return

    labels = {f"{p['profile_id']} • {p['name']}": p["profile_id"] for p in profiles}
    query = st.text_input("Query", value="")
    col1, col2, col3 = st.columns(3)
    selected = col1.selectbox("Profile", list(labels.keys()))
    mode = col2.selectbox("Mode", ["bm25", "dense", "hybrid"])
    top_k = col3.slider("Top K", min_value=1, max_value=20, value=5)

    if st.button("Search") and query.strip():
        payload = {
            "query": query,
            "profile_id": labels[selected],
            "top_k": int(top_k),
            "mode": mode,
        }
        status, body = api_post("/v1/search", json=payload)
        if status != 200:
            st.error(f"Search failed ({status}): {body}")
            return
        assert isinstance(body, dict)
        st.caption(
            f"{len(body.get('hits', []))} hit(s) • mode {body.get('mode')} "
            f"• profile {body.get('profile_id')} • config {body.get('config_hash', '')[:12]} "
            f"• total {body.get('total_ms')} ms • {body.get('timings_ms')}"
        )
        _render_hits(body)


def render_compare() -> None:
    st.header("Mode comparison")
    st.caption("Runs the same query through all three modes and reports rank overlap.")
    status, body = api_get("/v1/profiles")
    profiles = body.get("profiles", []) if status == 200 and isinstance(body, dict) else []
    dense_ready = [p for p in profiles if p.get("embedding_model")]
    if not dense_ready:
        st.warning("No dense profiles yet — ingest a document with 'Build a dense profile' enabled.")
        return

    labels = {f"{p['profile_id']} • {p['name']}": p["profile_id"] for p in dense_ready}
    query = st.text_input("Query", value="termination for convenience", key="compare_query")
    selected = st.selectbox("Dense profile", list(labels.keys()), key="compare_profile")
    top_k = st.slider("Top K", min_value=1, max_value=10, value=5, key="compare_topk")

    if not st.button("Compare") or not query.strip():
        return

    results: dict[str, list[dict]] = {}
    for mode in ("bm25", "dense", "hybrid"):
        status, body = api_post(
            "/v1/search",
            json={
                "query": query,
                "profile_id": labels[selected],
                "top_k": int(top_k),
                "mode": mode,
            },
        )
        if status != 200:
            st.error(f"{mode} failed ({status}): {body}")
            return
        assert isinstance(body, dict)
        results[mode] = body.get("hits", [])

    columns = st.columns(3)
    for column, mode in zip(columns, ("bm25", "dense", "hybrid"), strict=True):
        with column:
            st.subheader(mode.upper())
            for hit in results[mode]:
                st.markdown(
                    f"**#{hit['rank']}** `{hit['score']:.4f}` — {hit['source_name']}  "
                )
                st.caption(" ".join(hit["text"].split())[:160])

    bm25_ids = [hit["chunk_id"] for hit in results["bm25"]]
    dense_ids = [hit["chunk_id"] for hit in results["dense"]]
    hybrid_ids = [hit["chunk_id"] for hit in results["hybrid"]]
    if bm25_ids and dense_ids:
        overlap = len(set(bm25_ids) & set(dense_ids))
        st.info(
            f"BM25∩dense in top-{top_k}: {overlap} chunk(s). "
            f"Hybrid kept {len(set(hybrid_ids) & (set(bm25_ids) | set(dense_ids)))} known chunk(s)."
        )


PAGES = {
    "Status": render_status,
    "Corpus Manager": render_corpus,
    "Search Playground": render_search,
    "Mode Comparison": render_compare,
}


def main() -> None:
    st.sidebar.title("RAG Workbench")
    st.sidebar.caption("BM25 • dense (Sentence Transformers) • hybrid RRF")
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.title("Retrieval Laboratory")
    PAGES[page]()


main()
