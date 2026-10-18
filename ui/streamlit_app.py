"""Streamlit "Retrieval Laboratory" — a thin HTTP client of the FastAPI backend.

This module performs no retrieval, ingestion or evaluation logic itself: every
action is an HTTP request to the API, so the UI and the API can never disagree
about how the system works. Pages: search and mode comparison, the Chunking,
Reranking, Answer, Evaluation, Performance and Scalability Laboratories, a Failure
Explorer and the system/corpus views.
"""

from __future__ import annotations

import importlib

import httpx
import streamlit as st

# Import the helper module as a whole rather than ``from local_backend import
# name``. Streamlit re-executes this *main* script when a file changes but keeps
# already-imported modules in ``sys.modules``; on a mid-deploy hot reload the
# running process can therefore still hold an older ``local_backend``. Binding
# through the module object (and resolving ``server_is_read_only`` with
# ``getattr``) means a stale module degrades the read-only probe instead of
# crashing the whole app. A full redeploy/restart loads the current version.
_backend = importlib.import_module("local_backend")
resolve_api_url = _backend.resolve_api_url
seed_sample_if_empty = _backend.seed_sample_if_empty
_owner = importlib.import_module("owner_passkey")


def server_is_read_only(api_url: str) -> bool:
    """Probe the deployment's read-only flag, tolerating a stale module."""

    probe = getattr(_backend, "server_is_read_only", None)
    return bool(probe(api_url)) if probe is not None else False


def on_disk_version() -> str:
    """Read the package version from the source tree, tolerating a stale module."""

    probe = getattr(_backend, "on_disk_version", None)
    return probe() if probe is not None else "unknown"


def on_disk_revision() -> tuple[str | None, str]:
    """Resolve the source-tree deployment revision, tolerating a stale module."""

    probe = getattr(_backend, "on_disk_revision", None)
    if probe is None:
        return None, "unavailable"
    result = probe()
    return result if isinstance(result, tuple) else (None, "unavailable")

# Replaced at startup by resolve_api_url(). The default is the separately-run
# local API; a single-process host starts the API in-process instead.
API_URL = "http://127.0.0.1:8000"
READ_ONLY = False
# Session-enabled laboratory controls. These are per-browser-session preferences
# only; the backend remains the authority and enforces permissions regardless of
# these flags. They are recomputed on every rerun from st.session_state.
LAB_EVAL_ENABLED = False
LAB_GENERATION_ENABLED = False
LAB_INGESTION_ENABLED = False
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


def api_text(path: str, **params) -> tuple[int, str]:
    try:
        response = httpx.get(f"{API_URL}{path}", params=params, timeout=TIMEOUT)
        return response.status_code, response.text
    except httpx.HTTPError as exc:
        return 0, str(exc)


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


def _laboratory_control(
    display: str,
    capability: dict,
    state_key: str,
    *,
    available_label: str,
    unavailable_label: str,
) -> None:
    """One session toggle + status line (available / permitted / session-enabled)."""

    available = bool(capability.get("available"))
    permitted = bool(capability.get("permitted"))
    if not permitted:
        # Never let a stale session flag imply a permission the backend refuses.
        st.session_state[state_key] = False
    st.checkbox(f"Enable {display.lower()}", key=state_key, disabled=not permitted)
    enabled = permitted and bool(st.session_state.get(state_key, False))
    word = available_label if available else unavailable_label
    st.caption(f"{display}: {word} — {'On' if enabled else 'Off'}")
    reason = capability.get("reason")
    if reason and (not permitted or not available):
        st.caption(f"⚠ {reason}")


def render_laboratory_controls(cfg: dict) -> None:
    """Session-only Laboratory Controls (never global deployment configuration)."""

    laboratory = cfg.get("laboratory") or {} if isinstance(cfg, dict) else {}
    with st.sidebar.expander("Laboratory Controls", expanded=False):
        st.caption(
            "Per-session switches. They change only what this browser session runs, "
            "never the deployment's configuration or permissions."
        )
        _laboratory_control(
            "Evaluation runs",
            laboratory.get("evaluation", {}),
            "lab_eval_enabled",
            available_label="Available",
            unavailable_label="Unavailable",
        )
        _laboratory_control(
            "Live generation",
            laboratory.get("generation", {}),
            "lab_generation_enabled",
            available_label="Ollama available",
            unavailable_label="Ollama unavailable",
        )
        _laboratory_control(
            "Document ingestion",
            laboratory.get("ingestion", {}),
            "lab_ingestion_enabled",
            available_label="Available",
            unavailable_label="Unavailable",
        )
        owner = laboratory.get("owner_evaluation", {})
        if owner:
            passkey = (cfg.get("passkey") or {}) if isinstance(cfg, dict) else {}
            if not passkey.get("authentication_available"):
                note = "passkey sign-in not configured"
            elif not passkey.get("credential_configured"):
                note = "no passkey registered yet"
            else:
                note = "sign in under Owner Mode"
            st.caption(
                "Owner evaluations: "
                + ("permitted" if owner.get("permitted") else "not permitted")
                + f" — {note}"
            )


def render_corpus() -> None:
    st.header("Corpus manager")
    if READ_ONLY:
        st.info(
            "Read-only demo: document ingestion and index rebuilds are disabled. "
            "The bundled sample corpus is seeded by the backend itself at startup."
        )
    elif not LAB_INGESTION_ENABLED:
        st.info(
            "Document ingestion is available but switched off for this session. "
            "Enable it in the sidebar under Laboratory Controls."
        )
    else:
        with st.form("upload", clear_on_submit=True):
            uploaded = st.file_uploader("Upload a TXT, MD or text PDF", type=["txt", "md", "markdown", "pdf"])
            collection = st.text_input("Collection", value="default")
            strategy = st.selectbox(
                "Chunking strategy",
                ["paragraph", "fixed_token", "clause_aware"],
                help="clause_aware preserves legal clause boundaries and section hierarchy",
            )
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
    if any(not entry.get("in_sync") for entry in statuses):
        if READ_ONLY:
            st.warning(
                "One or more dense indexes are out of sync; rebuilds are disabled "
                "in read-only mode."
            )
        elif st.button("Rebuild dense indexes now"):
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
    col1, col2, col3, col4 = st.columns(4)
    selected = col1.selectbox("Profile", list(labels.keys()))
    mode = col2.selectbox("Mode", ["bm25", "dense", "hybrid"])
    query_mode = col3.selectbox(
        "Query mode",
        ["and", "or"],
        help="Lexical term combination: 'and' needs every term; 'or' matches any term.",
    )
    top_k = col4.slider("Top K", min_value=1, max_value=20, value=5)

    if st.button("Search") and query.strip():
        payload = {
            "query": query,
            "profile_id": labels[selected],
            "top_k": int(top_k),
            "mode": mode,
            "query_mode": query_mode,
        }
        status, body = api_post("/v1/search", json=payload)
        if status != 200:
            st.error(f"Search failed ({status}): {body}")
            return
        assert isinstance(body, dict)
        st.caption(
            f"{len(body.get('hits', []))} hit(s) • mode {body.get('mode')} "
            f"• query_mode {body.get('query_mode')} "
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


def _profile_options() -> list[dict]:
    status, body = api_get("/v1/profiles")
    if status == 200 and isinstance(body, dict):
        return body.get("profiles", [])
    return []


def _profile_labels(profiles: list[dict]) -> dict[str, str]:
    return {
        f"{p['profile_id']} • {p['name']} ({p.get('chunker_strategy', '?')})": p["profile_id"]
        for p in profiles
    }


def _render_chunk_profile(container, profile_id: str, label: str, limit: int) -> None:
    with container:
        st.subheader(label)
        status, body = api_get(f"/v1/profiles/{profile_id}/chunks", limit=limit)
        if status != 200 or not isinstance(body, dict):
            st.error(f"Failed to load chunks ({status}): {body}")
            return
        chunks = body.get("chunks", [])
        st.metric("Chunks", body.get("total", 0))
        if chunks:
            tokens = [chunk["token_count"] for chunk in chunks]
            st.caption(f"avg tokens {sum(tokens) / len(tokens):.1f} • max {max(tokens)}")
        else:
            st.caption("No chunks for this profile.")
        for chunk in chunks:
            head = f"#{chunk['chunk_order']} • {chunk['token_count']} tok"
            if chunk.get("section_number"):
                head += f" • {chunk['section_number']}"
            if chunk.get("section_title"):
                head += f" {chunk['section_title']}"
            with st.expander(head):
                for note in chunk.get("diagnostics", []):
                    st.info(note)
                if chunk.get("section_path"):
                    st.caption(f"section_path: {chunk['section_path']}")
                if chunk.get("parent_clause"):
                    st.caption(f"parent_clause: {chunk['parent_clause']}")
                if chunk.get("page_start") is not None:
                    st.caption(f"page: {chunk['page_start']}")
                st.caption(
                    f"chunk_id={chunk['chunk_id']} chars {chunk.get('char_start')}–{chunk.get('char_end')}"
                )
                st.write(chunk["text"])


def render_chunking_lab() -> None:
    st.header("Chunking Laboratory")
    st.caption(
        "Inspect stored chunks, clause boundaries and provenance per index profile. "
        "Compare two profiles side by side to see how chunking changes."
    )
    profiles = _profile_options()
    if not profiles:
        st.warning("No index profiles yet — ingest a document first.")
        return

    labels = _profile_labels(profiles)
    keys = list(labels.keys())
    col1, col2, col3 = st.columns(3)
    left = col1.selectbox("Profile A", keys, index=0, key="chunk_left")
    right_choices = ["(none)"] + keys
    # Default to a *different* profile so the comparison is meaningful on first use.
    right_index = 2 if len(keys) >= 2 else 0
    right = col2.selectbox(
        "Profile B", right_choices, index=right_index, key="chunk_right"
    )
    limit = col3.slider("Chunks per profile", min_value=10, max_value=500, value=100)

    if not st.button("Load chunks"):
        return

    left_col, right_col = st.columns(2)
    _render_chunk_profile(left_col, labels[left], "Profile A", int(limit))
    if right != "(none)":
        _render_chunk_profile(right_col, labels[right], "Profile B", int(limit))
    else:
        with right_col:
            st.caption("Select a second profile to compare.")


def _post_search(payload: dict) -> tuple[int, object]:
    return api_post("/v1/search", json=payload)


def _render_rank_table(hits: list[dict]) -> None:
    rows = []
    for hit in hits:
        rows.append(
            {
                "rank": hit["rank"],
                "section": hit.get("section_number") or hit.get("section_path") or "",
                "retrieval_rank": hit.get("retrieval_rank"),
                "movement": hit.get("rank_movement"),
                "rerank_score": hit.get("rerank_score"),
                "retrieval_score": round(hit["score"], 4),
                "source": hit.get("source_name"),
            }
        )
    if rows:
        st.dataframe(rows, use_container_width=True)


def render_reranking_lab() -> None:
    st.header("Reranking Laboratory")
    st.caption(
        "Run first-stage retrieval, then reorder the same candidates with a local "
        "cross-encoder. Scores are relevance logits, not calibrated probabilities."
    )
    profiles = _profile_options()
    if not profiles:
        st.warning("No index profiles yet — ingest a document first.")
        return

    labels = _profile_labels(profiles)
    query = st.text_input("Query", value="confidential information", key="rerank_query")
    col1, col2, col3 = st.columns(3)
    selected = col1.selectbox("Profile", list(labels.keys()), index=0, key="rerank_profile")
    mode = col2.selectbox("Mode", ["bm25", "dense", "hybrid"], key="rerank_mode")
    top_k = col3.slider("Top K", min_value=1, max_value=20, value=5, key="rerank_topk")
    col4, col5, col6 = st.columns(3)
    candidate_k = col4.slider("Rerank candidates", min_value=1, max_value=100, value=20)
    reranker_model = col5.text_input(
        "Cross-encoder model (blank = configured default)", value="", key="rerank_model"
    )
    col6.metric("Candidate depth", candidate_k)

    if not st.button("Compare retrieval vs reranking") or not query.strip():
        return
    if candidate_k < top_k:
        st.error("Rerank candidates must be greater than or equal to Top K.")
        return

    profile_id = labels[selected]
    # One first-stage pool, shared by both views: the baseline requests the full
    # candidate pool (not just top-K), and the reranker reorders that same pool.
    # Neither view can then show a passage the other never had.
    pool_payload = {
        "query": query,
        "profile_id": profile_id,
        "mode": mode,
        "top_k": int(candidate_k),
        "candidates": int(candidate_k),
    }
    status, baseline = _post_search(pool_payload)
    if status != 200 or not isinstance(baseline, dict):
        st.error(f"Retrieval failed ({status}): {baseline}")
        return

    rerank_payload = {
        **pool_payload,
        "rerank": True,
        "rerank_candidates": int(candidate_k),
    }
    if reranker_model.strip():
        rerank_payload["reranker_model"] = reranker_model.strip()
    status, reranked = _post_search(rerank_payload)
    if status != 200 or not isinstance(reranked, dict):
        st.error(f"Reranking failed ({status}): {reranked}")
        if status == 503:
            st.info("Install the optional extra to enable reranking:  uv sync --extra dense")
        return

    info = reranked.get("reranker") or {}
    st.info(
        f"Both views reorder the **same {candidate_k}-passage first-stage pool**; the "
        f"reranked view displays the full pool reordered, with the top-{top_k} cut "
        "marked. A passage absent from the pool cannot be recovered by reranking."
    )
    st.caption(
        f"model {info.get('model_id')} • candidates {reranked.get('rerank_candidates')} "
        f"• init {reranked.get('timings_ms', {}).get('rerank_init')} ms "
        f"• inference {reranked.get('timings_ms', {}).get('rerank_inference')} ms "
        f"• total {reranked.get('total_ms')} ms"
    )
    for note in info.get("diagnostics", []) or []:
        st.warning(note)

    base_ids = {hit["chunk_id"] for hit in baseline.get("hits", [])}
    rerank_ids = {hit["chunk_id"] for hit in reranked.get("hits", [])}
    only_base = base_ids - rerank_ids
    only_rerank = rerank_ids - base_ids
    if only_base or only_rerank:
        st.warning(
            f"Pool mismatch: {len(only_base)} passage(s) only in first-stage, "
            f"{len(only_rerank)} only after rerank. They should start from the same pool."
        )

    left_col, right_col = st.columns(2)
    with left_col:
        st.subheader(f"First-stage ranking (pool of {len(baseline.get('hits', []))})")
        st.caption(f"total {baseline.get('total_ms')} ms • {baseline.get('timings_ms')}")
        _render_rank_table(baseline.get("hits", []))
    with right_col:
        st.subheader("After cross-encoder rerank (same pool)")
        st.caption(f"total {reranked.get('total_ms')} ms • {reranked.get('timings_ms')}")
        _render_rank_table(reranked.get("hits", []))

    top_set = {hit["chunk_id"] for hit in reranked.get("hits", [])[: int(top_k)]}
    moved = [hit for hit in reranked.get("hits", []) if hit.get("rank_movement")]
    st.info(
        f"{len(moved)} passage(s) changed rank; {len(top_set)} inside the top-{top_k} cut."
    )


def _fetch_experiments(dataset: str | None = None) -> list[dict]:
    params = {"include_recorded": "true"}
    if dataset:
        params["dataset"] = dataset
    status, body = api_get("/v1/experiments", **params)
    if status == 200 and isinstance(body, dict):
        return body.get("experiments", [])
    return []


def _fetch_datasets() -> list[dict]:
    status, body = api_get("/v1/datasets")
    if status == 200 and isinstance(body, dict):
        return body.get("datasets", [])
    return []


def _experiment_label(experiment: dict) -> str:
    dataset = experiment.get("dataset") or "?"
    mode = experiment.get("mode") or "?"
    qmode = experiment.get("query_mode") or "?"
    profile = (experiment.get("profile_id") or "")[:12]
    rerank = "+rerank" if experiment.get("rerank") else ""
    label = experiment.get("label") or ""
    source = experiment.get("source") or "?"
    prefix = f"{experiment['experiment_id']}"
    tail = f"{dataset} {mode}/{qmode}{rerank} {profile}"
    if label:
        tail += f" — {label}"
    return f"{prefix} [{source}] {tail}"


def _download_buttons(experiment_id: str) -> None:
    left, right = st.columns(2)
    status, json_text = api_text(f"/v1/experiments/{experiment_id}/export", format="json")
    if status == 200:
        left.download_button(
            "Download JSON",
            data=json_text,
            file_name=f"{experiment_id}.json",
            mime="application/json",
            key=f"dl-json-{experiment_id}",
        )
    status, csv_text = api_text(f"/v1/experiments/{experiment_id}/export", format="csv")
    if status == 200:
        right.download_button(
            "Download CSV",
            data=csv_text,
            file_name=f"{experiment_id}.csv",
            mime="text/csv",
            key=f"dl-csv-{experiment_id}",
        )


def _metric_table(values: dict, *, key: str) -> None:
    if not values:
        st.caption("No metrics recorded.")
        return
    st.dataframe(
        [{"metric": name, "value": round(float(value), 4)} for name, value in sorted(values.items())],
        use_container_width=True,
        key=key,
    )


def _render_dataset_browser() -> None:
    datasets = _fetch_datasets()
    st.subheader("Datasets")
    if not datasets:
        st.caption("No datasets available. Import one with:  rag-workbench datasets import …")
        return
    st.dataframe(
        [
            {
                "slug": d.get("slug"),
                "name": d.get("name"),
                "version": d.get("version"),
                "mapping": d.get("relevance_mapping"),
                "docs": d.get("corpus_count"),
                "queries": d.get("query_count"),
                "qrels": d.get("qrel_count"),
                "source": d.get("source"),
                "license": d.get("license"),
            }
            for d in datasets
        ],
        use_container_width=True,
        key="evaluation-datasets",
    )


def _render_run_form() -> None:
    st.subheader("Run an evaluation")
    if READ_ONLY:
        st.info(
            "Read-only demo: running new evaluations is disabled. The committed "
            "recorded experiments are shown below."
        )
        return
    if not LAB_EVAL_ENABLED:
        st.info(
            "Evaluation runs are available but switched off for this session. "
            "Enable them in the sidebar under Laboratory Controls."
        )
        return
    datasets = _fetch_datasets()
    profiles = _profile_options()
    if not datasets:
        st.caption("Import a dataset first.")
        return
    if not profiles:
        st.caption("No index profiles yet — ingest a document or a dataset corpus first.")
        return

    dataset_labels = {
        f"{d['slug']} ({d['corpus_count']} docs, {d['query_count']} queries)": d["slug"]
        for d in datasets
    }
    profile_labels = _profile_labels(profiles)

    col1, col2, col3 = st.columns(3)
    dataset = col1.selectbox("Dataset", list(dataset_labels.keys()), key="eval-dataset")
    profile = col2.selectbox("Profile", list(profile_labels.keys()), key="eval-profile")
    mode = col3.selectbox("Mode", ["bm25", "dense", "hybrid"], key="eval-mode")
    col4, col5, col6 = st.columns(3)
    query_mode = col4.selectbox("Query mode", ["or", "and"], key="eval-query-mode")
    top_k = col5.slider("Top K", 1, 50, 10, key="eval-topk")
    candidate_depth = col6.slider("Candidate depth", 10, 200, 100, key="eval-depth")
    rerank = st.checkbox(
        "Rerank candidates with a cross-encoder", value=False, key="eval-rerank"
    )
    rerank_candidates = st.slider(
        "Rerank candidates", 1, 200, 20, key="eval-rerank-candidates"
    )
    label = st.text_input("Label (optional)", value="", key="eval-label")
    subset = st.checkbox("Quick subset (limit queries)", value=False, key="eval-subset")

    if not st.button("Run evaluation", key="eval-run"):
        return

    payload = {
        "dataset": dataset_labels[dataset],
        "profile_id": profile_labels[profile],
        "mode": mode,
        "query_mode": query_mode,
        "candidate_depth": int(candidate_depth),
        "top_k": int(top_k),
        "k_values": sorted({1, 3, 5, 10, int(top_k)}),
        "rerank": bool(rerank),
        "subset": bool(subset),
    }
    if subset:
        payload["max_queries"] = 20
    if rerank:
        payload["rerank_candidates"] = int(rerank_candidates)
    if label.strip():
        payload["label"] = label.strip()

    with st.spinner("Running evaluation over the dataset …"):
        status, body = api_post("/v1/evaluations", json=payload)
    if status != 200 or not isinstance(body, dict):
        st.error(f"Evaluation failed ({status}): {body}")
        if status == 503:
            st.info("Dense/rerank require the optional extra:  uv sync --extra dense")
        return
    st.success(f"Recorded experiment {body.get('experiment_id')}")
    _render_experiment_detail(body)


def _render_experiment_detail(record: dict) -> None:
    config = record.get("config", {})
    counts = record.get("counts", {})
    st.caption(
        f"dataset {record.get('dataset', {}).get('name')} "
        f"v{record.get('dataset', {}).get('version')} • "
        f"mode {config.get('mode')}/{config.get('query_mode')} • "
        f"profile {config.get('profile_id')} • rerank {config.get('rerank')} • "
        f"depth {config.get('candidate_depth')} • top_k {config.get('top_k')}"
    )
    st.markdown("**Counts**")
    st.dataframe(
        [{"key": k, "value": v} for k, v in counts.items()],
        use_container_width=True,
        key=f"counts-{record.get('experiment_id')}",
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Ranking metrics**")
        _metric_table(record.get("aggregate", {}), key=f"agg-{record.get('experiment_id')}")
    with right:
        st.markdown("**Candidate-pool metrics**")
        _metric_table(
            record.get("candidate_metrics", {}), key=f"cand-{record.get('experiment_id')}"
        )
    for note in record.get("notes", []):
        st.info(note)
    _download_buttons(record["experiment_id"])


def _render_deployment_diagnostics() -> None:
    """Show the deployment revision, and asset locations where the host allows.

    The public, read-only deployment must not disclose its filesystem layout, so
    the resolved paths come from ``/v1/config/diagnostics``, which is served only
    by a writable (operator/local) deployment. On the public demo that route
    returns 403 and this expander shows just the revision identity and booleans.
    """

    status, cfg = api_get("/v1/config")
    if status != 200 or not isinstance(cfg, dict):
        return
    with st.expander("Deployment diagnostics"):
        backend_version = cfg.get("version")
        disk_version = on_disk_version()
        backend_revision = cfg.get("revision")
        backend_revision_source = cfg.get("revision_source")
        disk_revision, disk_revision_source = on_disk_revision()
        st.caption(
            f"version {backend_version} • backend revision {backend_revision} "
            f"({backend_revision_source}) • source-tree revision {disk_revision} "
            f"({disk_revision_source})"
        )
        if (
            disk_revision
            and backend_revision
            and disk_revision != backend_revision
        ):
            st.warning(
                f"Backend revision mismatch: the API is serving {backend_revision} "
                f"but the source tree is {disk_revision}. A full app restart/redeploy "
                "may be required to load the current backend code."
            )
        elif disk_version != "unknown" and backend_version and backend_version != disk_version:
            st.warning(
                f"Version mismatch: the API reports {backend_version} but the source "
                f"tree is {disk_version}. A full app restart may be required to load "
                "the current backend."
            )
        diag_status, diag = api_get("/v1/config/diagnostics")
        if diag_status == 200 and isinstance(diag, dict):
            # Writable/local deployment: full resolved locations are available.
            st.caption(
                f"repo_root {diag.get('repo_root')} • read_only {cfg.get('read_only')} • "
                f"autoseed {cfg.get('autoseed_sample')} • "
                f"dense_available {cfg.get('dense_available')}"
            )
            st.caption(
                f"datasets {diag.get('bundled_datasets_dir')} "
                f"(exists={cfg.get('bundled_datasets_exists')}) • "
                f"recorded {diag.get('recorded_experiments_dir')} "
                f"(exists={cfg.get('recorded_experiments_exists')})"
            )
        else:
            st.caption(
                f"read_only {cfg.get('read_only')} • autoseed {cfg.get('autoseed_sample')} • "
                f"dense_available {cfg.get('dense_available')} • "
                f"datasets present {cfg.get('bundled_datasets_exists')} • "
                f"recorded experiments present {cfg.get('recorded_experiments_exists')}"
            )


def _render_owner_panel() -> None:
    """Authenticated, server-bounded evaluation for the deployment owner.

    Appears only when the backend explicitly permits hosted owner evaluations and
    this browser session holds a valid owner session. The session token is issued
    only after a verified passkey assertion, is sent solely as an ``Authorization``
    header and is never stored in the URL, diagnostics or a log. The backend
    re-authorizes every request and enforces the bounds; this panel merely calls
    the endpoint.
    """

    status, cfg = api_get("/v1/config")
    if status != 200 or not isinstance(cfg, dict):
        return
    owner = ((cfg.get("laboratory") or {}).get("owner_evaluation")) or {}
    if not owner.get("permitted"):
        return
    session = st.session_state.get("owner_session") or {}
    token = st.session_state.get("owner_session_token") or ""
    if not isinstance(session, dict) or not session.get("authenticated") or not token:
        return
    policy = cfg.get("owner_evaluation_policy") or {}
    allowed = policy.get("allowed_datasets") or []
    max_queries = int(policy.get("max_queries", 20))
    with st.expander("Owner testing (bounded, authenticated)"):
        st.caption(
            "Optional authenticated evaluation for the signed-in owner. Bounds are "
            f"enforced by the backend: BM25 only, datasets {', '.join(allowed) or '—'}, "
            f"max {max_queries} queries, candidate depth "
            f"{policy.get('max_candidate_depth')}, {policy.get('max_runtime_s')}s budget."
        )
        dataset = st.selectbox(
            "Bundled dataset",
            allowed or ["legal-contracts"],
            key="lab_owner_dataset",
        )
        queries = st.slider(
            "Max queries", 1, max_queries, min(5, max_queries), key="lab_owner_queries"
        )
        if st.button("Run bounded owner evaluation", key="lab_owner_run"):
            result_status, body = api_post(
                "/v1/owner/evaluations",
                headers={"Authorization": f"Bearer {token}"},
                json={"dataset": dataset, "max_queries": int(queries)},
            )
            if result_status == 200 and isinstance(body, dict):
                st.success(f"Owner evaluation {body.get('experiment_id')}")
                _render_experiment_detail(body)
            else:
                st.error(f"Owner evaluation failed ({result_status}): {body}")


def render_evaluation_lab() -> None:
    st.header("Evaluation Lab")
    st.caption(
        "Run a stored dataset through a real retrieval configuration and record the "
        "metrics. Everything here is measured in this project; nothing is simulated."
    )
    _render_deployment_diagnostics()
    if READ_ONLY:
        st.info(
            "Read-only deployment: new runs are disabled. Recorded benchmark "
            "experiments committed to the repository are shown below."
        )
    _render_owner_panel()
    _render_dataset_browser()
    st.divider()
    _render_run_form()
    st.divider()
    st.subheader("Recorded experiments")
    experiments = _fetch_experiments()
    if not experiments:
        st.caption("No experiments recorded yet.")
        return
    st.dataframe(
        [
            {
                "experiment_id": e.get("experiment_id"),
                "source": e.get("source"),
                "dataset": e.get("dataset"),
                "mode": e.get("mode"),
                "query_mode": e.get("query_mode"),
                "profile": e.get("profile_id"),
                "rerank": e.get("rerank"),
                "label": e.get("label"),
                "recall@10": (e.get("aggregate", {}) or {}).get("recall@10"),
                "ndcg@10": (e.get("aggregate", {}) or {}).get("ndcg@10"),
                "mrr@10": (e.get("aggregate", {}) or {}).get("mrr@10"),
                "candidate_recall@100": (e.get("candidate_metrics", {}) or {}).get(
                    "candidate_recall@100"
                ),
            }
            for e in experiments
        ],
        use_container_width=True,
        key="evaluation-experiments",
    )
    labels = {_experiment_label(e): e["experiment_id"] for e in experiments}
    selected = st.selectbox("Inspect experiment", list(labels.keys()), key="eval-inspect")
    if st.button("Load experiment", key="eval-load"):
        status, body = api_get(f"/v1/experiments/{labels[selected]}")
        if status == 200 and isinstance(body, dict):
            _render_experiment_detail(body)
        else:
            st.error(f"Could not load experiment ({status}): {body}")


def _fetch_answer_demos() -> list[dict]:
    status, body = api_get("/v1/answer-demos")
    if status != 200 or not isinstance(body, dict):
        return []
    demos = body.get("demos", [])
    return demos if isinstance(demos, list) else []


def _answer_demo_label(demo: dict) -> str:
    status = demo.get("status") or "?"
    question = " ".join((demo.get("question") or "").split())[:70]
    return f"{status} • {question}"


def _render_answer_record(record: dict) -> None:
    """Render one answer record — live or recorded. Never edits the content."""

    status = record.get("status", "?")
    if status == "answered":
        st.success(f"status: {status}")
    elif status == "partially_answered":
        st.info(f"status: {status}")
    elif status == "insufficient_evidence":
        st.warning(f"status: {status} — grounded abstention")
    else:
        st.error(f"status: {status}")

    st.markdown(f"**Question.** {record.get('question', '')}")
    answer = record.get("answer") or ""
    if answer:
        st.markdown("**Answer**")
        st.write(answer)
    if record.get("reason"):
        st.caption(f"reason: {record['reason']}")

    claims = record.get("claims") or []
    if claims:
        st.markdown("**Claims and cited passages**")
        st.caption(
            "Each claim is shown with the verbatim passage(s) it cites, so a claim "
            "whose citation does not actually support it can be seen here. Whether a "
            "citation supports the claim is *not* decided automatically (semantic "
            "support is not determined)."
        )
        evidence_by_id = {
            str(item.get("source_id")): item for item in (record.get("evidence") or [])
        }
        for index, claim in enumerate(claims, 1):
            st.markdown(f"**{index}. {claim.get('text', '')}**")
            cites = claim.get("citations") or []
            if not cites:
                st.caption("_(no citation)_")
                continue
            for raw in cites:
                key = str(raw).strip().upper()
                item = evidence_by_id.get(key) or evidence_by_id.get(str(raw))
                if item is None:
                    st.caption(f"`{raw}` — not one of the selected passages")
                    continue
                where = item.get("source_name") or ""
                if item.get("section_number"):
                    where += f" #{item['section_number']}"
                elif item.get("section_path"):
                    where += f" {item['section_path']}"
                excerpt = " ".join((item.get("text") or "").split())[:240]
                st.caption(f"`{raw}` → {where}: {excerpt}")
    if record.get("unanswered"):
        st.markdown("**Unanswered**")
        for item in record["unanswered"]:
            st.caption(f"• {item}")

    validation = record.get("citation_validation") or {}
    st.markdown("**Citation validation** (mechanical checks vs. human review)")
    st.caption(
        f"structural_valid {validation.get('structural_valid')} • "
        f"citation_valid {validation.get('citation_valid')} • "
        f"all_citations_resolved {validation.get('all_citations_resolved')} • "
        f"all_claims_cited {validation.get('all_claims_cited')}"
    )
    st.caption(
        f"semantic_support {validation.get('semantic_support', 'not_determined')} • "
        f"answer_claims_consistency "
        f"{validation.get('answer_claims_consistency', 'not_determined')} "
        "(not asserted: a citation resolving to a real passage does NOT prove it "
        "supports the claim, nor that the answer text matches the structured claims)"
    )
    if validation.get("unknown_source_ids"):
        st.caption(f"unknown citations: {validation['unknown_source_ids']}")
    if validation.get("malformed_citations"):
        st.caption(f"malformed citations: {validation['malformed_citations']}")
    if validation.get("uncited_claim_indices"):
        st.caption(f"uncited claim indices: {validation['uncited_claim_indices']}")
    for note in validation.get("notes", []):
        st.caption(f"note: {note}")

    context = record.get("context") or {}
    st.markdown("**Context: retrieved vs selected vs excluded**")
    st.caption(
        f"retrieved {context.get('retrieved')} • selected {context.get('selected')} • "
        f"excluded {len(record.get('excluded_evidence') or [])} • "
        f"tokens ~{context.get('token_estimate')}/{context.get('token_budget')} "
        f"({context.get('token_count_method')})"
    )

    evidence = record.get("evidence") or []
    if evidence:
        with st.expander(f"Selected evidence ({len(evidence)} passages) — verbatim"):
            for item in evidence:
                where = item.get("source_name") or ""
                if item.get("section_number"):
                    where += f" #{item['section_number']}"
                elif item.get("section_path"):
                    where += f" {item['section_path']}"
                if item.get("page_start") is not None:
                    where += f" p.{item['page_start']}"
                st.markdown(f"**{item.get('source_id')}** — {where}")
                st.write(item.get("text", ""))
    excluded = record.get("excluded_evidence") or []
    if excluded:
        with st.expander(f"Retrieved but excluded ({len(excluded)} passages)"):
            st.dataframe(
                [
                    {
                        "rank": item.get("rank"),
                        "source": item.get("source_name"),
                        "reason": item.get("reason"),
                        "detail": item.get("detail"),
                    }
                    for item in excluded
                ],
                hide_index=True,
            )

    retrieval = record.get("retrieval") or {}
    generation = record.get("generation") or {}
    timings = record.get("timings_ms") or {}
    st.caption(
        f"retrieval mode {retrieval.get('mode')} ({retrieval.get('query_mode')}) • "
        f"collection {retrieval.get('collection')} • "
        f"model {generation.get('model_id')} • attempted {generation.get('attempted')}"
    )
    st.caption("timings_ms: " + ", ".join(f"{k}={v}" for k, v in timings.items()))

    diagnostics = record.get("diagnostics") or []
    if diagnostics:
        with st.expander(f"Diagnostics ({len(diagnostics)})", expanded=True):
            for diag in diagnostics:
                st.warning(f"**{diag.get('code')}** — {diag.get('detail')}")

    if record.get("recorded"):
        st.caption(
            "Recorded run: previously executed locally and committed as an artifact. "
            "It is not live inference."
        )


def _render_answer_quality() -> None:
    st.subheader("Grounded-answer quality (recorded)")
    status, report = api_get("/v1/answer-quality")
    if status != 200 or not isinstance(report, dict):
        st.caption(
            "No recorded answer-quality report. Run "
            "`rag-workbench answer record` locally to produce one."
        )
        return
    metrics = report.get("metrics", {})
    config = report.get("config", {})
    dataset = report.get("dataset", {})
    st.caption(
        f"{report.get('report_id')} • dataset {dataset.get('name')} "
        f"v{dataset.get('version')} • model {config.get('model_id')} • "
        f"mode {config.get('mode')} • query_mode {config.get('query_mode')}"
    )
    st.dataframe([report.get("counts", {})], hide_index=True)
    labels = [
        ("structural_validity_rate", "structural validity"),
        ("citation_validity_rate", "citation validity"),
        ("invalid_citation_rate", "invalid citation rate"),
        ("malformed_citation_rate", "malformed citation rate"),
        ("status_accuracy_rate", "answer-state accuracy"),
        ("correct_abstention_rate", "correct abstention (unanswerable)"),
        ("over_abstention_rate", "over-abstention (answerable)"),
        ("complete_reference_coverage_rate", "reference coverage: complete"),
        ("partial_reference_coverage_rate", "reference coverage: partial"),
        ("reference_evidence_unit_coverage_rate", "reference coverage: per-anchor"),
        ("reference_evidence_selected_rate", "reference evidence selected (any, legacy)"),
        ("reference_evidence_retrieved_but_excluded_rate", "evidence retrieved but excluded"),
        ("reference_evidence_never_retrieved_rate", "evidence never retrieved"),
        ("context_exclusion_rate", "context exclusions"),
        ("generation_failure_rate", "generation failures"),
    ]
    st.dataframe(
        [{"metric": label, "value": metrics.get(key)} for key, label in labels],
        hide_index=True,
    )
    st.caption(
        f"semantic support: {metrics.get('semantic_support')} — whether a cited "
        "passage truly supports a claim is not deterministically decidable and is "
        "not scored."
    )

    loss = report.get("evidence_loss")
    if loss:
        st.markdown("**Evidence loss** (reference-evidence coverage per question)")
        st.caption(
            f"included {loss.get('included')} • "
            f"retrieved but excluded {loss.get('retrieved_but_excluded')} • "
            f"never retrieved {loss.get('never_retrieved')} "
            f"(of {loss.get('questions_with_reference')} questions with reference evidence)"
        )
        st.dataframe(
            [
                {
                    "id": case.get("id"),
                    "classification": case.get("classification"),
                    "required": case.get("required"),
                    "retrieved": case.get("retrieved"),
                    "selected": case.get("selected"),
                    "missing (not selected)": ", ".join(case.get("missing_selected") or []),
                    "missing (not retrieved)": ", ".join(case.get("missing_retrieved") or []),
                }
                for case in loss.get("cases", [])
            ],
            hide_index=True,
        )

    _render_quality_report_comparison()
    for item in report.get("limitations", []):
        st.caption(f"• {item}")


def _render_quality_report_comparison() -> None:
    """List every recorded quality report so configuration trade-offs are visible."""

    status, body = api_get("/v1/answer-quality-reports")
    reports = body.get("reports") if status == 200 and isinstance(body, dict) else None
    if not reports or len(reports) < 2:
        return
    st.markdown("**Recorded configurations** (newest first)")
    st.caption(
        "Each row is a separate recorded experiment. Original Phase 5 runs are kept; "
        "later rows are additional configurations. Compare coverage, abstention and "
        "latency rather than assuming more evidence is always better."
    )
    st.dataframe(
        [
            {
                "report_id": row.get("report_id"),
                "config": row.get("label"),
                "recommended": "✔" if row.get("recommended") else "",
                "prompt": row.get("prompt_variant"),
                "max_evidence": row.get("max_evidence"),
                "seed": row.get("seed"),
                "struct": row.get("structural_validity_rate"),
                "cite": row.get("citation_validity_rate"),
                "ref_complete": row.get("complete_reference_coverage_rate"),
                "ref_units": row.get("reference_evidence_unit_coverage_rate"),
                "correct_abstain": row.get("correct_abstention_rate"),
                "state_acc": row.get("status_accuracy_rate"),
                "gen_ms_p50": row.get("generation_ms_p50"),
            }
            for row in reports
        ],
        hide_index=True,
    )


def render_answer_lab() -> None:
    st.header("Answer Laboratory")
    st.caption(
        "Optional evidence-grounded answer generation over the retrieved passages, "
        "with deterministic citation verification. The intended public deployment is "
        "read-only and serves recorded runs; nothing here uploads documents or calls "
        "a paid API."
    )
    status, cfg = api_get("/v1/config")
    if status == 200 and isinstance(cfg, dict):
        st.caption(
            f"generation_enabled {cfg.get('generation_enabled')} • "
            f"generation_available {cfg.get('generation_available')} • "
            f"recorded demos {cfg.get('recorded_answer_demos')}"
        )
    st.divider()
    _render_answer_quality()
    st.divider()

    lab_generation = (
        ((cfg.get("laboratory") or {}).get("generation")) if isinstance(cfg, dict) else {}
    ) or {}
    generation_permitted = isinstance(cfg, dict) and bool(
        cfg.get("generation_enabled")
    ) and not READ_ONLY and bool(lab_generation.get("permitted", False))
    if not generation_permitted:
        st.info(
            lab_generation.get("reason")
            or "Live generation is disabled on this deployment. The recorded answer "
            "demonstrations below were produced locally with a real local model and "
            "committed as artifacts."
        )
    elif not LAB_GENERATION_ENABLED:
        st.info(
            "Live generation is available but switched off for this session. Enable "
            "it in the sidebar under Laboratory Controls."
        )
    else:
        with st.form("answer-form"):
            question = st.text_area("Question", height=80)
            profiles = _profile_options()
            labels = _profile_labels(profiles)
            profile_choice = st.selectbox("Index profile", ["(default)"] + list(labels.keys()))
            columns = st.columns(4)
            mode = columns[0].selectbox("Mode", ["bm25", "dense", "hybrid"])
            query_mode = columns[1].selectbox("Query mode", ["and", "or"], index=1)
            collection = columns[2].text_input("Collection filter", value="")
            rerank = columns[3].checkbox("Rerank")
            columns2 = st.columns(4)
            candidate_depth = columns2[0].number_input("Candidate depth", 1, 100, 10)
            max_evidence = columns2[1].number_input("Max evidence", 1, 20, 4)
            context_tokens = columns2[2].number_input("Context tokens", 128, 32768, 2048, step=128)
            submitted = st.form_submit_button("Generate grounded answer")

        if submitted:
            if not question.strip():
                st.error("Enter a question first.")
            else:
                payload = {
                    "question": question.strip(),
                    "profile_id": labels.get(profile_choice),
                    "mode": mode,
                    "query_mode": query_mode,
                    "collection": collection.strip() or None,
                    "candidate_depth": int(candidate_depth),
                    "max_evidence": int(max_evidence),
                    "context_tokens": int(context_tokens),
                    "rerank": bool(rerank),
                }
                answer_status, body = api_post("/v1/answer", json=payload)
                if answer_status != 200 or not isinstance(body, dict):
                    st.error(f"Generation unavailable ({answer_status}): {body}")
                else:
                    _render_answer_record(body)
        st.divider()

    st.subheader("Recorded answer demonstrations")
    demos = _fetch_answer_demos()
    if not demos:
        st.caption(
            "No recorded answer demos. Run `rag-workbench answer record` locally to "
            "produce them."
        )
        return
    labels_by_demo = {_answer_demo_label(d): d.get("demo_id") for d in demos}
    selected = st.selectbox(
        "Recorded run", list(labels_by_demo.keys()), key="answer-demo-select"
    )
    demo_id = labels_by_demo[selected]
    detail_status, detail = api_get(f"/v1/answer-demos/{demo_id}")
    if detail_status != 200 or not isinstance(detail, dict):
        st.error(f"Could not load demo ({detail_status}): {detail}")
        return
    _render_answer_record(detail.get("demo", {}))


def render_failure_explorer() -> None:
    st.header("Failure Explorer")
    st.caption(
        "Diagnoses are derived only from observable evidence — the query's "
        "judgments, the retrieved candidates and the final ranking. An unresolved "
        "case is reported as unresolved, never guessed."
    )
    experiments = _fetch_experiments()
    if not experiments:
        st.warning("No experiments recorded yet.")
        return
    labels = {_experiment_label(e): e["experiment_id"] for e in experiments}
    selected = st.selectbox("Experiment", list(labels.keys()), key="failure-experiment")
    if not st.button("Load failures", key="failure-load"):
        return

    status, record = api_get(f"/v1/experiments/{labels[selected]}")
    if status != 200 or not isinstance(record, dict):
        st.error(f"Could not load experiment ({status}): {record}")
        return

    per_query = record.get("per_query", [])
    # Only *evaluated* queries can carry a failure signal: a query with no
    # relevance judgments has no metric and is excluded from every aggregate, so
    # it must not inflate the diagnosis counts or the denominator.
    evaluated = [q for q in per_query if q.get("status") == "ok" and q.get("metrics")]
    unevaluated = len(per_query) - len(evaluated)
    counts: dict[str, int] = {}
    failing = []
    for query in evaluated:
        codes = [d["code"] for d in query.get("diagnostics", [])]
        real_codes = [c for c in codes if c != "unresolved"]
        if real_codes:
            failing.append((query, real_codes))
        for code in codes:
            counts[code] = counts.get(code, 0) + 1

    st.markdown("**Diagnosis frequency** (evaluated queries per code)")
    if counts:
        st.bar_chart(counts)
    else:
        st.caption("No diagnostics recorded for the evaluated queries of this experiment.")

    denominator = (
        f"{len(failing)} of {len(evaluated)} evaluated queries carry an observable "
        "failure signal"
    )
    if unevaluated:
        denominator += f"; {unevaluated} unjudged query(ies) were not evaluated"
    st.caption(denominator + ".")
    code_filter = st.multiselect(
        "Filter by diagnosis", sorted({c for _, codes in failing for c in codes}),
        key="failure-filter",
    )
    limit = st.slider("Max queries to show", 5, 100, 25, key="failure-limit")

    shown = 0
    for query, codes in failing:
        if code_filter and not set(codes) & set(code_filter):
            continue
        if shown >= limit:
            break
        shown += 1
        metrics = query.get("metrics") or {}
        recall10 = metrics.get("recall@10")
        header = (
            f"{query['query_id']} • relevant={query['num_relevant']} • "
            f"recall@10={recall10 if recall10 is None else round(recall10, 3)} • "
            f"candidates={query.get('candidate_count')}"
        )
        with st.expander(header):
            st.write(query.get("query", ""))
            for diag in query.get("diagnostics", []):
                st.warning(f"**{diag['code']}** — {diag['detail']}")
            st.markdown("**Ranked results**")
            for item in query.get("trace", [])[:10]:
                flag = "✅ relevant" if item.get("relevant") else "—"
                where = item.get("section_number") or item.get("section_path") or ""
                st.caption(
                    f"#{item['rank']} {flag} • key={item.get('key')} • "
                    f"{item.get('source_name')} {where} • "
                    f"{item.get('score_kind')}={round(float(item.get('score', 0.0)), 3)}"
                )
                excerpt = " ".join((item.get("excerpt") or "").split())[:200]
                if excerpt:
                    st.write(excerpt)


def render_performance_lab() -> None:
    st.header("Performance Lab")
    st.caption(
        "Latency percentiles are measured only over the queries that actually "
        "retrieved (unjudged queries are skipped before retrieval); candidate "
        "recall (first-stage pool coverage) is reported separately from ranking "
        "recall (top-K coverage)."
    )
    experiments = _fetch_experiments()
    if not experiments:
        st.warning("No experiments recorded yet.")
        return
    labels = {_experiment_label(e): e["experiment_id"] for e in experiments}
    selected = st.selectbox("Experiment", list(labels.keys()), key="perf-experiment")
    if not st.button("Load performance", key="perf-load"):
        return
    status, record = api_get(f"/v1/experiments/{labels[selected]}")
    if status != 200 or not isinstance(record, dict):
        st.error(f"Could not load experiment ({status}): {record}")
        return

    latency = record.get("latency", {})
    rows = []
    for stage, stats in latency.items():
        if isinstance(stats, dict):
            rows.append({"stage": stage, **stats})
        else:
            rows.append({"stage": stage, "value": stats})
    st.markdown("**Latency (ms)**")
    st.dataframe(rows, use_container_width=True, key="perf-latency")

    candidate = record.get("candidate_metrics", {})
    aggregate = record.get("aggregate", {})
    recall_by_stage: dict[str, float] = {}
    for depth in (10, 20, 50, 100):
        name = f"candidate_recall@{depth}"
        if name in candidate:
            recall_by_stage[f"candidate@{depth}"] = float(candidate[name])
    for k in (1, 3, 5, 10):
        name = f"recall@{k}"
        if name in aggregate:
            recall_by_stage[f"rank@{k}"] = float(aggregate[name])
    if recall_by_stage:
        st.markdown("**Candidate pool vs ranking recall**")
        st.bar_chart(recall_by_stage)
    st.caption(
        "candidate@N = share of relevant units present anywhere in the first-stage "
        "pool of N chunks (before reranking); rank@K = share present in the final "
        "top-K. A large gap points at ranking; a low candidate@N points at the pool "
        "or the index."
    )

    st.divider()
    st.subheader("Configuration comparison")
    st.caption("Pick two or more experiments to compare aggregate metrics side by side.")
    compare = st.multiselect(
        "Experiments", list(labels.keys()), key="perf-compare", default=[]
    )
    if len(compare) < 2:
        return
    records = []
    for label in compare:
        status, body = api_get(f"/v1/experiments/{labels[label]}")
        if status == 200 and isinstance(body, dict):
            records.append(body)
    metrics = sorted({m for r in records for m in (r.get("aggregate") or {})})
    table = []
    for metric in metrics:
        row = {"metric": metric}
        for record in records:
            row[record["experiment_id"]] = round(
                float((record.get("aggregate") or {}).get(metric, 0.0)), 4
            )
        table.append(row)
    st.dataframe(table, use_container_width=True, key="perf-compare-table")


def _human_bytes(num_bytes: float | None) -> str:
    """Render a byte count in binary units; mirrors the backend formatter."""

    if num_bytes is None:
        return "—"
    size = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1024 or unit == "TB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{size:.1f} TB"


def _render_scaling_baselines() -> None:
    st.caption(
        "Locally measured indexing and retrieval cost at several corpus sizes. "
        "These are real measurements from one machine, committed as artifacts so the "
        "read-only demo can show them — they are not a production throughput claim."
    )
    status, body = api_get("/v1/scaling-benchmarks")
    if status != 200 or not isinstance(body, dict):
        st.error(f"Could not load scaling benchmarks ({status}): {body}")
        return
    benchmarks = body.get("benchmarks", [])
    if not benchmarks:
        st.info(
            "No recorded scaling benchmarks. Run "
            "`rag-workbench scaling run --documents 100,1000,5000 --record` locally."
        )
        return

    labels = {}
    for item in benchmarks:
        marker = " [invalid]" if item.get("invalid") else ""
        labels[
            f"{item.get('benchmark_id')}{marker} • "
            f"{item.get('workload_count')} workloads • docs≤{item.get('documents_max')}"
        ] = item.get("benchmark_id")
    selected = st.selectbox("Recorded benchmark", list(labels.keys()), key="scaling-select")
    detail_status, record = api_get(f"/v1/scaling-benchmarks/{labels[selected]}")
    if detail_status != 200 or not isinstance(record, dict):
        st.error(f"Could not load benchmark ({detail_status}): {record}")
        return

    env = record.get("environment") or {}
    hardware = record.get("hardware") or {}
    provenance = record.get("provenance") or record.get("git_commit")
    st.caption(
        f"created {record.get('created_at')} • revision {provenance} • "
        f"platform {env.get('platform')} • python {env.get('python')} • "
        f"cpu {hardware.get('processor')} ({hardware.get('cpu_count')} cores)"
    )
    if record.get("git_dirty"):
        st.warning(
            "This artifact was recorded from a working tree with uncommitted "
            "changes, so its revision identifies HEAD, not the exact code that ran."
        )
    corpus = record.get("corpus") or {}
    st.caption(
        f"corpus {corpus.get('documents')} docs • "
        f"seed {(corpus.get('config') or {}).get('seed')} • "
        f"fingerprint {str(corpus.get('fingerprint'))[:16]}"
    )

    rows = []
    for workload in record.get("workloads", []):
        ingestion = workload.get("ingestion") or {}
        index = workload.get("index") or {}
        memory = workload.get("memory") or {}
        rows.append(
            {
                "documents": workload.get("documents"),
                "chunks": workload.get("chunks"),
                "docs_per_second": ingestion.get("documents_per_second"),
                "source_bytes": index.get("source_bytes"),
                "db_bytes": index.get("db_bytes"),
                "fts_bytes": index.get("fts_bytes"),
                "sampled_peak_rss_mb": memory.get("sampled_peak_rss_mb"),
                "tracemalloc_peak_mb": memory.get("tracemalloc_peak_mb"),
            }
        )
    if rows:
        st.markdown("**Indexing footprint by corpus size**")
        st.dataframe(rows, hide_index=True, use_container_width=True)
        methods = {
            (workload.get("memory") or {}).get("method")
            for workload in record.get("workloads", [])
        }
        methods.discard(None)
        st.caption(
            "Memory: `sampled_peak_rss_mb` is the maximum of periodic process-RSS "
            "samples and needs `psutil`; `tracemalloc_peak_mb` is a Python-managed "
            "allocation peak and is **not** total process memory. "
            f"Method(s): {', '.join(sorted(str(m) for m in methods)) or 'unknown'}."
        )

    latency_rows = []
    for workload in record.get("workloads", []):
        for retrieval in workload.get("retrieval", []):
            stats = retrieval.get("latency_ms") or {}
            latency_rows.append(
                {
                    "documents": workload.get("documents"),
                    "mode": retrieval.get("mode"),
                    "depth": retrieval.get("candidate_depth"),
                    "queries": retrieval.get("queries"),
                    "p50_ms": stats.get("p50"),
                    "p95_ms": stats.get("p95"),
                    "errors": retrieval.get("errors"),
                }
            )
    if latency_rows:
        st.markdown("**Retrieval latency by corpus size** (20 queries per configuration)")
        st.dataframe(latency_rows, hide_index=True, use_container_width=True)

    if record.get("failures"):
        st.warning("Recorded failures during this run: " + "; ".join(record["failures"]))
    st.markdown("**Limitations** (recorded with the artifact)")
    for note in record.get("limitations", []):
        st.caption(f"• {note}")


def _render_scaling_capacity() -> None:
    st.caption(
        "A deterministic capacity model, not a simulation or a measurement. Every "
        "line is labelled with its basis: `exact` arithmetic under the stated "
        "assumptions, `assumed` illustrative defaults, `documented` vendor ratios, "
        "`measured` values from a recorded run, or `unknown`. The default throughput "
        "and storage-overhead values are assumptions — replace them with your own."
    )
    columns = st.columns(3)
    documents = columns[0].number_input(
        "Documents", 1_000, 1_000_000_000, 10_000_000, step=100_000, key="cap-docs"
    )
    chunks = columns[1].number_input(
        "Chunks / document", 0.1, 200.0, 10.0, step=0.5, key="cap-chunks"
    )
    dimensions = columns[2].selectbox(
        "Embedding dimensions", [384, 512, 768, 1024, 1536], key="cap-dims"
    )
    columns2 = st.columns(4)
    vector_indexes = columns2[0].number_input("Vector indexes", 1, 8, 1, key="cap-vidx")
    replication = columns2[1].number_input("Replication factor", 1, 6, 2, key="cap-repl")
    backups = columns2[2].number_input("Backup copies", 0, 4, 1, key="cap-backup")
    headroom = columns2[3].slider("Operational headroom", 0.0, 1.0, 0.25, 0.05, key="cap-head")

    st.markdown("**Ingestion projection (optional)**")
    columns3 = st.columns(4)
    docs_per_day = columns3[0].number_input(
        "Documents / day", 0, 50_000_000, 100_000, step=10_000, key="cap-dpd"
    )
    throughput = columns3[1].number_input(
        "Embedding chunks/s/worker", 0.1, 5000.0, 20.0, step=1.0, key="cap-tput"
    )
    workers = columns3[2].number_input("Workers", 1, 256, 4, key="cap-workers")
    indexing_rate = columns3[3].number_input(
        "Indexing chunks/s", 1.0, 100000.0, 200.0, step=10.0, key="cap-idx"
    )

    payload = {
        "storage": {
            "documents": int(documents),
            "chunks_per_document": float(chunks),
            "embedding_dimensions": int(dimensions),
            "vector_indexes": int(vector_indexes),
            "replication_factor": int(replication),
            "backup_copies": int(backups),
            "operational_headroom": float(headroom),
        },
        "ingestion": {
            "documents_per_day": int(docs_per_day),
            "chunks_per_document": float(chunks),
            "embedding_chunks_per_second_per_worker": float(throughput),
            "indexing_chunks_per_second": float(indexing_rate),
            "workers": int(workers),
        },
    }
    status, body = api_post("/v1/scaling/capacity", json=payload)
    if status != 200 or not isinstance(body, dict):
        st.error(f"Capacity model failed ({status}): {body}")
        return

    storage = body.get("storage") or {}
    components = storage.get("components") or []
    if components:
        rows = [
            {
                "component": component.get("label"),
                "size": _human_bytes(component.get("bytes")),
                "basis": component.get("basis"),
                "note": component.get("note"),
            }
            for component in components
        ]
        st.dataframe(rows, hide_index=True, use_container_width=True)
        assumed = [row["component"] for row in rows if row.get("basis") == "assumed"]
        if assumed:
            st.caption(
                "Assumed (not measured) components: "
                + ", ".join(str(name) for name in assumed)
                + "."
            )
    totals = [
        ("primary", storage.get("primary_bytes")),
        ("replicas", storage.get("replica_bytes")),
        ("backups", storage.get("backup_bytes")),
        ("reindex working set", storage.get("reindex_bytes")),
        ("operational headroom", storage.get("headroom_bytes")),
        ("TOTAL", storage.get("total_bytes")),
    ]
    st.markdown("**Totals**")
    st.dataframe(
        [{"line": name, "size": _human_bytes(value)} for name, value in totals],
        hide_index=True,
        use_container_width=True,
    )

    ingestion = body.get("ingestion")
    if isinstance(ingestion, dict):
        st.markdown("**Ingestion**")
        message = (
            f"chunks/day {ingestion.get('chunks_per_day'):,.0f} • "
            f"bottleneck {ingestion.get('bottleneck_stage')} "
            f"{ingestion.get('bottleneck_per_day'):,.0f} chunks/day • "
            f"utilization {ingestion.get('utilization'):.2f}"
        )
        if ingestion.get("keeps_up"):
            st.info(
                "Under these assumptions the configuration would keep up — a "
                "conditional projection, not a demonstrated production capability. "
                + message
            )
        else:
            st.warning("Projected backlog under these assumptions. " + message)
        st.caption(
            f"effective workers {ingestion.get('effective_workers'):.2f} "
            f"(non-linear scaling) • throughput basis '{ingestion.get('basis')}'"
        )
        if ingestion.get("basis") == "assumed":
            st.caption(
                "⛑️ The throughput values above are illustrative assumptions. Supply a "
                "measured embedding/indexing rate to make this a measurement-backed "
                "projection."
            )
        for note in ingestion.get("notes", []):
            st.caption(f"• {note}")

    scenarios = body.get("scenarios")
    if isinstance(scenarios, dict) and scenarios:
        st.markdown("**Document-size scenarios**")
        rows = []
        for name, estimate in scenarios.items():
            rows.append(
                {
                    "scenario": name,
                    "chunks/doc": (estimate.get("assumptions") or {}).get(
                        "chunks_per_document"
                    ),
                    "total": _human_bytes(estimate.get("total_bytes")),
                }
            )
        st.dataframe(rows, hide_index=True, use_container_width=True)


def _render_scaling_architecture() -> None:
    status, body = api_get("/v1/scaling/architecture")
    if status != 200 or not isinstance(body, dict):
        st.error(f"Could not load architecture ({status}): {body}")
        return
    st.caption(
        "Design data shared by this page and `docs/scaling_to_10m.md`, so the UI and "
        "the document state the same migration story."
    )
    left, right = st.columns(2)
    with left:
        st.markdown("**Current implementation** (measured)")
        st.code(body.get("current_diagram", ""), language="text")
    with right:
        st.markdown("**Proposed 10M-document architecture** (design, not built)")
        st.code(body.get("proposed_diagram", ""), language="text")

    for row in body.get("rows", []):
        with st.expander(f"{row.get('area')} — {row.get('current')} → {row.get('proposed')}"):
            st.markdown(f"**Why replaced.** {row.get('reason')}")
            st.markdown(f"**Interface preserved.** {row.get('interface')}")
            st.markdown(f"**Trade-off.** {row.get('tradeoff')}")
            st.markdown(f"**Still unknown.** {row.get('unknowns')}")


def render_scalability_lab() -> None:
    st.header("Scalability Laboratory")
    st.caption(
        "How the local prototype behaves as data grows, and the design required to "
        "reach ~10 million documents. The prototype does not serve that scale; the "
        "measurements are real and the design is explicitly a plan."
    )
    status, cfg = api_get("/v1/config")
    if status == 200 and isinstance(cfg, dict):
        st.caption(
            f"recorded scaling benchmarks {cfg.get('recorded_scaling_benchmarks')} • "
            f"document cap {cfg.get('scaling_max_documents')} • "
            f"dense available {cfg.get('dense_available')}"
        )
    tab_baselines, tab_capacity, tab_architecture = st.tabs(
        ["Measured baselines", "Capacity explorer", "Architecture explorer"]
    )
    with tab_baselines:
        _render_scaling_baselines()
    with tab_capacity:
        _render_scaling_capacity()
    with tab_architecture:
        _render_scaling_architecture()


PAGES = {
    "Status": render_status,
    "Corpus Manager": render_corpus,
    "Search Playground": render_search,
    "Mode Comparison": render_compare,
    "Chunking Laboratory": render_chunking_lab,
    "Reranking Laboratory": render_reranking_lab,
    "Answer Laboratory": render_answer_lab,
    "Evaluation Lab": render_evaluation_lab,
    "Failure Explorer": render_failure_explorer,
    "Performance Lab": render_performance_lab,
    "Scalability Laboratory": render_scalability_lab,
}


def main() -> None:
    global API_URL, READ_ONLY
    global LAB_EVAL_ENABLED, LAB_GENERATION_ENABLED, LAB_INGESTION_ENABLED
    API_URL = resolve_api_url()
    READ_ONLY = server_is_read_only(API_URL)
    seed_note = None if READ_ONLY else seed_sample_if_empty(API_URL)

    config_status, config_body = api_get("/v1/config")
    config = config_body if config_status == 200 and isinstance(config_body, dict) else {}
    for key in ("lab_eval_enabled", "lab_generation_enabled", "lab_ingestion_enabled"):
        st.session_state.setdefault(key, False)

    st.sidebar.title("RAG Workbench")
    st.sidebar.caption(
        "BM25 • dense (Sentence Transformers) • hybrid RRF • clause-aware chunking "
        "• cross-encoder reranking • retrieval evaluation • grounded answers "
        "• scalability baselines & design"
    )
    if READ_ONLY:
        st.sidebar.info(
            "Read-only demo: ingestion, new evaluation runs and live generation are "
            "disabled."
        )
    if seed_note:
        st.sidebar.caption(seed_note)
    render_laboratory_controls(config)
    st.session_state["owner_session"] = _owner.render_owner_mode(API_URL, config)

    laboratory = (config.get("laboratory") or {}) if config else {}
    LAB_EVAL_ENABLED = bool(
        laboratory.get("evaluation", {}).get("permitted")
    ) and bool(st.session_state.get("lab_eval_enabled"))
    LAB_GENERATION_ENABLED = bool(
        laboratory.get("generation", {}).get("permitted")
    ) and bool(st.session_state.get("lab_generation_enabled"))
    LAB_INGESTION_ENABLED = bool(
        laboratory.get("ingestion", {}).get("permitted")
    ) and bool(st.session_state.get("lab_ingestion_enabled"))

    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.title("Retrieval Laboratory")
    PAGES[page]()


main()
