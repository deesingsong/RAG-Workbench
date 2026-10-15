"""Streamlit "Retrieval Laboratory" — a thin HTTP client of the FastAPI backend.

This module performs no retrieval, ingestion or evaluation logic itself: every
action is an HTTP request to the API, so the UI and the API can never disagree
about how the system works. Pages: search and mode comparison, the Chunking,
Reranking, Evaluation and Performance Laboratories, a Failure Explorer and the
system/corpus views.
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


def server_is_read_only(api_url: str) -> bool:
    """Probe the deployment's read-only flag, tolerating a stale module."""

    probe = getattr(_backend, "server_is_read_only", None)
    return bool(probe(api_url)) if probe is not None else False


def on_disk_version() -> str:
    """Read the package version from the source tree, tolerating a stale module."""

    probe = getattr(_backend, "on_disk_version", None)
    return probe() if probe is not None else "unknown"

# Replaced at startup by resolve_api_url(). The default is the separately-run
# local API; a single-process host starts the API in-process instead.
API_URL = "http://127.0.0.1:8000"
READ_ONLY = False
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


def render_corpus() -> None:
    st.header("Corpus manager")
    if READ_ONLY:
        st.info(
            "Read-only demo: document ingestion and index rebuilds are disabled. "
            "The bundled sample corpus is seeded by the backend itself at startup."
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
    """Show the resolved asset locations so a misconfigured deploy is obvious."""

    status, cfg = api_get("/v1/config")
    if status != 200 or not isinstance(cfg, dict):
        return
    with st.expander("Deployment diagnostics (read-only)"):
        st.caption(
            f"repo_root {cfg.get('repo_root')} • read_only {cfg.get('read_only')} • "
            f"autoseed {cfg.get('autoseed_sample')} • dense_available {cfg.get('dense_available')}"
        )
        st.caption(
            f"datasets {cfg.get('bundled_datasets_dir')} "
            f"(exists={cfg.get('bundled_datasets_exists')}) • "
            f"recorded {cfg.get('recorded_experiments_dir')} "
            f"(exists={cfg.get('recorded_experiments_exists')})"
        )
        backend_version = cfg.get("version")
        disk_version = on_disk_version()
        st.caption(f"backend version {backend_version} • source tree version {disk_version}")
        if disk_version != "unknown" and backend_version and backend_version != disk_version:
            st.warning(
                f"Version mismatch: the API reports {backend_version} but the source "
                f"tree is {disk_version}. A full app restart may be required to load "
                "the current backend."
            )


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


PAGES = {
    "Status": render_status,
    "Corpus Manager": render_corpus,
    "Search Playground": render_search,
    "Mode Comparison": render_compare,
    "Chunking Laboratory": render_chunking_lab,
    "Reranking Laboratory": render_reranking_lab,
    "Evaluation Lab": render_evaluation_lab,
    "Failure Explorer": render_failure_explorer,
    "Performance Lab": render_performance_lab,
}


def main() -> None:
    global API_URL, READ_ONLY
    API_URL = resolve_api_url()
    READ_ONLY = server_is_read_only(API_URL)
    seed_note = None if READ_ONLY else seed_sample_if_empty(API_URL)

    st.sidebar.title("RAG Workbench")
    st.sidebar.caption(
        "BM25 • dense (Sentence Transformers) • hybrid RRF • clause-aware chunking "
        "• cross-encoder reranking • retrieval evaluation"
    )
    if READ_ONLY:
        st.sidebar.info("Read-only demo: ingestion and new evaluation runs are disabled.")
    if seed_note:
        st.sidebar.caption(seed_note)
    page = st.sidebar.radio("Navigate", list(PAGES.keys()))
    st.title("Retrieval Laboratory")
    PAGES[page]()


main()
