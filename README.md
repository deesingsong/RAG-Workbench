# RAG Workbench

A local-first, zero-mandatory-cost **RAG retrieval engineering laboratory**. The
goal is to make retrieval itself inspectable: ingest documents, chunk them
deterministically, index them in SQLite, run lexical (BM25) and dense retrieval
(or a hybrid of both), and read back provenance-rich citations you can verify
against the source.

This is an engineering research platform, not a "chat with your PDFs" clone. In
this repository there is **no LLM, no API key, no GPU and no Docker requirement**
for anything documented below.

> **Status: Phase 0 + Phase 1 + Phase 2 complete.**
> Implemented: project foundation, ingestion (TXT / Markdown / text PDF),
> deterministic chunking (`fixed_token`, `paragraph`), SQLite metadata + FTS5
> indexing, BM25 search, local **Sentence Transformers embeddings**, a
> **Qdrant-local vector index**, **dense search**, **RRF hybrid search**, vector
> index status / rebuild / reconciliation, a FastAPI backend, a CLI and a thin
> Streamlit client.
> **Not implemented yet (later phases):** cross-encoder reranking, evaluation
> metrics, the RAG failure explorer, and optional Ollama answer generation.
> Nothing in this README claims those exist.

---

## 1. What works today

- **Ingestion** of `.txt`, `.md` / `.markdown` and text-extractable `.pdf`
  documents, with graceful failure for malformed or empty input.
- **Deterministic chunking** with two strategies, configuration-hashed into
  isolated *index profiles* (see §6).
- **Persistent index** in a single SQLite file: relational metadata plus an
  FTS5 virtual table kept synchronized by database triggers.
- **BM25 retrieval** with a single consistent rank convention, provenance-rich
  hits (filename, page range where extractable, section path where extractable)
  and measured stage timings.
- **Dense retrieval** with a local Sentence Transformers model, persisted in a
  Qdrant *local-mode* collection that is keyed to the immutable index profile.
- **Hybrid retrieval** using Reciprocal Rank Fusion, preserving the original
  BM25 and dense ranks and scores on every hit.
- **Vector index management**: status, rebuild-from-SQLite, and reconciliation of
  missing points / orphan points, so an interrupted embedding job is recoverable
  without re-uploading documents.
- **Idempotent re-ingestion**: the same bytes into the same profile are reported
  as a duplicate; vector points are keyed by the stable chunk ID and overwritten.
- **FastAPI** endpoints with typed Pydantic contracts and OpenAPI docs, a
  **CLI**, and a **Streamlit** client that talks to the API over HTTP only.

---

## 2. Requirements

- **Python 3.12** (3.11 is likely fine but is not tested here).
- **uv** — https://docs.astral.sh/uv/ — for environment and dependency
  management.
- Windows PowerShell is the documented shell; the commands are equivalent on
  macOS/Linux.

**BM25-only use needs no model, no download and no network.** Dense and hybrid
retrieval need the optional `dense` extra (§3) and download the embedding model
once (about 90 MB). No Docker, cloud account, API key, GPU or Ollama is needed.

---

## 3. Setup (Windows PowerShell)

```powershell
# 1. Install uv if you do not have it
#    (see https://docs.astral.sh/uv/getting-started/installation/)

# 2. From the repository root, create the environment and install core dependencies
uv sync

# 2b. OPTIONAL: add dense retrieval (Sentence Transformers + torch, CPU build)
uv sync --extra dense

# 3. Optional: copy the example configuration
Copy-Item .env.example .env

# 4. Create the SQLite schema + FTS5 index
uv run rag-workbench init-db
```

`uv sync` creates an isolated `.venv` from `pyproject.toml` / `uv.lock`. The core
install deliberately excludes deep-learning packages so that BM25 and the test
suite stay light and offline.

---

## 4. Run it

Start the API (first terminal):

```powershell
uv run rag-workbench serve
# or, equivalently:
uv run uvicorn --factory rag_workbench.api.app:create_app --reload
```

API docs: http://127.0.0.1:8000/docs — health check: http://127.0.0.1:8000/health

Start the UI (second terminal):

```powershell
uv run streamlit run ui/streamlit_app.py
```

### Ingest and search from the CLI

```powershell
# Lexical (BM25) profile
uv run rag-workbench ingest tests/fixtures/sample_contract.txt --collection contracts
uv run rag-workbench search "termination for convenience" --top-k 5

# Dense profile: creates embeddings on ingest (downloads the model on first use)
uv run rag-workbench ingest tests/fixtures/sample_contract.txt --collection contracts --embed

# Three retrieval modes over the same query
uv run rag-workbench search "termination for convenience" --mode bm25   --top-k 3
uv run rag-workbench search "termination for convenience" --mode dense  --top-k 3
uv run rag-workbench search "termination for convenience" --mode hybrid --top-k 3

# Inspect the vector index and repair it if needed
uv run rag-workbench vectors-status
uv run rag-workbench rebuild-vectors --all
```

Example dense output (real MiniLM run, see §13):

```
query='termination for convenience' mode=dense profile=p-fd18e21e2a5b3664 hits=3
  #1 cosine=0.4360 sample_contract.txt | 3. Term and Termination > 3.1 Termination for Convenience
  #2 cosine=0.4343 sample_contract.txt | 3. Term and Termination > 3.2 Termination for Cause
  #3 cosine=0.1460 sample_contract.txt | 2. Fees and Payment > 2.1 Payment Terms
```

Example hybrid output (component ranks preserved):

```
  #1 rrf=0.0328 sample_contract.txt | ... 3.1 Termination for Convenience [bm25_rank=1 dense_rank=1]
  #2 rrf=0.0161 sample_contract.txt | ... 3.2 Termination for Cause         [bm25_rank=None dense_rank=2]
```

### Try a PDF

```powershell
uv run rag-workbench ingest "C:\path\to\contract.pdf" --collection contracts --embed
```

### Deploying to Streamlit Community Cloud

The repo is set up to run as a **single Streamlit app** there, because Community
Cloud runs one Streamlit process and cannot run a separate FastAPI process:

- When `RAG_WORKBENCH_API_URL` is unset, `ui/local_backend.py` starts the *same*
  FastAPI app with uvicorn on `127.0.0.1:8765` inside the Streamlit process; the
  UI talks to it over HTTP. No retrieval logic moves into the UI.
- On an empty (ephemeral) instance the first load ingests the bundled sample
  contract so there is something to search immediately; set
  `RAG_WORKBENCH_AUTOSEED=0` to disable.

To deploy:

1. Push to the repo's `main` branch. Streamlit Cloud **auto-redeploys on every
   push**, so each completed phase appears in the live app.
2. On <https://share.streamlit.io> → *New app* → repository
   `deesingsong/RAG-Workbench`, branch `main`, **Main file path
   `ui/streamlit_app.py`**.
3. Set the Python version to **3.12**. Dependencies come from
   `uv.lock` / `pyproject.toml` (Community Cloud supports uv). The optional
   `dense` extra is **not** installed, so the live demo runs on BM25 and
   dense/hybrid return a clear `503` until a dense profile is built.

Cloud caveats: the filesystem is **ephemeral** — the SQLite index and vector
store reset when the app sleeps or reboots (hence the auto-seed) — and the
deep-learning dense stack is heavy for the free tier, so BM25-only is the
default live demo.

---

## 5. Configuration

All settings are environment variables prefixed `RAG_WORKBENCH_` (see
`.env.example`). Sensible defaults mean an empty environment works.

| Variable | Default | Meaning |
|---|---|---|
| `RAG_WORKBENCH_DATA_DIR` | `data` | Local state directory (SQLite DB + vector store). Git-ignored. |
| `RAG_WORKBENCH_DATABASE_URL` | SQLite file in `DATA_DIR` | Explicit SQLAlchemy URL. |
| `RAG_WORKBENCH_MAX_UPLOAD_MB` | `25` | Upload size limit. |
| `RAG_WORKBENCH_ALLOWED_EXTENSIONS` | `[".txt", ".md", ".markdown", ".pdf"]` | Accepted upload types (JSON array). |
| `RAG_WORKBENCH_LOG_LEVEL` | `INFO` | Structured JSON log level. |
| `RAG_WORKBENCH_DEFAULT_COLLECTION` | `default` | Default collection name. |
| `RAG_WORKBENCH_DEFAULT_CHUNK_STRATEGY` | `paragraph` | `paragraph` or `fixed_token`. |
| `RAG_WORKBENCH_DEFAULT_MAX_TOKENS` | `220` | Default token budget per chunk. |
| `RAG_WORKBENCH_DEFAULT_OVERLAP_TOKENS` | `40` | Default overlap tokens. |
| `RAG_WORKBENCH_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Default embedding model. |
| `RAG_WORKBENCH_EMBEDDING_DIMENSION` | derived (384 for MiniLM) | Vector dimension; required for unknown models. |
| `RAG_WORKBENCH_EMBEDDING_REVISION` | unset | Optional model revision. |
| `RAG_WORKBENCH_EMBEDDING_BATCH_SIZE` | `32` | Encode batch size. |
| `RAG_WORKBENCH_EMBEDDING_NORMALIZE` | `true` | L2-normalize embeddings. |
| `RAG_WORKBENCH_VECTOR_DIR_NAME` | `qdrant` | Vector store directory under `DATA_DIR`. |
| `RAG_WORKBENCH_DENSE_CANDIDATE_K` | `50` | Candidate pool for dense/hybrid. |
| `RAG_WORKBENCH_RRF_K` | `60` | Reciprocal Rank Fusion constant. |
| `RAG_WORKBENCH_API_URL` | `http://127.0.0.1:8000` | API base URL used by the Streamlit UI. |
| `RAG_WORKBENCH_INPROC_HOST` | `127.0.0.1` | Host for the in-process API on a single-process host. |
| `RAG_WORKBENCH_INPROC_PORT` | `8765` | Port for the in-process API (used when `RAG_WORKBENCH_API_URL` is unset). |
| `RAG_WORKBENCH_AUTOSEED` | `1` | Seed the bundled sample contract when the index is empty (`0` disables). |

---

## 6. Architecture

```text
              Streamlit UI (HTTP only)
                        |
                        v
                 FastAPI  /health  /v1/...
                        |
      +-----------------+--------------------------+
      |                                            |
 IngestionService                             SearchService
 (parse -> chunk -> persist -> embed)         (profile -> bm25 | dense | hybrid)
      |                                            |
      v                          +-----------------+------------------+
 SQLAlchemy models ---- SQLite --+                                    |
 (documents, index_profiles,   FTS5 `chunk_fts`        Qdrant local-mode collection
  chunks, ingestion_jobs)      (triggers)              `rw_<profile_key>`
```

The pipeline is explicit and framework-free:

```text
DocumentParser -> Chunker -> Document/Chunk rows -> FTS5 -> BM25 retrieval
                                        \
                                         -> EmbeddingProvider -> Qdrant -> dense search
                                                              (BM25 + dense) -> RRF -> hybrid
```

Interfaces (`src/rag_workbench/domain/interfaces.py`) are `Protocol`s and exist
only where multiple implementations exist or are planned:

- `DocumentParser`: bytes + filename → ordered `ParsedBlock`s with provenance.
- `Chunker`: `ParsedDocument` + `ChunkerConfig` → `ChunkData` list.
- `SparseRetriever`: query + profile → ranked `SearchHit`s (BM25).
- `EmbeddingProvider`: text → vectors with explicit model identity. Implemented by
  the Sentence Transformers provider in production and a deterministic fake in
  tests (the seam that keeps the test suite offline).
- `VectorStore`: named collections, upsert/search/delete/list/count. Implemented
  by the Qdrant local-mode store.

### Index profiles and isolation

An `IndexProfile` is identified by a **config hash** over behaviour-affecting
fields only. For a lexical profile that is parser version, chunker strategy/params,
null embedding model/dimension and index version — **byte-for-byte the Phase 1
identity**, so existing BM25 profiles are not invalidated (locked by a unit test).
Dense profiles additionally hash the embedding model, dimension, revision,
similarity metric, normalization and any document/query prefixes. The human name
is a label, not identity. Consequences:

- Re-ingesting into the same configuration reuses the same profile (idempotent).
- Different chunking or embedding configurations produce **separate, non-mixing**
  indexes; incompatible vector representations can never share a collection.
- Chunk IDs are `uuid5` values derived from `(profile_key, document_id, order)`,
  deterministic and profile-scoped; they double as Qdrant point IDs, so upserts
  are idempotent.
- The vector collection name is `rw_<profile_key>`, derived purely from identity.

### Rank and score conventions

Every hit exposes a 1-based `rank` where `1` is always best, plus a `score_kind`
documenting how to read `score`:

- `bm25` — FTS5 score; **smaller (more negative) is better**.
- `cosine` — embedding similarity; **larger is better**.
- `rrf` — fused score; **larger is better**.

Hybrid hits additionally carry `bm25_rank`, `bm25_score`, `dense_rank`,
`dense_score` and `fused_score`, so rank movements caused by fusion are
inspectable. Fusion combines **ranks only** (never raw scores), which is why it
never adds a BM25 score to a cosine similarity.

### FTS5 and vector synchronization

`chunk_fts` is an FTS5 **external-content** table backed by `chunks`
(`content_rowid = 'id'`). Three triggers (`INSERT`, `UPDATE`, `DELETE`) mirror
every row change, so the lexical index cannot drift from the relational rows
within a transaction.

The vector index is a *derived* structure kept in sync by the application:

- Embedding runs **after** the chunk write commits, so a long embedding job never
  holds the SQLite writer lock.
- If embedding or the vector store fails, the chunks stay durable and the job is
  recorded as **`partial`** (BM25-searchable) with a diagnostic — never silently
  reported as fully indexed.
- `vectors-status` compares SQLite chunks with vector points and reports
  `missing_points` and `orphan_points`; `rebuild-vectors` regenerates the whole
  collection purely from SQLite (no re-upload or re-parse).

### Schema migrations

Phase 2 adds one column (`index_profiles.embedding_params`). Schema creation
applies **additive** `ALTER TABLE ... ADD COLUMN` migrations, so a Phase 1
database keeps every document, chunk and profile (covered by a test).

---

## 7. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + counts. |
| `POST` | `/v1/documents` | Upload and ingest (multipart; `embed=true` builds a dense profile). |
| `GET` | `/v1/documents` | Paginated catalog. |
| `GET` | `/v1/documents/{id}` | Document detail. |
| `DELETE` | `/v1/documents/{id}?profile_id=` | Delete chunks (optionally one profile) and their vectors; drop the document when empty. |
| `GET` | `/v1/ingestion-jobs/{job_id}` | Job state, timings, diagnostics. |
| `POST` | `/v1/search` | `{query, profile_id, mode, top_k, collection?, candidates?}` → provenance-rich hits. |
| `GET` | `/v1/profiles` | Index profiles with config hashes and embedding identity. |
| `GET` | `/v1/vectors/status?profile_id=` | Reconcile vector points against SQLite chunks. |
| `POST` | `/v1/vectors/rebuild` | `{profile_id?}` or `{all_profiles: true}` → rebuild from SQLite. |
| `POST` | `/v1/vectors/reconcile` | `{profile_id, repair}` → detect and optionally repair drift. |

`mode` accepts `bm25`, `dense` and `hybrid`. Errors are actionable: `415`
unsupported type, `413` too large, `400` parse or search failure, `404` missing
document/profile, `422` invalid query/config/unsupported mode, `503` dense
dependencies or the vector index being unavailable (`dense_unavailable`,
`vector_index_not_built`). Body shape: `{"error": "<code>", "message": "<text>"}`.

---

## 8. Testing and quality

```powershell
uv run ruff check .
uv run pytest                 # core suite; dense/real-model tests are deselected
uv run pytest -m dense -s     # opt-in: real Sentence Transformers smoke test
```

The core suite is deterministic and runs **offline** with no model downloads. The
dense/hybrid paths are exercised end-to-end with a deterministic fake embedding
provider over the **real** Qdrant local store and the real RRF implementation:

- **Unit:** tokenizer and chunk boundaries; parsers (headings, encoding fallback,
  empty input, malformed and image-only PDFs, unsupported extensions); the
  untrusted-input FTS query builder; configuration and profile identity
  (including the Phase-1 hash lock); Reciprocal Rank Fusion; the vector store
  (idempotent upsert, dimension checks, filtering, deletion, persistence across
  reopen); embedding laziness and dependency handling.
- **Integration:** contract ingestion → exact-clause search with provenance;
  idempotent re-ingestion; profile isolation; deletion removing relational rows,
  FTS entries **and** vector points; dense and hybrid search with preserved
  component ranks; `partial` recovery and rebuild; vector reconciliation of
  missing and orphan points; the Phase-1 → Phase-2 schema migration; the full
  HTTP round-trip with the same error mapping.
- **Dense (opt-in, `-m dense`):** the real `all-MiniLM-L6-v2` model builds a dense
  profile for the sample contract and returns the expected top clause.

Run the tests to see the current numbers on your machine; this README does not
pin a pass count.

---

## 9. Chunking strategies

Two strategies are implemented:

- **`fixed_token`** — sliding token windows of `max_tokens` with `overlap_tokens`
  overlap. Windows never cross a parsed block boundary, so page and section
  provenance stay exact.
- **`paragraph`** (default) — recursive/paragraph-aware packing: whole paragraphs
  are packed up to the budget, a paragraph that alone exceeds the budget is
  split with fixed windows, and a bounded trailing-token overlap is carried into
  the next chunk. Chunks never span a section boundary.

### What "token" means here

Token counting uses a **regex word tokenizer** (`\w+` plus individual
punctuation marks). It is a real, deterministic unit that requires no download,
so chunk boundaries are reproducible and testable offline. It is **not** the
same as a BPE tokenizer such as GPT's `tiktoken` or a Llama tokenizer, so
token counts will differ from those models. The embedding model applies its own
tokenizer at encode time; inputs longer than the model's sequence limit are
counted and reported as a diagnostic rather than silently truncated without
notice. Character counts are never used as token counts.

---

## 10. Limitations and non-goals

- **SQLite, single writer.** Concurrency is limited to one writer process; there
  are no distributed transactions, job queue or public-facing authentication.
  This is a local prototype.
- **Vector store is single-owner.** Qdrant *local mode* takes an exclusive lock on
  its storage directory, so one process owns `data/qdrant` at a time. The process
  caches one client per path and closes it on shutdown. Running the API and a
  simultaneous ingest CLI against the same `DATA_DIR` is not supported; treat the
  running API as the owner and drive ingestion through it.
- **Model download and load cost.** The first dense ingest downloads the model
  (one time, cached by Hugging Face). Loading the model is the dominant dense
  latency; the API keeps it warm in-process, while each CLI invocation reloads it,
  so CLI dense timings include model load.
- **Dense indexing is in-process and sequential.** There is no background worker
  or queue; a very large corpus would block the request. Rebuild batches are
  bounded for memory, not run in parallel.
- **Cosine only.** The vector store is created with cosine distance; other
  metrics are rejected rather than silently ignored.
- **No OCR.** Scanned/image-only PDFs yield zero blocks plus a diagnostic; no
  text is invented. Table extraction is not attempted.
- **Heading detection is heuristic.** Markdown ATX headings and a conservative
  numbered-heading pattern (`2.1 Definitions`); it is not a universal contract
  parser. Ambiguous parses surface as `section_path = None` rather than a guess.
- **`section_path` is not OCR'd from layout** and page numbers are unknown
  (`None`) for plain text, where the concept does not apply.
- **Migrations are additive only.** New nullable columns are added automatically;
  destructive or type-changing migrations are not yet handled (no Alembic).
- **Uploads are processed in memory** up to the size limit; there is no
  streaming ingest path yet.
- **No reranking, evaluation metrics or answer generation yet** — those are
  later phases. `/v1/search` rejects unknown `mode` values with a clear message
  rather than pretending to support them.

---

## 11. Repository layout

```text
src/rag_workbench/
  config.py            Settings + index-profile identity
  embeddings.py        Sentence Transformers provider (lazy) + factory
  logging_config.py    Structured JSON logging
  cli.py               rag-workbench CLI
  domain/              schemas, Protocol interfaces, typed errors
  persistence/         SQLAlchemy models, engine, FTS5 schema, vector store
  ingestion/           parsers, chunkers, ingestion pipeline, vector indexer
  retrieval/           BM25, dense, hybrid (RRF), retrieval service
  api/                 FastAPI factory, dependencies, routes
ui/streamlit_app.py    Thin HTTP client
ui/local_backend.py    In-process API for single-process (Streamlit Cloud) deploys
tests/                 unit + integration tests, tiny redistributable fixtures
.github/workflows/ci.yml
```

---

## 12. Roadmap

- **Phase 2 (done):** local Sentence Transformers embeddings, Qdrant-local
  persistence, dense search, RRF hybrid, profile isolation and stage timings.
- **Phase 3:** clause-aware legal chunker, local cross-encoder reranking,
  rank-comparison UI, reranker latency reporting.
- **Phase 4:** BEIR SciFact sample + document-level qrels, metrics with unit
  tests, per-query RAG failure explorer.
- **Phase 5:** optional Ollama answer generation with validated, inspectable
  citations and abstention; portfolio polish and the scaling-to-10M design.

---

## 13. Observed Phase 2 results (sample contract)

Measured on Windows with the bundled `tests/fixtures/sample_contract.txt`
(8 paragraph chunks), `sentence-transformers/all-MiniLM-L6-v2` (384-d), Qdrant
local mode, query `termination for convenience`. These are single-machine smoke
observations, not a benchmark.

- **BM25:** 1 hit; top hit is the expected clause
  `3. Term and Termination > 3.1 Termination for Convenience` (bm25 `-3.651`).
- **Dense (cosine):** the same clause is rank 1 (`0.436`), followed by
  `3.2 Termination for Cause` (`0.434`) and `2.1 Payment Terms` (`0.146`); i.e.
  dense surfaces semantic neighbours that the exact-term BM25 query does not.
- **Hybrid (RRF, k=60):** the correct clause is rank 1 with
  `bm25_rank=1 dense_rank=1`; the dense-only neighbours follow at
  `bm25_rank=None`.
- **Vector index:** `in-sync`, `points=8`, `chunks=8`.
- **Timings:** `vector_index` dominated by one-time model load on first call;
  steady-state per-query `vector_search` was ~1–2.5 ms and fusion ~0.04 ms on this
  machine. BM25 `retrieval` was ~1–4 ms.

---

## License

MIT. The bundled `tests/fixtures/sample_contract.txt` is a synthetic,
redistributable document written for this project.
