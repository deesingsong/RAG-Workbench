# RAG Workbench

A local-first, zero-mandatory-cost **RAG retrieval engineering laboratory**. The
goal is to make retrieval itself inspectable: ingest documents, chunk them
deterministically, index them in SQLite, run lexical (BM25) and dense retrieval
(or a hybrid of both), and read back provenance-rich citations you can verify
against the source.

This is an engineering research platform, not a "chat with your PDFs" clone. In
this repository there is **no LLM, no API key, no GPU and no Docker requirement**
for anything documented below.

> **Status: Phases 0–3 complete.**
> Implemented: project foundation, ingestion (TXT / Markdown / text PDF),
> deterministic chunking (`fixed_token`, `paragraph`, **`clause_aware`**), SQLite
> metadata + FTS5 indexing, BM25 search, local **Sentence Transformers
> embeddings**, a **Qdrant-local vector index**, **dense search**, **RRF hybrid
> search**, vector index status / rebuild / reconciliation, optional local
> **cross-encoder reranking**, a FastAPI backend, a CLI and a thin Streamlit
> client.
> **Not implemented yet (later phases):** evaluation metrics, the RAG failure
> explorer, and optional Ollama answer generation. Nothing in this README claims
> those exist.

---

## 1. What works today

- **Ingestion** of `.txt`, `.md` / `.markdown` and text-extractable `.pdf`
  documents, with graceful failure for malformed or empty input.
- **Deterministic chunking** with three strategies, configuration-hashed into
  isolated *index profiles* (see §6 and §9).
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
- **Clause-aware legal chunking** that preserves section boundaries and a
  hierarchical `section_path` (e.g. `ARTICLE IV > 4.2 Limitation of Liability >
  4.2(a) ...`), records `section_number` / `section_title` / `parent_clause` /
  `section_level` when they can be derived, groups the paragraphs of one clause,
  splits long clauses without losing their parent identity, and flags chunks that
  carry a split or heading-only diagnostic.
- **Optional local cross-encoder reranking** over the first-stage candidate pool
  (`rerank=true`, configurable `rerank_candidates`), which reorders — never adds —
  candidates and returns the original rank/score alongside the rerank score and
  the rank movement, with model initialization and inference timed separately.
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
once (about 90 MB). **Cross-encoder reranking** needs the same `dense` extra and
downloads the reranker model once on first use (about 90 MB). No Docker, cloud
account, API key, GPU or Ollama is needed.

---

## 3. Setup (Windows PowerShell)

```powershell
# 1. Install uv if you do not have it
#    (see https://docs.astral.sh/uv/getting-started/installation/)

# 2. From the repository root, create the environment and install core dependencies
uv sync

# 2b. OPTIONAL: add dense retrieval and cross-encoder reranking
#     (Sentence Transformers + torch, CPU build)
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

# Clause-aware legal chunking and clause inspection
uv run rag-workbench ingest tests/fixtures/legal_contract.txt --collection contracts --strategy clause_aware
uv run rag-workbench chunks --limit 50

# Cross-encoder reranking over a first-stage candidate pool
uv run rag-workbench search "confidential information" --mode bm25 --top-k 3 --rerank --rerank-candidates 20
uv run rag-workbench search "confidential information" --mode bm25 --top-k 3 --rerank --reranker-model cross-encoder/ms-marco-MiniLM-L6-v2
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

Example rerank output (real cross-encoder, clause-aware profile, see §13):

```
  #1 rerank=10.227 ... 1.1 "Confidential Information" [retrieval_rank=1 movement=0]
  #2 rerank=5.505  ... 4.1 Confidentiality            [retrieval_rank=3 movement=+1]
  #3 rerank=1.919  ... 3.3(b) Return of Materials     [retrieval_rank=2 movement=-1]
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
   `dense` extra is **not** installed and the auto-seeded profiles
   (`paragraph` and `clause_aware`) are lexical-only, so the live demo runs on
   BM25: selecting `dense`/`hybrid` returns a clear `422 profile_not_dense` (a
   degenerate dense profile would instead return `503 dense_unavailable`), and
   the Reranking Laboratory returns a clear `503 reranker_unavailable`.

Cloud caveats: the filesystem is **ephemeral** — the SQLite index and vector
store reset when the app sleeps or reboots (hence the auto-seed, which also
fills in a missing demonstration profile after a code upgrade) — and the
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
| `RAG_WORKBENCH_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Local cross-encoder for reranking. |
| `RAG_WORKBENCH_RERANKER_REVISION` | unset | Optional reranker model revision. |
| `RAG_WORKBENCH_RERANKER_BATCH_SIZE` | `32` | Cross-encoder inference batch size. |
| `RAG_WORKBENCH_RERANKER_MAX_LENGTH` | derived | Optional max sequence length override. |
| `RAG_WORKBENCH_RERANK_CANDIDATES` | `20` | Default candidate depth to rerank. |
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
- `Reranker`: query + passages → cross-encoder scores with explicit model
  identity; `ensure_loaded()` reports one-off initialization time separately from
  inference. Implemented by the Sentence Transformers cross-encoder in production
  and a deterministic fake in tests.

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

When reranking runs, `rank` is the **final** (post-rerank) position and each hit
also carries `retrieval_rank`, `retrieval_score`, `retrieval_score_kind`,
`rerank_score` and `rank_movement` (`retrieval_rank - rank`; positive means the
passage moved up). `rerank_score` is a cross-encoder **relevance logit** — larger
is more relevant, but it is **not** a calibrated probability of relevance. Ties
are broken deterministically by original retrieval rank and then by chunk ID.

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

Phase 2 added one column (`index_profiles.embedding_params`) and Phase 3 adds
clause provenance columns to `chunks` (`section_number`, `section_title`,
`parent_clause`, `section_level`, `diagnostics`). Schema creation applies
**additive** `ALTER TABLE ... ADD COLUMN` migrations, so a Phase 1 or Phase 2
database keeps every document, chunk and profile (covered by tests).

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
| `POST` | `/v1/search` | `{query, profile_id, mode, top_k, collection?, candidates?, rerank?, reranker_model?, rerank_candidates?}` → provenance-rich hits. |
| `GET` | `/v1/profiles` | Index profiles with config hashes and embedding identity. |
| `GET` | `/v1/profiles/{id}/chunks?document_id=&limit=&offset=` | Stored chunks with clause provenance and diagnostics. |
| `GET` | `/v1/vectors/status?profile_id=` | Reconcile vector points against SQLite chunks. |
| `POST` | `/v1/vectors/rebuild` | `{profile_id?}` or `{all_profiles: true}` → rebuild from SQLite. |
| `POST` | `/v1/vectors/reconcile` | `{profile_id, repair}` → detect and optionally repair drift. |

`mode` accepts `bm25`, `dense` and `hybrid`. Errors are actionable: `415`
unsupported type, `413` too large, `400` parse or search failure, `404` missing
document/profile, `422` invalid query/config/unsupported mode, `503` dense or
reranker dependencies being unavailable (`dense_unavailable`,
`vector_index_not_built`, `reranker_unavailable`). Body shape:
`{"error": "<code>", "message": "<text>"}`.

Reranking is optional and off by default. `rerank=true` retrieves
`rerank_candidates` (default 20) candidates and reorders them, returning the top
`top_k`; `rerank_candidates` must be `>= top_k` (else `422 invalid_candidates`).

---

## 8. Testing and quality

```powershell
uv run ruff check .
uv run pytest                 # core suite; dense/real-model tests are deselected
uv run pytest -m dense -s     # opt-in: real Sentence Transformers smoke test
uv run pytest -m reranker -s  # opt-in: real cross-encoder reranker smoke test
```

The core suite is deterministic and runs **offline** with no model downloads. The
dense/hybrid/rerank paths are exercised end-to-end with a deterministic fake
embedding provider and fake reranker over the **real** Qdrant local store and the
real RRF implementation:

- **Unit:** tokenizer and chunk boundaries; parsers (headings, encoding fallback,
  empty input, malformed and image-only PDFs, unsupported extensions); the
  clause-marker detector and clause-aware chunker (hierarchy, grouping, long-clause
  splitting, definition grouping, overlap bounds, determinism, no content loss);
  the untrusted-input FTS query builder; configuration and profile identity
  (including the Phase-1 hash lock and clause-aware isolation); Reciprocal Rank
  Fusion; the vector store (idempotent upsert, dimension checks, filtering,
  deletion, persistence across reopen); embedding laziness and dependency
  handling; reranker wiring and laziness.
- **Integration:** contract ingestion → exact-clause search with provenance;
  idempotent re-ingestion; profile isolation; deletion removing relational rows,
  FTS entries **and** vector points; dense and hybrid search with preserved
  component ranks; `partial` recovery and rebuild; vector reconciliation of
  missing and orphan points; clause ingestion + the chunks endpoint; reranking
  across bm25/dense/hybrid with rank movement, tie-breaking, `rerank_candidates`
  validation, truncation diagnostics and `reranker_unavailable`; the Phase-1 → 2
  and Phase-2 → 3 additive schema migrations; the full HTTP round-trip with the
  same error mapping.
- **Dense (opt-in, `-m dense`):** the real `all-MiniLM-L6-v2` model builds a dense
  profile for the sample contract and returns the expected top clause.
- **Reranker (opt-in, `-m reranker`):** the real
  `cross-encoder/ms-marco-MiniLM-L6-v2` model reranks a real candidate list,
  keeping retrieval and rerank scores separate and reporting its own timings.

Run the tests to see the current numbers on your machine; this README does not
pin a pass count.

---

## 9. Chunking strategies

Three strategies are implemented:

- **`fixed_token`** — sliding token windows of `max_tokens` with `overlap_tokens`
  overlap. Windows never cross a parsed block boundary, so page and section
  provenance stay exact.
- **`paragraph`** (default) — recursive/paragraph-aware packing: whole paragraphs
  are packed up to the budget, a paragraph that alone exceeds the budget is
  split with fixed windows, and a bounded trailing-token overlap is carried into
  the next chunk. Chunks never span a section boundary.
- **`clause_aware`** — structure-aware chunking for English legal documents. It
  recognises numbered sections (`2.1`, `2.1.3`), `(a)` subsections, `ARTICLE IV`,
  `SCHEDULE` / `ANNEX` / `APPENDIX` / `EXHIBIT` headings and definition entries
  (`"Term" means ...`); groups the paragraphs that belong to one clause while they
  fit the budget; splits a long clause with fixed windows while keeping its parent
  identity; carries a bounded overlap only *within* the same clause; and records
  `section_number`, `section_title`, `parent_clause` and `section_level` when they
  can be derived. A heading line that would otherwise become a heading-only chunk
  is merged with the content that follows; if it cannot be, it is kept with a
  diagnostic rather than dropped. Detection is conservative (a stray number, date,
  amount or lowercase continuation is not a heading), deterministic and
  model-free — it is a structural **heuristic**, not a contract interpreter, and
  unrecognised text is never discarded.

`clause_aware` has its own profile identity (the strategy is part of the config
hash), so it shares no chunks or vectors with `paragraph`/`fixed_token`.

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
  (one time, cached by Hugging Face); the first reranked search downloads the
  cross-encoder (one time). Loading either model is the dominant first-call
  latency; the API keeps them warm in-process, while each CLI invocation reloads
  them, so CLI timings include model load. `timings_ms.rerank_init` reports the
  one-off initialization separately from steady-state `rerank_inference`.
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
  The `clause_aware` chunker adds its own conservative marker detection on top of
  the parser but is still a heuristic.
- **`section_path` is not OCR'd from layout** and page numbers are unknown
  (`None`) for plain text, where the concept does not apply.
- **Reranker scores are logits, not probabilities.** A cross-encoder produces an
  unbounded relevance logit; it orders candidates well but is not a calibrated
  confidence and should not be read as a percentage. The default reranker is a
  small English MS MARCO model, so non-English and highly technical clauses are
  out of its training distribution.
- **Migrations are additive only.** New nullable columns are added automatically;
  destructive or type-changing migrations are not yet handled (no Alembic).
- **Uploads are processed in memory** up to the size limit; there is no
  streaming ingest path yet.
- **No evaluation metrics or answer generation yet** — those are later phases.
  `/v1/search` rejects unknown `mode` values with a clear message rather than
  pretending to support them.

---

## 11. Repository layout

```text
src/rag_workbench/
  config.py            Settings + index-profile identity
  embeddings.py        Sentence Transformers provider (lazy) + factory
  reranking.py         Cross-encoder reranker (lazy) + factory
  logging_config.py    Structured JSON logging
  cli.py               rag-workbench CLI
  domain/              schemas, Protocol interfaces, typed errors
  persistence/         SQLAlchemy models, engine, FTS5 schema, vector store
  ingestion/           parsers, chunkers (incl. clause-aware), pipeline, vector indexer
  retrieval/           BM25, dense, hybrid (RRF), retrieval service (+ rerank stage)
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
- **Phase 3 (done):** clause-aware legal chunker, local cross-encoder reranking,
  rank-comparison UI (Chunking and Reranking Laboratories), reranker latency
  reporting.
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

## 14. Observed Phase 3 results (legal fixture)

Measured on Windows with the synthetic `tests/fixtures/legal_contract.txt`,
`cross-encoder/ms-marco-MiniLM-L6-v2`, query `confidential information`. These
are single-machine smoke observations, not a benchmark.

- **Clause-aware chunking** (`max_tokens=120`, `overlap_tokens=20`) produces 18
  chunks with a preserved hierarchy, e.g.
  `ARTICLE III > 3.3 Consequences of Termination > 3.3(a) Accrued Fees`, and
  `ARTICLE IV > 4.1 Confidentiality` (the parser does not itself encode `ARTICLE`
  headings; the chunker recovers them as parents).
- **First-stage BM25** ranked `1.1 "Confidential Information"` (1),
  `3.3(b) Return of Materials` (2) and `4.1 Confidentiality` (3).
- **After cross-encoder rerank** the cross-encoder swapped `4.1` above `3.3(b)`:
  `1.1` (rerank `10.227`, movement `0`), `4.1` (rerank `5.505`, movement `+1`),
  `3.3(b)` (rerank `1.919`, movement `-1`).
- **Timings:** `rerank_init` was `2877.9 ms` on the first call (this includes the
  one-time model download/load) and `rerank_inference` was `130.1 ms` for three
  `(query, passage)` pairs; `lexical` was `0.4 ms`. The API keeps the loaded model
  warm, so later reranked queries pay inference only.

---

## License

MIT. The bundled `tests/fixtures/sample_contract.txt` and
`tests/fixtures/legal_contract.txt` are synthetic, redistributable documents
written for this project.
