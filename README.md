# RAG Workbench

A local-first, zero-mandatory-cost **RAG retrieval engineering laboratory**. The
goal is to make retrieval itself inspectable: ingest documents, chunk them
deterministically, index them in SQLite, run BM25 search, and read back
provenance-rich citations you can verify against the source.

This is an engineering research platform, not a "chat with your PDFs" clone. In
this repository there is **no LLM, no API key, no GPU and no Docker requirement**
for anything documented below.

> **Status: Phase 0 + Phase 1 complete.**
> Implemented: project foundation, ingestion (TXT / Markdown / text PDF),
> deterministic chunking (`fixed_token`, `paragraph`), SQLite metadata + FTS5
> indexing, BM25 search with citations, a FastAPI backend, a CLI and a thin
> Streamlit client.
> **Not implemented yet (later phases):** dense embeddings, Qdrant, hybrid RRF,
> cross-encoder reranking, evaluation metrics, the RAG failure explorer, and
> optional Ollama answer generation. Nothing in this README claims those exist.

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
- **Idempotent re-ingestion**: the same bytes into the same profile are reported
  as a duplicate instead of creating repeated chunks.
- **FastAPI** endpoints with typed Pydantic contracts and OpenAPI docs, a
  **CLI**, and a **Streamlit** client that talks to the API over HTTP only.

---

## 2. Requirements

- **Python 3.12** (3.11 is likely fine but is not tested here).
- **uv** — https://docs.astral.sh/uv/ — for environment and dependency
  management.
- Windows PowerShell is the documented shell; the commands are equivalent on
  macOS/Linux.

No Docker, cloud account, API key, Ollama install or model download is needed.

---

## 3. Setup (Windows PowerShell)

```powershell
# 1. Install uv if you do not have it
#    (see https://docs.astral.sh/uv/getting-started/installation/)

# 2. From the repository root, create the environment and install dependencies
uv sync

# 3. Optional: copy the example configuration
Copy-Item .env.example .env

# 4. Create the SQLite schema + FTS5 index
uv run rag-workbench init-db
```

`uv sync` creates an isolated `.venv` from `pyproject.toml` / `uv.lock`.

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

Ingest and search from the CLI:

```powershell
# Ingest the bundled sample contract
uv run rag-workbench ingest tests/fixtures/sample_contract.txt --collection contracts

# Search for an exact clause term
uv run rag-workbench search "termination for convenience" --top-k 5

# Inspect what is indexed
uv run rag-workbench documents
uv run rag-workbench profiles
```

Example search output:

```
query='termination for convenience' mode=bm25 profile=p-... (paragraph-220) hits=2 retrieval_ms=0.8
  #1 score=-1.2345 sample_contract.txt | 3. Term and Termination > 3.1 Termination for Convenience
     ...Either party may terminate this Agreement for convenience upon sixty (60) days...
```

### Try a PDF

```powershell
uv run rag-workbench ingest "C:\path\to\contract.pdf" --collection contracts
```

---

## 5. Configuration

All settings are environment variables prefixed `RAG_WORKBENCH_` (see
`.env.example`). Sensible defaults mean an empty environment works.

| Variable | Default | Meaning |
|---|---|---|
| `RAG_WORKBENCH_DATA_DIR` | `data` | Local state directory (SQLite DB). Git-ignored. |
| `RAG_WORKBENCH_DATABASE_URL` | SQLite file in `DATA_DIR` | Explicit SQLAlchemy URL. |
| `RAG_WORKBENCH_MAX_UPLOAD_MB` | `25` | Upload size limit. |
| `RAG_WORKBENCH_ALLOWED_EXTENSIONS` | `[".txt", ".md", ".markdown", ".pdf"]` | Accepted upload types (JSON array). |
| `RAG_WORKBENCH_LOG_LEVEL` | `INFO` | Structured JSON log level. |
| `RAG_WORKBENCH_DEFAULT_COLLECTION` | `default` | Default collection name. |
| `RAG_WORKBENCH_DEFAULT_CHUNK_STRATEGY` | `paragraph` | `paragraph` or `fixed_token`. |
| `RAG_WORKBENCH_DEFAULT_MAX_TOKENS` | `220` | Default token budget per chunk. |
| `RAG_WORKBENCH_DEFAULT_OVERLAP_TOKENS` | `40` | Default overlap tokens. |
| `RAG_WORKBENCH_API_URL` | `http://127.0.0.1:8000` | API base URL used by the Streamlit UI. |

---

## 6. Architecture

```text
              Streamlit UI (HTTP only)
                        |
                        v
                 FastAPI  /health  /v1/...
                        |
      +-----------------+------------------+
      |                                    |
 IngestionService                     SearchService
 (parse -> chunk -> persist)          (profile resolve -> BM25)
      |                                    |
      v                                    v
 SQLAlchemy models  ---  SQLite  ---  FTS5 `chunk_fts`
 (documents, index_profiles,          (external-content table,
  chunks, ingestion_jobs)             maintained by triggers)
```

The pipeline is explicit and framework-free:

```text
DocumentParser -> Chunker -> Document/Chunk rows -> FTS5 -> BM25 retrieval
```

Interfaces (`src/rag_workbench/domain/interfaces.py`) are `Protocol`s and exist
only where multiple implementations exist or are planned:

- `DocumentParser`: bytes + filename → ordered `ParsedBlock`s with provenance.
- `Chunker`: `ParsedDocument` + `ChunkerConfig` → `ChunkData` list.
- `SparseRetriever`: query + profile → ranked `SearchHit`s.

### Index profiles and isolation

An `IndexProfile` is identified by a **config hash** over behaviour-affecting
fields only — parser version, chunker strategy/params, embedding model/dimension
and index version. The human name is a label, not identity. Consequences:

- Re-ingesting into the same configuration reuses the same profile (idempotent).
- Different chunking configurations produce **separate, non-mixing** indexes.
- Chunk IDs are `uuid5` values derived from `(profile_key, checksum, order)`, so
  they are deterministic and profile-scoped.

### BM25 rank convention

SQLite FTS5's `bm25()` returns scores where **smaller is better**. The retriever
sorts ascending and exposes a 1-based `rank` where `1` is always best, with a
stable tie-break on the chunk's primary key. Callers never need to know the raw
direction.

### FTS5 synchronization

`chunk_fts` is an FTS5 **external-content** table backed by `chunks`
(`content_rowid = 'id'`). Three triggers (`INSERT`, `UPDATE`, `DELETE`) mirror
every row change, so the lexical index cannot drift from the relational rows
within a transaction. Deleting a document removes both.

---

## 7. API

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/health` | Liveness + counts. |
| `POST` | `/v1/documents` | Upload and ingest (multipart). |
| `GET` | `/v1/documents` | Paginated catalog. |
| `GET` | `/v1/documents/{id}` | Document detail. |
| `DELETE` | `/v1/documents/{id}?profile_id=` | Delete chunks (optionally one profile) and drop the document when empty. |
| `GET` | `/v1/ingestion-jobs/{job_id}` | Job state, timings, diagnostics. |
| `POST` | `/v1/search` | Query, profile, mode, top_k → provenance-rich hits. |
| `GET` | `/v1/profiles` | Index profiles with config hashes. |

Errors are actionable: `415` unsupported type, `413` too large, `400` parse or
search failure, `404` missing document/profile, `422` invalid query or config.
Body shape: `{"error": "<code>", "message": "<text>"}`.

---

## 8. Testing and quality

```powershell
uv run ruff check .
uv run pytest
```

Tests are deterministic and run **offline** with no model downloads:

- **Unit:** tokenizer and chunk boundaries (fixed and paragraph, overlap,
  section boundaries, long-paragraph splitting, deterministic IDs, offsets);
  parsers (numbered/Markdown headings, encoding fallback, empty input, malformed
  and image-only PDFs, unsupported extensions); the untrusted-input FTS query
  builder; configuration and profile identity.
- **Integration:** contract ingestion → exact-clause search with provenance;
  idempotent re-ingestion; profile isolation; deletion removing both relational
  rows and FTS entries; oversize/unsupported/empty uploads; failed-job recording;
  the full HTTP round-trip (health, upload, search, list, delete, profiles, job).

Run the tests to see the current numbers on your machine; this README does not
pin a pass count.

---

## 9. Chunking strategies

Two strategies are implemented for Phase 1:

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
token counts will differ from those models. A tokenizer-accurate profile is
planned for a later phase. Character counts are never used as token counts.

---

## 10. Limitations and non-goals (Phase 0-1)

- **SQLite, single writer.** Concurrency is limited to one writer process; there
  are no distributed transactions, job queue or public-facing authentication.
  This is a local prototype.
- **No OCR.** Scanned/image-only PDFs yield zero blocks plus a diagnostic; no
  text is invented. Table extraction is not attempted.
- **Heading detection is heuristic.** Markdown ATX headings and a conservative
  numbered-heading pattern (`2.1 Definitions`); it is not a universal contract
  parser. Ambiguous parses surface as `section_path = None` rather than a guess.
- **`section_path` is not OCR'd from layout** and page numbers are unknown
  (`None`) for plain text, where the concept does not apply.
- **No dense/hybrid retrieval, reranking or evaluation metrics yet** — these are
  Phase 2-4. `/v1/search` rejects `mode` values other than `bm25` with a clear
  message rather than pretending to support them.
- **No Alembic migrations yet.** The schema is created from SQLAlchemy metadata
  plus explicit FTS5/trigger DDL. Migrations will be introduced when the schema
  first needs to evolve.
- **Uploads are processed in memory** up to the size limit; there is no
  streaming ingest path yet.

---

## 11. Repository layout

```text
src/rag_workbench/
  config.py            Settings + index-profile identity
  logging_config.py    Structured JSON logging
  cli.py               rag-workbench CLI
  domain/              schemas, Protocol interfaces, typed errors
  persistence/         SQLAlchemy models, engine, FTS5 schema + triggers
  ingestion/           parsers, chunkers, ingestion pipeline
  retrieval/           BM25 retriever, retrieval service
  api/                 FastAPI factory, dependencies, routes
ui/streamlit_app.py    Thin HTTP client
tests/                 unit + integration tests, tiny redistributable fixtures
.github/workflows/ci.yml
```

---

## 12. Roadmap

- **Phase 2:** local Sentence Transformers embeddings, Qdrant-local persistence,
  dense search, RRF hybrid, profile isolation and stage timings.
- **Phase 3:** clause-aware legal chunker, local cross-encoder reranking,
  rank-comparison UI, reranker latency reporting.
- **Phase 4:** BEIR SciFact sample + document-level qrels, metrics with unit
  tests, per-query RAG failure explorer.
- **Phase 5:** optional Ollama answer generation with validated, inspectable
  citations and abstention; portfolio polish and the scaling-to-10M design.

---

## License

MIT. The bundled `tests/fixtures/sample_contract.txt` is a synthetic,
redistributable document written for this project.
