# RAG Workbench

A local-first, zero-mandatory-cost **RAG retrieval engineering laboratory**. The
goal is to make retrieval itself inspectable: ingest documents, chunk them
deterministically, index them in SQLite, run lexical (BM25) and dense retrieval
(or a hybrid of both), and read back provenance-rich citations you can verify
against the source.

This is an engineering research platform, not a "chat with your PDFs" clone. In
this repository there is **no LLM, no API key, no GPU and no Docker requirement**
for anything documented below. The only optional components are local models:
Sentence Transformers (dense/rerank) and a separately installed Ollama
(grounded answers).

**Portfolio arc.** The project is built in reviewed phases, each a study in a
different part of retrieval engineering: a working BM25 vertical slice (Phases
0–1), dense + hybrid retrieval (Phase 2), clause-aware chunking and cross-encoder
reranking (Phase 3), a reproducible evaluation and failure-analysis platform
(Phase 4), evidence-grounded answers with deterministic citation verification
(Phase 5), and a measured scalability case study for ~10 million documents
(Phase 6).

> **Status: Phases 0–6 complete.**
> Implemented: ingestion (TXT / Markdown / text PDF), deterministic chunking
> (`fixed_token`, `paragraph`, **`clause_aware`**), SQLite metadata + FTS5
> indexing, BM25 search, local **Sentence Transformers** embeddings, a
> **Qdrant-local vector index**, **dense** and **RRF hybrid** search, vector index
> status / rebuild / reconciliation, optional local **cross-encoder reranking**,
> a retrieval **evaluation** platform (SciFact + a synthetic legal set) with
> failure diagnostics and committed artifacts, optional **Ollama grounded
> answers** with deterministic citation verification and an answer-quality audit,
> and a **scalability case study**: a bounded performance harness, a
> deterministic capacity model, recorded baselines and a documented design for
> ~10M documents. A FastAPI backend, a CLI and a thin Streamlit client sit on top.
> Nothing in this README claims a capability that was not implemented and tested;
> measured results are labelled as such.

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
- **Retrieval evaluation** over stored datasets with two relevance-mapping
  policies: *document-level* (BEIR SciFact) and *evidence/clause-level* (a
  bundled synthetic legal-contract set). Every run records a reproducible
  configuration, per-query outcomes and provenance.
- **Pure-Python IR metrics** (recall, precision, MRR, nDCG, MAP) plus a separate
  **candidate-pool recall** so pool coverage is never confused with ranking
  quality. Reranking experiments start from the *same* candidate pool as their
  baseline.
- **Rule-based failure diagnostics** derived only from observable evidence
  (judgments, candidate pool, final ranking), with an explicit `unresolved`
  fallback rather than a guessed root cause.
- **Experiment store and export** — JSON and per-query CSV — inspectable through
  the CLI, the API and the Streamlit **Evaluation Lab**, **Failure Explorer** and
  **Performance Lab**.
- **Read-only deployment mode** for the public demo: the backend refuses every
  mutating endpoint and serves the bundled datasets plus committed experiment
  artifacts, so the live app shows real recorded results without mutable state.
- **Optional evidence-grounded answers** (Phase 5) over the retrieved passages,
  using a **local Ollama** model: a strict JSON answer schema, a fixed trusted
  instruction block kept separate from the untrusted passages, **deterministic
  citation verification**, first-class `insufficient_evidence` and
  `generation_error` states, and *retrieved* vs *selected* context kept distinct.
  Generation is refused in read-only mode and when it is disabled, so the public
  demo — which serves **recorded** answer runs and a grounded-answer quality
  report — never becomes a model endpoint.
- **Scalability engineering** (Phase 6): a bounded, reproducible **performance
  harness** over a deterministic synthetic corpus (ingestion throughput, on-disk
  footprint and BM25 latency by corpus size), a **deterministic capacity model**
  that labels every line with its basis (`exact` / `measured` / `assumed` /
  `documented` / `unknown`), a **lightweight local load-testing harness** (with an
  optional Locust definition), committed scaling baselines with recorded code
  provenance, and a **design for ~10 million documents** that preserves the
  retrieval interfaces. The prototype does **not** serve 10M documents; the design
  is explicitly a plan.
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
downloads the reranker model once on first use (about 90 MB). **Answer generation**
(Phase 5) is optional and needs a **separately installed local Ollama** with one
model pulled or imported; with no Ollama the rest of the project is unaffected.
The core install, the tests and the default demo need no Docker, cloud account,
API key or GPU. The **Phase 6 scalability tools** (synthetic corpus generator,
performance harness, capacity model and built-in load-test harness) are part of the
core install; Locust (`load` extra) and `psutil` (`profiling` extra) are optional
extras, or can be used ad hoc via `uv run --with ...`.

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

### Optional: enable answer generation (Phase 5)

Generation is off by default. To turn it on you need a **local** Ollama serving one
model. Install Ollama from https://ollama.com/download (a signed Windows installer),
then either pull a model or *import* an existing GGUF without downloading it:

```powershell
# Option A: pull a small model (network, ~1 GB)
ollama pull qwen2.5:1.5b-instruct

# Option B: import a GGUF you already have (no download)
#   create a Modelfile whose first line is:  FROM C:\path\to\model.gguf
ollama create qwen2.5:1.5b-instruct -f .\Modelfile

# Start the server if it is not already running
ollama serve

# Point RAG Workbench at it and enable generation
$env:RAG_WORKBENCH_GENERATION_ENABLED = "true"
$env:RAG_WORKBENCH_OLLAMA_MODEL = "qwen2.5:1.5b-instruct"
```

Only a **loopback** Ollama endpoint is accepted by default; a remote base URL
requires the explicit `RAG_WORKBENCH_OLLAMA_ALLOW_REMOTE=true` opt-in. Nothing is
downloaded at app start-up.

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

### Evaluate retrieval from the CLI

```powershell
# List bundled / imported datasets
uv run rag-workbench datasets list

# Import a BEIR-format dataset from local files (no network)
uv run rag-workbench datasets import --name mycorpus `
  --corpus corpus.jsonl --queries queries.jsonl --qrels qrels.tsv `
  --source "…" --license "…" --mapping document

# Download the BEIR SciFact benchmark (explicit command; ~2 MB)
uv run rag-workbench datasets download-scifact

# Ingest a dataset's corpus into an index profile, then evaluate it
uv run rag-workbench datasets ingest scifact --strategy paragraph          # lexical profile
uv run rag-workbench datasets ingest scifact --strategy paragraph --embed  # dense profile

uv run rag-workbench eval run --dataset scifact --profile <profile-id> `
  --mode bm25 --query-mode or --candidate-depth 20 --top-k 10 --k 1,3,5,10 `
  --label "SciFact BM25"

# Matched reranking: baseline and reranked start from the SAME 20-passage pool
uv run rag-workbench eval run --dataset scifact --profile <profile-id> `
  --mode bm25 --candidate-depth 20 --top-k 10 --label "pool20 baseline"
uv run rag-workbench eval run --dataset scifact --profile <profile-id> `
  --mode bm25 --candidate-depth 20 --top-k 10 --rerank --rerank-candidates 20 `
  --label "pool20 + rerank"

# Inspect, diagnose, export, and commit a read-only artifact
uv run rag-workbench eval list --dataset scifact
uv run rag-workbench eval show <experiment-id>
uv run rag-workbench eval failures <experiment-id>
uv run rag-workbench eval export <experiment-id> --format csv --out run.csv
uv run rag-workbench eval record <experiment-id> --name scifact-bm25.json
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

### Answer from the CLI (optional, needs local Ollama)

```powershell
# Generate a grounded answer locally (requires RAG_WORKBENCH_GENERATION_ENABLED=true)
uv run rag-workbench answer ask "What is the notice period for termination?" `
  --collection legal-contracts --query-mode or --max-evidence 6

# Run the bundled grounded-answer quality set and print the metrics.
# --seed makes repeated runs reproducible; --label names the configuration.
uv run rag-workbench answer quality `
  --profile <profile-id> --dataset answer-quality/legal-contracts-answers `
  --query-mode or --max-evidence 6 --seed 0 --label "v1 / me=6 / bm25 / or"

# Record answer demos + a quality report as committed read-only artifacts
# (--no-demos writes only the aggregate report, for a configuration sweep)
uv run rag-workbench answer record `
  --profile <profile-id> --dataset answer-quality/legal-contracts-answers `
  --query-mode or --max-evidence 6 --seed 0 --label "recommended"

# Inspect recorded demos with no model running
uv run rag-workbench answer demos
uv run rag-workbench answer demo <demo-id>
```

Answer artifacts live in `benchmarks/recorded_answers/`: one `answer_demo` JSON per
question plus one `answer_quality` report per configuration. All carry
`source="recorded"` so the public demo can never present them as live inference.
The instruction wording is selectable with `RAG_WORKBENCH_ANSWER_PROMPT_VARIANT`
(`v1` default, `v2` for the audited alternative — see §16). A fixed `--seed` makes
a repeated request reproducible; leaving it unset samples normally.

### Scalability and load testing from the CLI (Phase 6)

```powershell
# Generate a reproducible synthetic corpus to inspect on disk
uv run rag-workbench scaling corpus --documents 500 --out data\scaling-corpus

# Run bounded indexing + BM25 benchmarks and record committed artifacts.
# No extra install: BM25 uses only the core dependencies.
uv run rag-workbench scaling run --documents 100,1000,5000 --depths 10,50,100 --record

# Compare chunking strategies on the same corpus
uv run rag-workbench scaling run --documents 1000,5000 --strategy fixed_token `
  --max-tokens 512 --overlap-tokens 64 --record

# Capacity model for ~10M documents (storage + optional ingestion projection).
# Defaults for throughput/overheads are illustrative assumptions (basis "assumed").
uv run rag-workbench scaling capacity --documents 10000000 --scenario typical
uv run rag-workbench scaling capacity --documents 10000000 --scenario typical `
  --documents-per-day 100000 --embedding-chunks-per-second-per-worker 20 --workers 4

# Inspect recorded baselines with no API or model running
uv run rag-workbench scaling list
uv run rag-workbench scaling show <benchmark-id>

# Bounded SMOKE-level load test against a LOCAL server (never the public demo).
# A warm-up phase is excluded, and per-endpoint latency is reported.
uv run rag-workbench serve            # in another terminal
uv run rag-workbench scaling load-test --url http://127.0.0.1:8765 --workload mixed `
  --users 4 --duration 10 --warmup 2
```

`scaling run` refuses to index more than `RAG_WORKBENCH_SCALING_MAX_DOCUMENTS`
documents (default 20,000), so an accidental run cannot exhaust local storage.
Every recorded artifact carries the code revision and whether the working tree was
clean (`provenance`); a run from a dirty tree is flagged rather than presented as a
clean revision.

`rag-workbench scaling load-test` uses only `httpx`. Its numbers are a
**smoke-level, single-machine observation** of **HTTP/API throughput across
endpoints** — they are **not** RAG generation throughput, not retrieval throughput,
and not a latency SLO. For richer reporting the same read-only workloads are
available through the optional Locust definition:

```powershell
uv run --extra load locust -f benchmarks\load\locustfile.py --host http://127.0.0.1:8765
```

Locust and `psutil` are declared as **optional extras** (`load` and `profiling`),
so they are never installed by the core `uv sync` and the default demo stays
zero-mandatory-cost. Install them when you want them:

```powershell
uv sync --extra load          # Locust
uv sync --extra profiling     # psutil -> the harness records process RSS
```

`uv run --with locust ...` / `uv run --with psutil ...` also work without changing
the environment. See [`docs/scaling_to_10m.md`](docs/scaling_to_10m.md) and
[`docs/case_studies.md`](docs/case_studies.md).

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
- `st.secrets` values named `RAG_WORKBENCH_*` are copied into the environment
  once, before the in-process API starts, so Cloud configuration needs no shell.
- The public demo runs in **read-only mode**: set
  `RAG_WORKBENCH_READ_ONLY=true` and `RAG_WORKBENCH_AUTOSEED_SAMPLE=true`.
  The backend then refuses every mutating endpoint with `403 read_only_mode`,
  while a trusted internal startup hook seeds the bundled sample contract so a
  fresh instance still has something to search. The **Evaluation Lab**,
  **Failure Explorer**, **Performance Lab** and **Answer Laboratory** show the
  committed datasets under `datasets/` and the committed artifacts under
  `benchmarks/recorded/` and `benchmarks/recorded_answers/`. In read-only mode the
  answer endpoints are refused, so the app is never a model endpoint; live
  generation additionally requires `RAG_WORKBENCH_GENERATION_ENABLED=true`.
- The **Deployment diagnostics** expander on the Evaluation Lab reports the
  backend `version` and its separate build `revision` (plus how it was resolved:
  `git`, `git-file` or `source-hash`) and compares the revision to the source
  tree on disk. If a hot reload left an older backend process running, the
  mismatch appears as a warning instead of the app silently serving stale code.

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
   the Reranking Laboratory returns a clear `503 reranker_unavailable`. Because
   the extra is absent, dense/hybrid/reranked *evaluation runs* are disabled too
   — the recorded SciFact and legal experiments still display.
4. Add these **secrets** in the Cloud app settings:

   ```toml
   RAG_WORKBENCH_READ_ONLY = "true"
   RAG_WORKBENCH_AUTOSEED_SAMPLE = "true"
   ```

Cloud caveats: the filesystem is **ephemeral** — the SQLite index and vector
store reset when the app sleeps or reboots (hence the auto-seed, which also
fills in a missing demonstration profile after a code upgrade) — and the
deep-learning dense stack is heavy for the free tier, so BM25-only is the
default live demo. The committed datasets and recorded artifacts are located by
finding the repository root (marker: `pyproject.toml` beside `src/rag_workbench`)
via the module path, the working directory and `sys.path`, so they are still
found when the package is installed non-editable rather than from the source
tree.

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
| `RAG_WORKBENCH_AUTOSEED` | `1` | UI-side sample seeding when the index is empty (`0` disables). |
| `RAG_WORKBENCH_DEFAULT_CANDIDATE_DEPTH` | `100` | Default candidate-pool depth for evaluation runs. |
| `RAG_WORKBENCH_DEFAULT_EVAL_K` | `[1, 3, 5, 10]` | Default rank cut-offs reported by evaluation. |
| `RAG_WORKBENCH_DATASET_DIR_NAME` | `datasets` | Writable dataset root, relative to `DATA_DIR`. |
| `RAG_WORKBENCH_BUNDLED_DATASETS_DIR` | `datasets` | Committed, bundled datasets shipped with the repo. |
| `RAG_WORKBENCH_RECORDED_EXPERIMENTS_DIR` | `benchmarks/recorded` | Committed experiment artifacts shown by a read-only deployment. |
| `RAG_WORKBENCH_READ_ONLY` | `false` | Refuse every mutating endpoint (`403 read_only_mode`). |
| `RAG_WORKBENCH_AUTOSEED_SAMPLE` | `false` | Trusted startup bootstrap that seeds the sample corpus once (for a fresh read-only instance). |
| `RAG_WORKBENCH_GENERATION_ENABLED` | `false` | Master switch for live answer generation (`POST /v1/answer`). Off by default; the public demo leaves it off. |
| `RAG_WORKBENCH_OLLAMA_BASE_URL` | `http://127.0.0.1:11434` | Ollama endpoint. Only loopback is accepted unless remote is opted in. |
| `RAG_WORKBENCH_OLLAMA_MODEL` | `qwen2.5:1.5b-instruct` | Model name to request. Never downloaded by the app. |
| `RAG_WORKBENCH_OLLAMA_TIMEOUT_S` | `120` | Per-request generation timeout. |
| `RAG_WORKBENCH_OLLAMA_TEMPERATURE` | `0.1` | Sampling temperature. |
| `RAG_WORKBENCH_OLLAMA_NUM_PREDICT` | `512` | Maximum tokens to generate. |
| `RAG_WORKBENCH_OLLAMA_SEED` | unset | Optional sampling seed; set it to make repeated requests reproducible. |
| `RAG_WORKBENCH_OLLAMA_ALLOW_REMOTE` | `false` | Explicit opt-in to a non-loopback Ollama endpoint. |
| `RAG_WORKBENCH_ANSWER_PROMPT_VARIANT` | `v1` | Instruction wording: `v1` (default) or `v2` (status definitions + abstention example; audited, §16). Wording only. |
| `RAG_WORKBENCH_ANSWER_MAX_EVIDENCE` | `6` | Maximum passages selected into the model context. |
| `RAG_WORKBENCH_ANSWER_CONTEXT_TOKENS` | `2048` | Token budget for the evidence block. |
| `RAG_WORKBENCH_RECORDED_ANSWERS_DIR` | `benchmarks/recorded_answers` | Committed recorded answer demos + quality reports for the read-only demo. |

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
| `GET` | `/v1/config` | Deployment capabilities: `read_only`, dense/rerank availability, default eval cut-offs, and the build identity (`version` plus a `revision`/`revision_source`). |
| `GET` | `/v1/datasets` | List bundled and imported evaluation datasets with counts. |
| `GET` | `/v1/datasets/{name}` | One dataset's manifest summary. |
| `POST` | `/v1/datasets/ingest` | `{dataset, strategy?, max_tokens?, embed?, …}` → ingest a corpus into an index profile. |
| `POST` | `/v1/evaluations` | `{dataset, profile_id, mode, query_mode?, candidate_depth?, top_k, k_values?, rerank?, rerank_candidates?, …}` → run and persist an evaluation. |
| `GET` | `/v1/experiments?dataset=&include_recorded=` | List live and recorded experiments. |
| `GET` | `/v1/experiments/{id}` | Full experiment record with per-query outcomes and traces. |
| `GET` | `/v1/experiments/{id}/queries/{query_id}` | One query's metrics, diagnostics and ranked trace. |
| `GET` | `/v1/experiments/{id}/export?format=json\|csv` | Download a full JSON record or a per-query CSV. |
| `POST` | `/v1/answer` | `{question, profile_id?, mode?, collection?, query_mode?, candidate_depth?, max_evidence?, context_tokens?, temperature?, num_predict?, seed?, rerank?}` → grounded answer with selected evidence, citation verification and diagnostics. Needs `RAG_WORKBENCH_GENERATION_ENABLED=true` and a non-read-only deployment. |
| `GET` | `/v1/answer-demos` | List recorded answer demonstrations (used by the read-only demo). |
| `GET` | `/v1/answer-demos/{id}` | One recorded answer demonstration. |
| `GET` | `/v1/answer-quality` | The newest recorded grounded-answer quality report. |
| `GET` | `/v1/answer-quality-reports` | Every recorded quality report (summary), newest first, for configuration comparison. |
| `GET` | `/v1/scaling-benchmarks` | List recorded scalability benchmarks (summary), newest first. |
| `GET` | `/v1/scaling-benchmarks/{id}` | Full recorded scaling benchmark: environment, corpus, per-size workloads and latencies. |
| `GET` | `/v1/scaling/scenarios` | The three documented storage scenarios (small / typical / large). |
| `POST` | `/v1/scaling/capacity` | `{storage?, ingestion?}` → deterministic storage and ingestion estimates. Pure arithmetic; works in read-only mode. |
| `GET` | `/v1/scaling/architecture` | Current-vs-proposed architecture data used by the Scalability Laboratory and `docs/scaling_to_10m.md`. |

`mode` accepts `bm25`, `dense` and `hybrid`. `query_mode` accepts `and` (the
interactive default, every term must appear) and `or` (any term; BM25 ranks,
which is what natural-language and benchmark queries need). Errors are
actionable: `415`
unsupported type, `413` too large, `400` parse or search failure, `404` missing
document/profile/dataset/experiment (`404 answer_not_found` for a missing answer
demo or quality report), `422` invalid query/config/unsupported mode,
`403 read_only_mode` for any mutation — including live generation — in a
read-only deployment, `503` dense, reranker or generation dependencies being
unavailable (`dense_unavailable`, `vector_index_not_built`,
`reranker_unavailable`, `generator_unavailable` when generation is disabled or
Ollama/the model is unreachable), and `400 generation_error` for a generation
that ran but produced no usable structured answer. Body shape:
`{"error": "<code>", "message": "<text>"}`.

The scalability endpoints add `404 scaling_benchmark_not_found` for a missing
recorded artifact, `400 scaling_error` for an invalid benchmark/capacity request
(e.g. a workload above the configured document cap), and `422 validation_error`
for capacity assumptions outside their bounds. `POST /v1/scaling/capacity` is
**not** gated by read-only mode: it computes arithmetic and mutates nothing.

`POST /v1/answer` is deliberately layered: `candidate_depth` controls the
first-stage retrieval pool, `max_evidence` controls how many passages reach the
model, and `context_tokens` bounds the evidence block. The response always keeps
the **retrieved** hits and the **selected** evidence separate, reports
`insufficient_evidence` as a first-class outcome (not an error), and returns
`generation_error` rather than raising when the model times out or returns
malformed output.

Reranking is optional and off by default. The evaluation runner keeps four
quantities explicit and consistent: the **first-stage candidate depth**
(`candidate_depth` — how deep the pool is retrieved, and the depth at which
candidate recall is measured), the **reranker input** (`rerank_candidates`,
default `candidate_depth` — how many pooled candidates are actually scored), the
**final result count** (`top_k` — the cut ranking metrics use), and the
candidate-recall depth (the first-stage pool, always measured on a rerank-free
retrieval, so reranking can never change pool coverage). At request time
`rerank=true` retrieves `rerank_candidates` candidates, reorders them and returns
the top `top_k`; `rerank_candidates` must be `>= top_k` (else `422
invalid_candidates`). The runner additionally requires `candidate_depth >= top_k`
and `rerank_candidates <= candidate_depth` (else `422 invalid_evaluation_config`):
a reranker can only score candidates that were actually retrieved. For a
*matched* comparison set `rerank_candidates == candidate_depth`; when the
reranker input is smaller, the pooled candidates beyond it are still counted in
candidate recall rather than silently discarded. Because evaluation ranks
*documents* and repeated chunks collapse to one document, the runner requests
the whole reranked pool from the service and applies the `top_k` cut after
de-duplication, so a short request cannot collapse to fewer than `top_k`
documents.

---

## 8. Testing and quality

```powershell
uv run ruff check .
uv run pytest                 # core suite; dense/real-model tests are deselected
uv run pytest -m dense -s     # opt-in: real Sentence Transformers smoke test
uv run pytest -m reranker -s  # opt-in: real cross-encoder reranker smoke test
uv run pytest -m ollama -s    # opt-in: real local-Ollama generation smoke test
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
- **Evaluation:** hand-computed IR metrics (recall/precision/MRR/nDCG/MAP and
  candidate-pool recall) against known rankings; the dataset schemas, checksums,
  BEIR parsing and the two relevance-mapping policies; the bundled synthetic
  legal dataset's integrity; document-level and evidence-level evaluation
  end-to-end over real SQLite/BM25; experiment persistence plus JSON/CSV export
  round-trips; and the evaluation API and read-only enforcement (a mutating
  endpoint returns `403 read_only_mode`).
- **Answer (Phase 5):** context selection (sequential `[S#]` ids, duplicate and
  containment de-duplication, `max_evidence` and token-budget exclusion with an
  explicit reason for every dropped passage); deterministic citation verification
  (unknown ids, uncited claims, `answered`-without-citations, **strict citation
  parsing** so a string merely *containing* an id is rejected as malformed, and the
  semantic-support and answer-vs-claims properties always reported `not_determined`);
  reference-evidence coverage (`never_retrieved` / `retrieved_but_excluded` /
  `included`, with partial vs. complete coverage kept distinct); the prompt trust
  boundary and both instruction variants (retrieved text appears only in the
  delimited user message, never in the trusted system instructions); the Ollama
  adapter over an in-memory HTTP transport (model listing, availability, schema-valid
  parsing, code-fence recovery, malformed / HTTP-error / timeout error results, and
  remote-endpoint refusal); the recorded-artifact loader (demos and multiple quality
  reports); the grounded-answer quality runner and its aggregation; and the API —
  grounded answer, unknown-citation flagging, `generation_error`, `503` when
  generation is disabled or unavailable, `403 read_only_mode`, empty-evidence
  abstention, and the recorded demos / quality / quality-reports endpoints.
- **Dense (opt-in, `-m dense`):** the real `all-MiniLM-L6-v2` model builds a dense
  profile for the sample contract and returns the expected top clause.
- **Reranker (opt-in, `-m reranker`):** the real
  `cross-encoder/ms-marco-MiniLM-L6-v2` model reranks a real candidate list,
  keeping retrieval and rerank scores separate and reporting its own timings.
- **Ollama (opt-in, `-m ollama`):** the real local model answers a question over a
  real ingested contract and the reply passes structural and citation verification
  (skipped when Ollama or the model is unavailable).

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
- **Optional answer generation is local, small and non-deterministic.** Live
  generation needs a separately installed Ollama; the app never downloads a model.
  Answers come from the configured local model and are **not** guaranteed correct:
  the pipeline verifies *structure* and *citation ids* deterministically but cannot
  verify that a cited passage *semantically supports* a claim, so
  `semantic_support` is always reported as `not_determined`. A small model can
  answer a question the passages do not support (see §16), add unsupported claims,
  or — as observed with 1.5B Qwen — fail citation-id formatting unless the output
  schema constrains it (the schema now uses a `pattern` so decoding emits `S#`). Its
  abstention behaviour is also **not stable**: the same prompt and configuration can
  abstain on one run and not another (§16), so a single recorded number is a draw,
  not a property.
  Generation is refused in read-only mode and when disabled, so the public demo is
  not a model endpoint. Prompt-based defences reduce, but do not eliminate,
  prompt-injection risk from retrieved text.
- **Scalability measurements are single-node and synthetic.** The performance
  harness measures the *current prototype* on one machine over a generated corpus;
  the numbers characterise cost, not relevance, and are not a production
  throughput claim. Each recorded artifact names its code revision and whether the
  working tree was clean, so a dirty-tree run is never presented as a clean
  revision. The 10M-document capacity model is **deterministic arithmetic over
  stated assumptions**, not a simulation: its default throughput and overhead
  values are labelled `assumed`, and `keeps_up` is a conditional projection, not a
  demonstrated capability. Process memory is only measured when `psutil` is
  installed; otherwise RSS is reported as not measured and the Python-heap peak is
  never extrapolated to it. The proposed distributed architecture is **design
  only** — none of PostgreSQL, a distributed lexical engine, a queue or a Qdrant
  server is implemented or required. Dense/hybrid/rerank scaling is not measured.
  Load testing targets a **locally controlled** server, never the public demo, and
  reports smoke-level HTTP/API throughput, not generation throughput.
- `/v1/search` rejects unknown `mode` values with a clear message rather than
  pretending to support them.
- **Evaluation scope.** The metrics, diagnostics and failure explorer operate on
  **stored datasets** with explicit relevance judgments; they are not a
  leaderboard and the bundled legal set is a small synthetic fixture, not a
  legal-validity benchmark. SciFact is the only external benchmark imported, and
  its retrieval depth is a **chunk** count, so a document-level metric at K may
  need a deeper chunk pool than K (reported separately as candidate-pool recall).
  Diagnostic categories are rule-based over observable evidence and fall back to
  `unresolved` rather than inferring a cause; a diagnosis is a navigation aid,
  not a verified root cause.
- **BM25 term combination matters.** The interactive default is `and` (every
  term must appear), which is brittle for long natural-language queries; the
  evaluation default is `or`. The measured difference is reported in
  §15 and is a property of this lexical stack, not a universal BM25 result.

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
  evaluation/          metrics, datasets, diagnostics, runner, experiments, service
  answering/           context selection, prompt trust boundary (v1/v2 variants),
                       Ollama adapter, citation verification, reference-evidence
                       coverage, pipeline, quality eval, recorded artifacts
  scaling/             synthetic corpus generator, performance harness, capacity
                       model, load-testing harness, architecture design data,
                       recorded-benchmark loading, application service
  api/                 FastAPI factory, dependencies, routes
ui/streamlit_app.py    Thin HTTP client
ui/local_backend.py    In-process API for single-process (Streamlit Cloud) deploys
datasets/              Committed evaluation datasets (bundled legal-contracts,
                       answer-quality set)
docs/                  scaling_to_10m.md (10M-document design), case_studies.md
benchmarks/recorded/   Committed experiment artifacts for the read-only demo
benchmarks/recorded_answers/  Committed recorded answer demos + quality reports
benchmarks/recorded_scaling/  Committed scalability baselines (Phase 6)
benchmarks/load/       Optional Locust load-test definition
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
- **Phase 4 (done):** BEIR SciFact import with document-level qrels, a curated
  synthetic legal set with clause/evidence-level relevance, pure-Python IR
  metrics, candidate-pool **and** ranking recall, rule-based failure diagnostics,
  persisted experiments with JSON/CSV export, the Evaluation Lab / Failure
  Explorer / Performance Lab, a read-only demo mode over committed artifacts, and
  a corrected matched reranking comparison (§15).
- **Phase 5 (done):** optional local Ollama answer generation over the retrieved
  passages, with a strict answer schema, a prompt trust boundary, deterministic
  citation verification, first-class `insufficient_evidence` / `generation_error`
  states, retrieved-vs-selected context, an **Answer Laboratory** UI page, a CLI
  `answer` command group, a committed grounded-answer quality set and report, and
  recorded answer artifacts for the read-only demo.
- **Phase 6 (done):** a bounded local **performance harness** over a deterministic
  synthetic corpus, a **deterministic capacity model** with explicit bases, a
  **lightweight load-testing harness** (plus an optional Locust definition),
  committed scaling baselines, a **Scalability Laboratory** Streamlit page, and a
  documented design for ~10M documents that preserves the retrieval interfaces
  (`docs/scaling_to_10m.md`, `docs/case_studies.md`).
- **Later (optional):** a hosted dense/reranker benchmark, and a real distributed
  prototype behind the existing Protocols. Neither is claimed as built.

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

## 15. Observed Phase 4 results (SciFact + synthetic legal set)

These figures were **measured in this project** on one Windows machine (CPU),
Python 3.12, with `sentence-transformers/all-MiniLM-L6-v2` (384-d) and
`cross-encoder/ms-marco-MiniLM-L6-v2`. They are single-machine observations, not
a leaderboard. The exact runs are committed as artifacts in
`benchmarks/recorded/` (see `benchmarks/README.md`); the artifacts were
regenerated with record schema `2`, so every number below is reproducible from
the committed code. Ranking metrics are unchanged from the original Phase 4 run
where the code is unchanged; latency now covers only the queries that actually
retrieved (see the note on unjudged queries).

### SciFact (BEIR, document-level)

The **full** SciFact corpus (5183 documents) is chunked with the `paragraph`
profile (9050 chunks) and evaluated on the 300 test claims that carry judgments.
The 809 unjudged queries in `queries.jsonl` are **skipped before retrieval** (a
query with no judgments can never produce a metric, so retrieving it would only
waste work) and excluded from every aggregate and from latency; they are still
reported in the run's `counts`, never silently dropped. `k=10`, chunk candidate
depth 100.

| Configuration | recall@10 | nDCG@10 | MRR@10 | candidate_recall@100 |
|---|---|---|---|---|
| BM25 (`or`, paragraph) | 0.777 | 0.648 | 0.611 | 0.871 |
| Dense (MiniLM) | 0.808 | 0.647 | 0.599 | 0.913 |
| Hybrid (RRF, 200 candidates) | **0.825** | **0.686** | **0.647** | **0.943** |

- BM25's nDCG@10 (~0.65) is in the range usually reported for Anserini BM25 on
  SciFact, which is a sanity check that the metric plumbing is correct.
- Hybrid RRF is best on every ranking metric here: the lexical and dense
  candidate lists are complementary, and fusion keeps passages found by only one
  retriever eligible.
- `candidate_recall@100` (0.87–0.94) bounds what any reranker could achieve from
  that pool: reranking can only reorder, never recover.

### Matched reranking (SciFact, same 20-passage pool)

To compare fairly, the baseline and the reranked run start from the **identical**
20-chunk first-stage pool (this is the correction required by the phase spec: the
baseline is the first-stage ranking of the *same* candidates, so a passage the
reranker "finds" cannot have been absent from the pool). Both runs use BM25
(`or`), `top_k=10`, chunk depth 20, with `rerank_candidates == candidate_depth ==
20`; candidate recall is measured on an independent rerank-free retrieval of the
same pool, so it matches between the two rows by construction.

| Configuration (pool of 20) | recall@10 | nDCG@10 | MRR@10 | candidate_recall@20 |
|---|---|---|---|---|
| BM25 (`or`) baseline | 0.777 | 0.648 | 0.611 | 0.814 |
| BM25 + `ms-marco-MiniLM-L6-v2` rerank | **0.802** | **0.695** | **0.670** | 0.814 |

Reranking improves nDCG@10 by ~0.047 and MRR@10 by ~0.059 with **no change to
candidate recall at any depth**: `candidate_recall@10` (0.777) and
`candidate_recall@20` (0.814) are identical for the baseline and the reranked
run, because candidate recall is measured on the first-stage pool and reranking
only reorders that same pool. That is direct evidence the reranker could not
have recovered anything absent from the pool. Reranking cost is real on CPU:
cross-encoder inference averaged ~677 ms per query for 20 candidates
(p50 670 ms, p95 812 ms) after a one-time ~18.3 s model load, so it is worth its
quality gain only when ranking quality matters more than latency.

### Legal chunking comparison (bundled synthetic set, evidence-level)

The bundled `datasets/legal-contracts` set (7 synthetic contracts, 48
natural-language queries, 65 clause/evidence units) is evaluated with
**clause/evidence-level** relevance: a retrieved chunk counts only if it contains
the judged clause's evidence text (or matches its section number). Pool depth 100.

| Configuration | recall@10 | nDCG@10 | MRR@10 | candidate_recall@100 |
|---|---|---|---|---|
| BM25 (`or`), `paragraph` | 0.833 | 0.590 | 0.544 | 1.000 |
| BM25 (`or`), `clause_aware` | **0.854** | **0.604** | **0.554** | 1.000 |

Clause-aware chunking is a small, *within-noise* improvement at the top of the
ranking on this tiny set (recall@10 0.854 vs 0.833), and the candidate pool is
complete in both cases. This is an honest "no dramatic difference at this scale"
result, not a claim that clause-aware chunking is unnecessary. (The evidence
mapping scores a retrieved chunk only when it contains the judged clause's
evidence text or matches its section number, so these numbers are sensitive to
exact chunk boundaries; both profiles are measured on the same corpus and
judgments.)

### BM25 term combination (`and` vs `or`)

The same legal set sizes the effect of lexical term combination:

| Query mode | recall@10 | nDCG@10 | candidate_recall@100 |
|---|---|---|---|
| `and` (every term must appear) | 0.104 | 0.104 | 0.104 |
| `or` (any term; BM25 ranks) | **0.833** | **0.590** | **1.000** |

Long natural-language queries almost never contain *all* their terms in one
chunk, so FTS5's implicit conjunction collapses to near-zero recall. This is a
property of this lexical stack, reported rather than hidden; the interactive
default stays `and` for backward-compatible exact-phrase behaviour, and
evaluation defaults to `or`.

---

## 16. Observed Phase 5 results (grounded answers)

**This section reports a run actually performed on this machine.** Generation uses
a *local* model served by Ollama; it is a single-machine smoke observation, not a
model benchmark, and every number depends on the chosen model.

Setup:

- Model: `qwen2.5:1.5b-instruct`, **imported into Ollama from an existing GGUF file**
  — no model was downloaded for this run. Ollama 0.11.x, CPU inference.
- Corpus: the bundled synthetic `legal-contracts` set (7 contracts), ingested with
  the `clause_aware` profile `p-2c8893ea8e1621ef`.
- Retrieval: BM25, `query_mode=or`, candidate pool 20, `context_tokens=2048`.
  `max_evidence` varies by run (4/6/8); the **recommended** live default is 6.
- The quality set has ten questions: seven fully answerable, one partially
  answerable and two unanswerable (one answer is absent from the corpus, one from
  the specific contract).

Every report listed below is a real execution committed under
`benchmarks/recorded_answers/aq-*.json`; `benchmarks/recorded_answers/` also holds
the raw per-question answer demos. The API exposes the newest report at
`/v1/answer-quality` and the full comparison list at `/v1/answer-quality-reports`,
which the Answer Laboratory renders as a table.

### The original committed run (`aq-20261015T225614`, `max_evidence=4`)

| Metric | Value |
|---|---|
| structural validity (schema-valid output) | 1.00 |
| citation validity (cited ids resolve to selected passages) | 0.90 |
| **correct abstention on the unanswerable questions** | **0.00** |
| over-abstention on answerable questions | 0.00 |
| generation failures | 0.00 |
| semantic support | *not determined* |

The original run answered all ten questions, including the two unanswerable ones.
Re-running the **identical** prompt and configuration later on the same machine now
abstains correctly on both (`correct_abstention_rate = 1.00`, stable across seeded
and unseeded repeats). The model artifact in Ollama was re-imported between the two
runs (`modified_at` moved from 2026-10-15 to 2026-10-16), so the original `0.00` is
recorded here as **one non-reproducible draw**, not a stable property of the
pipeline. Both the original and the re-runs are kept as committed artifacts.

### Evidence-loss audit: why `aq-002` was answered wrongly

`aq-002` asks two things — the invoicing period **and** the late-payment interest.
It is the interesting failure: the late-payment clause (`msa-02::3.2`, containing
"one percent per month") *was* retrieved (rank 5) but was **dropped from the
context**, while the invoicing clause reached the model. The old metric only asked
"was *any* reference keyword present?", so it reported full coverage (1.00) and hid
the loss.

Coverage is now computed per required evidence unit and classified three ways
(`src/rag_workbench/answering/reference.py`): **`included`** (every required anchor
reached the context), **`retrieved_but_excluded`** (retrieved but dropped by
`max_evidence` or the token budget) and **`never_retrieved`** (absent from the whole
pool). Complete and partial coverage are reported separately:

| Metric | Value |
|---|---|
| complete reference coverage | 0.875 (7/8 anchored questions) |
| partial reference coverage | 0.125 (`aq-002`) |
| per-anchor coverage (units selected / required) | 0.917 (11/12) |
| evidence included / retrieved-but-excluded / never-retrieved | 7 / 1 / 0 |

`aq-002` is likewise visible at runtime: its `evidence_excluded_by_max` diagnostic
now names the dropped passages instead of reporting a bare count.

### Controlled configuration comparison

Configurations recorded with a fixed `seed=0` (unseeded repeats of the same
config agreed, so the differences below are configuration effects, not sampling
noise). All runs use the same corpus, profile and BM25/`or` retrieval. The chosen
configuration is marked `recommended` in its report, which is what
`GET /v1/answer-quality` headlines for the read-only demo while
`/v1/answer-quality-reports` lists them all.

| Report | Prompt | max_ev | Citation validity | Correct abstention | Answer-state accuracy | Complete ref. coverage | gen p50 |
|---|---|---|---|---|---|---|---|
| `…225614` (original) | v1 | 4 | 0.90 | 0.00 | — | — | 3.12 s |
| `…081735` | v1 | 4 | 1.00 | 1.00 | 0.90 | 0.875 | 1.80 s |
| **`…082903` (recommended)** | **v1** | **6** | **1.00** | **1.00** | **0.90** | **1.00** | **1.95 s** |
| `…081839` | v1 | 8 | 1.00 | 1.00 | 0.90 | 1.00 | 1.97 s |
| `…081904` | v2 | 4 | 1.00 | 0.50 | 0.80 | 0.875 | 1.96 s |
| `…081932` | v2 | 6 | 0.90 | 1.00 | 0.90 | 1.00 | 1.81 s |

What this shows:

- **`max_evidence=6` resolves the evidence loss.** Raising the limit from 4 to 6
  takes complete reference coverage from 0.875 to 1.00 (it selects `msa-02::3.2`),
  improving answer-state accuracy to 0.90 with no meaningful latency change. Going
  to 8 buys nothing further — a concrete demonstration that *more context is not
  automatically better*.
- **The "clearer status definitions + abstention example" prompt did not help.**
  It is recorded as variant `v2` and compared directly against the original `v1` at
  the same config. At `max_evidence=4`, `v2` **regressed** correct abstention from
  1.00 to 0.50 — it answered `aq-008` (late-delivery penalty) with an unrelated
  indemnification clause instead of abstaining — and at `max_evidence=6` it
  produced one invalid citation. The hypothesis is therefore **not supported**;
  `v1` remains the default and `v2` is kept selectable (via
  `RAG_WORKBENCH_ANSWER_PROMPT_VARIANT=v2`) purely for reproducibility.
- **Coverage is not correctness.** See `aq-007` below: complete coverage can coexist
  with a wrong answer.

### Preserved failure: `aq-007` (citation resolves, but does not support)

`aq-007` asks the NDA's governing law **and** a disaster-recovery objective. The
correct first part is *Delaware*, and the Delaware clause (`nda-01::8`) **was**
selected into the context. The model nevertheless answered *"governed by the laws of
the State of California"* and marked the answer `answered`, citing `S1` — which is
the NDA preamble, not the governing-law clause.

This is kept deliberately, because it is the exact class of error the deterministic
verifier **cannot** catch: `S1` is a real selected passage, so `citation_valid` is
`true`; whether it *supports* the claim is semantic and is reported as
`not_determined`. The Answer Laboratory now shows each claim next to the verbatim
text of the passage(s) it cites, so the mismatch is inspectable rather than hidden:
the cited text is the preamble while the claim is about governing law. Citation
completeness is likewise split into mechanical properties (every id resolves, every
claim is cited) and a `not_determined` "answer text vs. structured claims"
consistency property that is never asserted automatically.

### Latency (single machine, CPU, 1.5B Q8)

| Stage | p50 (recommended run) |
|---|---|
| retrieval (BM25, 20-candidate pool) | ~28 ms |
| generation | ~1.94 s |
| total per question | ~2.23 s |

First-call latency is higher because the model is loaded on demand. Run
`rag-workbench answer record` to reproduce with your own model.

---

## 17. Observed Phase 6 results (scalability baselines)

Measured by `rag-workbench scaling run` on one machine (Windows 11, Python 3.13.5,
16 logical cores, AMD64) over a deterministic synthetic corpus (seed 0), committed
as `benchmarks/recorded_scaling/*.json`. Each record stores its code revision and
working-tree state (`provenance`). **These are cost measurements on a single node,
not a throughput claim;** see [`docs/scaling_to_10m.md`](docs/scaling_to_10m.md).

**Paragraph chunking (220/40 tokens), BM25, `query_mode=or`, 20 queries per cell:**

| Documents | Chunks | Ingest (docs/s) | DB size | FTS5 index | @10 p50 | @100 p50 |
|---:|---:|---:|---:|---:|---:|---:|
| 100 | 687 | 34.7 | 1.8 MB | 177 KB | 3.8 ms | 6.5 ms |
| 1,000 | 7,774 | 38.2 | 18.8 MB | 2.0 MB | 21.1 ms | 26.3 ms |
| 5,000 | 38,709 | 33.1 | 87.3 MB | 9.9 MB | 112.2 ms | 105.0 ms |

**Fixed-token chunking (512/64 tokens), BM25, 20 queries per configuration:**

| Documents | Chunks | Ingest (docs/s) | DB size | FTS5 index |
|---:|---:|---:|---:|---:|
| 1,000 | 18,154 | 24.5 | 21.2 MB | 2.4 MB |
| 5,000 | 90,470 | 21.6 | 97.3 MB | 11.8 MB |

There is **no single throughput number** here to quote out of context: ingestion is
~22–38 tiny documents/second (single writer), and each retrieval configuration is
measured over **20 queries** — not a benchmark at production query volume. The
committed CLI load test is a **four-second-ish smoke observation of HTTP/API
throughput across endpoints**, not RAG generation throughput.

**Findings.**

- Single-writer ingestion is ~22–38 tiny documents/second and BM25 p50 grows from
  ~4 ms to ~112 ms across a ~56× increase in chunks. The prototype does not scale
  linearly, and resizing a single node is not the path to 10M documents.
- **Chunking strategy dominates chunk count:** a larger token budget produced
  *more* chunks, because fixed-token windows never cross block boundaries. This is
  the corpus's most influential cost assumption.
- The capacity model's ~3.5 TB total for the `typical` 10M-document scenario is an
  **assumption-based planning envelope** (illustrative overhead multipliers, plus
  replicas, one backup, a reindex working set and 25% headroom), **not a measured
  size**. The measured FTS5 size (~267 B/chunk) is only a calibration point against
  its assumed lexical ratio.
- **Capacity provenance:** the ingestion calculator defaults to illustrative
  throughput with basis `assumed`; storage-overhead multipliers are also `assumed`.
  `keeps_up=true` is a conditional projection, not a demonstrated capability.
- **Memory provenance:** the harness distinguishes an instantaneous `rss_mb`, a
  `sampled_peak_rss_mb` and a `tracemalloc_peak_mb` Python-heap peak; without
  `psutil`, process RSS is reported as not measured and never inferred from the heap.
- Dense, hybrid and reranked scaling were **not** measured (they need the optional
  `dense` extra and local model weights).

---

## License

MIT. The bundled `tests/fixtures/sample_contract.txt` and
`tests/fixtures/legal_contract.txt` are synthetic, redistributable documents
written for this project.
