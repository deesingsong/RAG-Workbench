# RAG Workbench — Technical Product & Engineering Specification

**Project type:** Local-first, open-source RAG experimentation and evaluation platform  
**Primary objective:** Demonstrate the ability to design, debug, benchmark, and improve retrieval-augmented generation rather than build another generic “chat with PDF” app.  
**Budget:** RM0 in mandatory API, hosting, and software fees. Internet access is needed once to install dependencies and download open-weight models/datasets; models run locally.  
**Target environment:** Windows + VS Code + Python 3.11/3.12; CPU-first with optional GPU acceleration. Docker is optional, not required for development.  
**Audience:** AI engineer interviewers, recruiters, and reviewers who want reproducible evidence of architectural choices.

## 1. Problems this project should prove

The portfolio project is built to address five interview scenarios:

1. **Design RAG for 10 million documents.** Build a modest real implementation and separately document an honest, quantitative plan for scaling it to 10 million documents; do not imply the laptop prototype handles that scale.
2. **Debug wrong RAG answers.** Trace stages and distinguish document ingestion, retrieval, reranking, context assembly, and answer-generation failures.
3. **Choose chunking for legal contracts.** Compare fixed-token, paragraph/recursive, and contract-clause-aware chunking with identifiable source boundaries.
4. **BM25 vs. dense vs. hybrid.** Implement all three, run exactly the same evaluation queries, and compare retrieval quality and latency.
5. **Measure reranker benefit.** Add an optional local cross-encoder and report both ranking improvement *and* added latency.

**Core design principle:** Retrieval works fully without a generative LLM. Local answer generation with Ollama is an optional later-stage feature; evaluation of retrieval is the main product.

## 2. Selected stack (deliberately lean)

| Layer | Default | Reason / constraints |
|---|---|---|
| Language & packaging | Python 3.12, `uv` and `pyproject.toml` | Simple reproducible local installs. Python 3.11 is acceptable if dependency compatibility requires it. |
| API | FastAPI + Pydantic v2 | Typed request/response contracts and generated OpenAPI docs. |
| Local metadata & sparse retrieval | SQLite + FTS5 + SQLAlchemy 2 + Alembic | Zero server fees; FTS5 includes a BM25 ranking function. Use parameterized queries and correct BM25 score ordering. |
| Vector retrieval | `qdrant-client` in persisted local mode | No Docker for the MVP, same client concepts when migrating to a Qdrant server. Local mode is only for relatively small corpora (the client warns above ~20,000 points). |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` through Sentence Transformers | Lightweight English baseline, runs on CPU. Model choice and embedding dimension must be versioned in index profiles. |
| Reranking | `cross-encoder/ms-marco-MiniLM-L6-v2` through Sentence Transformers | Local second-stage reranking of top-N candidates. Optional and lazy-loaded. |
| Parsing | Plain text/Markdown + PyMuPDF for text PDFs; `python-docx` optional | Preserve document/page/section provenance. Do not silently claim full OCR or reliable table extraction. |
| Advanced parsing (later) | Docling as an optional plugin | Better structured extraction where appropriate, but avoid heavy mandatory dependencies for the MVP. |
| LLM generation (optional) | Ollama HTTP API using HTTPX | Local-only; no paid provider or API key needed. Retrieval/demo continues if Ollama is unavailable. |
| UI | Streamlit consuming FastAPI endpoints | Fast to demonstrate comparison workflows; do not duplicate backend business logic inside Streamlit. |
| Testing/quality | pytest, httpx, Ruff, mypy or Pyright where practical | Deterministic fixtures; integration tests run without model downloads. |
| Evaluation | Pure-Python metric implementations, validated on hand-calculated fixtures; BEIR-compatible qrels importer | Avoid packages requiring complicated Windows compilation. Record corpus/qrels versions. |
| Logs & benchmarks | Structured JSON logs, SQLite experiment metadata, exportable JSON/CSV | Compare strategies reproducibly without managed observability services. |

**Avoid in v1:** LangChain/LlamaIndex orchestration, Kubernetes, Redis, Celery, hosted embedding APIs, cloud databases, multi-agent frameworks, a separate authentication service, and a React frontend. Add infrastructure only when a measured requirement justifies it.

## 3. Target architecture

```text
                  Streamlit experiment UI
                            |
                            v
                    FastAPI HTTP API
                      /     |      \
                     /      |       \
      ingestion & index   retrieval   eval + optional QA
             |               |               |
     parse -> chunks        query        qrels + metrics
             |          /     |     \          |
      SQLite metadata    BM25  Dense  Hybrid   report
             |             |      |     |
      SQLite FTS5 <-------+      |     | 
      Qdrant local <-------------+     |
                                    RRF
                                     |
                               optional reranker
                                     |
                               cited search hits
                                     |
                              optional Ollama LLM
```

Architecture must be driven by explicit interfaces, not by a framework pipeline:

- `DocumentParser`: bytes/path + MIME type -> ordered parsed blocks with provenance.
- `Chunker`: parsed document + configuration -> chunks and section/page metadata.
- `EmbeddingProvider`: list of strings -> vectors with explicit model/version/dimension.
- `SparseRetriever`: query + profile/filters -> ranked hits.
- `DenseRetriever`: query + profile/filters -> ranked hits.
- `Reranker`: query + candidates -> reranked hits.
- `AnswerGenerator` (optional): prompt/context -> answer.
- `EvaluationRunner`: corpus/query/qrels + retrieval configuration -> metrics and per-query diagnostics.

Use simple `Protocol` interfaces or abstract classes only where multiple concrete implementations exist. No interfaces added merely for appearances.

## 4. Proposed repository structure

This is the **target end state**, not a request to scaffold empty placeholder files. Create modules incrementally as implemented.

```text
rag-workbench/
├── AGENTS.md
├── PROJECT_SPEC.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .env.example
├── .gitignore
├── src/
│   └── rag_workbench/
│       ├── __init__.py
│       ├── config.py
│       ├── cli.py
│       ├── domain/
│       │   ├── schemas.py
│       │   └── interfaces.py
│       ├── persistence/
│       │   ├── database.py
│       │   ├── models.py
│       │   └── migrations/          # Alembic, if selected
│       ├── ingestion/
│       │   ├── parsers.py
│       │   ├── chunkers.py
│       │   └── pipeline.py
│       ├── retrieval/
│       │   ├── bm25.py
│       │   ├── dense.py
│       │   ├── hybrid.py
│       │   ├── reranker.py
│       │   └── service.py
│       ├── generation/
│       │   ├── ollama.py
│       │   └── grounding.py
│       ├── evaluation/
│       │   ├── datasets.py
│       │   ├── metrics.py
│       │   ├── runner.py
│       │   └── diagnostics.py
│       └── api/
│           ├── app.py
│           └── routes/
│               ├── documents.py
│               ├── search.py
│               ├── answers.py
│               └── evaluations.py
├── ui/
│   └── streamlit_app.py
├── tests/
│   ├── fixtures/
│   ├── unit/
│   └── integration/
├── benchmarks/
│   ├── configs/
│   └── outputs/               # generated outputs ignored by Git
├── docs/
│   ├── architecture.md
│   ├── methodology.md
│   ├── failure_analysis.md
│   └── scaling_to_10m.md
└── data/                        # ignored: source docs, indexes, downloads
```

**Rule:** Streamlit makes HTTP requests to the API. CLI, API and experiments all call the same core Python services.

## 5. Data model and correctness requirements

Persistent entities:

- `Document`: document ID, dataset/collection ID, sanitized display name, source checksum (SHA-256), MIME type, byte size, import time, ingestion status, and parsing diagnostics.
- `IndexProfile`: immutable/config-hashed combination of parser version, chunker strategy/settings, embedding model ID/dimension and index version. Different chunking configurations must never contaminate each other's indexes.
- `Chunk`: integer DB primary key, stable external UUID, document ID, profile ID, order, text, character offsets if available, page range if available, section path if available, and source provenance.
- `IngestionJob`: document/profile, queued/running/completed/failed state, progress, timings and a clear error code.
- `EvaluationRun`: corpus version, qrels version, full retrieval/reranking configuration, seed, model identifiers, timestamps, summary metrics and per-query results.

Critical behaviors:

1. Re-importing identical document bytes into the same profile is idempotent; report duplicate rather than creating repeated chunks.
2. Edited documents generate a new version/checksum and permit correct reindexing; deletion removes metadata, FTS rows and vector points for that document/profile.
3. SQLite FTS5 uses chunk row IDs consistently. Use triggers or explicit transactionally maintained tables; prove synchronization after create/update/delete.
4. Qdrant and SQLite do not share a transaction. Stage ingestion and track state; retry/reconcile partial failures rather than silently displaying a half-indexed document.
5. Each hit identifies its `chunk_id`, `document_id`, source filename, page range where possible, section title/path where possible, rank and retrieval scores. Missing page information must be represented as unknown, not invented.
6. Full-text user input is untrusted. Escape/build FTS expressions safely, bind SQL parameters, limit upload size and allowed file types, and never execute uploaded content.
7. Do not transmit document text to any external provider without an explicit opt-in. Default to localhost inference only.
8. Lazy-load heavy embedding/reranking models so API startup and BM25-only workflows remain lightweight.
9. CPU-only mode is first-class. Batch embedding work and expose progress. No GPU is assumed.
10. For the initial local prototype, support one writer process; document concurrency limitations clearly.

## 6. Chunking strategies and legal contracts

Implement three comparable strategies:

- **Fixed-token:** Tokenizer-aware windows with configurable max tokens and overlap. Specify exactly which tokenizer counts tokens; character counting is not a token count.
- **Recursive/paragraph-aware:** Preserve paragraphs and headings where possible; split overly long paragraphs; add limited context overlap without duplicating entire sections.
- **Clause-aware legal:** Identify numbered headings (such as `2.1`, `2.1(a)`), definitions, section titles, and exhibits via conservative heuristics. Keep clause boundaries when possible, preserve a hierarchical `section_path`, and split long clauses only when necessary. Treat ambiguous parsing as a documented limitation; do not present regex rules as a universal contract parser.

Compare short, medium and long chunk budgets using identical source documents and equivalent query sets. Capture chunk counts, retrieval performance and ingestion cost. Store chunking configuration and model provenance with each experiment.

## 7. Retrieval and reranking

Modes to implement:

1. **BM25** over SQLite FTS5; document that FTS5 `bm25()` ranks *lower numerical scores as better*. Normalize ranking direction at the interface boundary.
2. **Dense** using Sentence Transformers embeddings stored in persisted Qdrant local mode. Use an explicit distance metric and a persistent profile-to-collection mapping.
3. **Hybrid** using Reciprocal Rank Fusion (RRF) of BM25 and dense candidate rankings. Default `rrf_k=60`, configurable. Do **not** mix raw BM25 and cosine scores by simple addition. Deduplicate by stable chunk ID, preserve each original rank, and use stable tie-breaking.
4. **Hybrid + rerank** using local cross-encoder scores over a bounded top-N candidate set (default N=20). Measure rerank latency separately.

For all modes, same query, same dataset/index profile, same allowed source filters and comparable top-K. Profiles with different chunkers require separate indexes and explicit comparison labels. Capture stage timings: parsing, chunking, embedding, indexing, sparse retrieval, dense retrieval, fusion, reranking, context assembly and optional generation.

## 8. Optional grounded answer generation

Later stage only, via locally running Ollama:

- Answers must derive from retrieved passages, with clickable/inspectable `[S1]`, `[S2]` identifiers tied to actual chunks and provenance.
- The system must refuse to fabricate citations. If supporting evidence is absent, return a clear insufficient-evidence response.
- Verify that every returned citation ID refers to a retrieved passage; distinguish syntactically valid citations from genuinely supported claims (which require separate evaluation).
- Display raw retrieval evidence alongside answer text and a compact trace of the retrieval configuration.
- Retrieved document text is **data**, not authoritative system instructions. Do not allow document content to override developer instructions or trigger tool use.
- If Ollama is not installed/running, search, ingestion and evaluation continue working normally.

## 9. Evaluation design (the heart of the portfolio)

**First benchmark:** BEIR `scifact`, because it includes a manageable corpus, queries and relevance judgments (`qrels`). Do not download the entire BEIR suite by default. Include a small committed, legally redistributable synthetic fixture to run tests entirely offline.

**Second showcase:** Public commercial contracts such as CUAD, for document parsing/chunking and qualitative retrieval inspection. CUAD's QA annotations are not automatically BEIR-style relevance judgments; construct and document a small manually verified query-to-relevant-document/passage dataset. Respect dataset licenses and avoid checking copyrighted document dumps into the repository.

Metrics:

- `Recall@5`, `Recall@10`: proportion of known relevant items retrieved by cutoff.
- `MRR@10`: reciprocal rank of first relevant retrieved item.
- `nDCG@10`: ranked relevance quality against qrels.
- `Precision@5` as supplementary.
- P50/P95 retrieval latency, reranker latency, indexing throughput, corpus/chunk counts, model sizes and rough peak process memory if measured.
- Explicit error counts: no relevant item retrieved, retrieved but discarded during reranking/context selection, answer unsupported despite relevant evidence, extraction/index errors.

**Important qrels granularity rule:** BEIR relevance labels are usually at the **document level**. A search hit is a **chunk**. Aggregate chunk rankings to unique parent-document rankings before computing BEIR document-level metrics, and report any passage-level metrics only when actual passage labels exist. Do not inflate Recall@K by counting repeated chunks from the same relevant document.

Each run should record the exact index profile, embedding model, reranker model, query set, qrels, hyperparameters, software version and timestamp; export both machine-readable JSON and a comparison CSV.

Compare at minimum:

- BM25 alone
- Dense alone
- Hybrid (RRF)
- Hybrid + reranker
- At least two chunking strategies using identical documents

Do not assert hybrid or reranking *must* win. Publish negative results honestly and analyze why.

## 10. RAG failure explorer

For each evaluation query, display:

- Gold relevant documents/answer, where available.
- BM25 candidates, dense candidates, fused candidates, reranked candidates.
- Overlaps, missing gold documents, and score/rank movements by stage.
- Whether gold evidence was indexed; whether it was retrieved in candidate top-N; whether a reranker demoted it; whether context truncation excluded it; whether generated statements lack supporting evidence.
- Timing per retrieval stage and the complete configuration.

Failure categories should be rule-based where observable, with an `unknown / requires human review` fallback. Do not pretend to infer root causes that were not measured.

## 11. API sketch

Implement endpoints incrementally; validate with typed Pydantic models and OpenAPI:

```text
GET    /health
POST   /v1/documents                 # local file upload; returns doc/job ID
GET    /v1/documents                 # paginated, indexed/profile status
GET    /v1/documents/{document_id}
DELETE /v1/documents/{document_id}   # scoped to requested profile or all versions
GET    /v1/ingestion-jobs/{job_id}
POST   /v1/search                    # query, profile, mode, top_k, rerank bool
POST   /v1/answer                    # optional Ollama; grounded citations
POST   /v1/evaluations               # benchmark + configs
GET    /v1/evaluations/{run_id}
```

Response of `/v1/search` must contain configuration fingerprint, elapsed timings and a list of cited, provenance-rich hits. Set helpful size/time bounds and report actionable errors.

For MVP, **prefer a synchronous CLI ingestion command** (`rag-workbench ingest ...`) and explicit API behavior over using FastAPI's in-process `BackgroundTasks` for long CPU-intensive document embedding. When a UI ingestion job queue is needed, introduce one persistent SQLite-backed single-worker loop; avoid starting an uncontrolled thread per request. Maintain clear job states.

## 12. UI: Streamlit “Retrieval Laboratory”

Keep the UI a thin API client. End-state navigation:

- **Corpus Manager:** import documents, display ingestion progress, source and chunk counts; inspect indexed chunks and metadata.
- **Search Playground:** enter a query, choose BM25/dense/hybrid, select chunking profile, toggle reranking, set top-K; display evidence and per-stage scores/timings.
- **Compare:** run the same query with multiple modes side by side; visually highlight rank changes and citations.
- **Evaluation Lab:** choose dataset and stored profiles; execute or inspect benchmarks; compare metrics and export CSV/JSON.
- **Failure Explorer:** investigate a low-performing query and identify evidence loss by stage.
- **Ask (optional):** locally generated answer with inspectable source evidence and an abstention path.

Aim for a clean research-tool aesthetic, not a decorative generic chatbot. Show provenance prominently and avoid fake progress bars or placeholder metrics.

## 13. Build plan and acceptance tests

**Phase 0 — Environment and foundation**

- Initialize a typed, installable Python project with `uv`, Ruff, pytest, configuration via environment variables, `.env.example`, `.gitignore`, CI workflow and a documented Windows setup.
- Create minimal FastAPI `/health`, Streamlit shell calling `/health`, SQLite initialization, and a deterministic unit test.
- **Exit criteria:** fresh clone installs with one documented `uv sync` workflow; `uv run pytest`, `uv run ruff check .` pass; API and UI launch locally without Docker, Ollama, model downloads or API keys.

**Phase 1 — Ingestion and BM25 baseline**

- Add TXT/MD and text-based PDF parsing with graceful extraction failures; deterministic fixed-token and paragraph-aware chunkers; SQLite metadata + FTS5 indexing; CLI ingestion; BM25 `/v1/search` and cited results.
- Add deterministic tests for chunk boundaries, empty/bad files, idempotent reingestion, deletion, provenance, FTS5 synchronization and SQL/FTS input robustness.
- **Exit criteria:** import a sample contract, search an exact clause term and inspect an authentic citation. Complete round-trip test runs without an ML model or internet.

**Phase 2 — Dense and hybrid retrieval**

- Add local embeddings, Qdrant-local vector persistence, profile isolation, dense search, RRF hybrid, delete/rebuild synchronization and stage timings.
- **Exit criteria:** BM25, dense and hybrid return reproducible rankings from the same corpus and are individually testable. Reopening the app preserves indexes.

**Phase 3 — Legal chunking and reranking**

- Add clause-aware chunker, local cross-encoder, configurable candidate count, ranked-comparison UI and reranker latency reporting.
- **Exit criteria:** compare at least two chunking strategies and measure how reranking changes ranking and latency. Failed clause detection is visible rather than silently incorrect.

**Phase 4 — Evaluation and debugging**

- Import a manageable BEIR SciFact sample, preserve source IDs and document-level qrels, calculate metrics, render comparisons, save runs; add per-query error explorer.
- Add a small human-checked legal-contract query set with documented provenance and licensing.
- **Exit criteria:** one documented command generates a reproducible benchmark report for BM25/dense/hybrid/hybrid+rerank; every metric has unit tests, and document/chunk IDs are correctly handled.

**Phase 5 — Optional answer generation and portfolio polish**

- Optional local Ollama integration, source-linked answers, citations validation, insufficient-evidence handling, screenshots, architecture diagram, limits, failure case writeup, basic profiling, scaling design.
- **Exit criteria:** robust behavior if Ollama is stopped, honest answer/citation limitations, clean README demo, benchmark tables and clear measured-vs-theoretical claims.

**Instructions for first Codex session:** implement **Phase 0 and Phase 1 only**. Do not scaffold the later phases as unimplemented code or install model libraries before needed. After those phases, stop and report tests, commands, decisions and known gaps.

## 14. Scaling design: interview scenario, not performance fiction

Maintain a separate `docs/scaling_to_10m.md` addressing:

- Assumptions: mean pages per document, chunks per document, embedding dimension, update rate, query rate and replication requirement.
- Raw vector size formula: `num_documents * mean_chunks_per_document * embedding_dimensions * bytes_per_dimension`, plus explicit ANN index, metadata, replicas and free-space overhead.
- Pipeline scaling: asynchronous durable job queue, bounded worker concurrency, batched embeddings, retries, poison-document quarantine, backfills and incremental reindexing.
- Storage/index scaling: migrate Qdrant local to Qdrant server/cluster, SQLite metadata to PostgreSQL, and SQLite FTS5 to an appropriate scalable lexical search engine (e.g. OpenSearch); preserve interfaces so migration is local to adapters.
- Query path: filtered candidate retrieval, dense ANN tuning, parallel lexical/vector retrieval, fusion, bounded reranking and cacheable embeddings.
- Lifecycle: tenant/source access controls before retrieval, delete propagation, embedding model version migrations, index swaps, rollout and rollback.
- Observability and SLOs: ingestion throughput, index freshness, Recall@K, latency percentiles, index sizes and operational cost.
- **Do not claim capacity without measurements.** Load-test manageable synthetic data, record the actual limit encountered and model larger capacity separately.

## 15. Completion criteria / portfolio README

The final README should include:

1. Problem statement and architecture diagram.
2. A short local setup, demo dataset and commands that work without paid APIs.
3. Side-by-side BM25/dense/hybrid/rerank benchmark table including latency and relevance, with corpus/qrels sizes.
4. A contract-clause chunking example with real page/section citations.
5. One concrete documented debugging incident where a bad answer/search result is traced to a specific stage.
6. Limitations, failures, reproducibility details and a scaling design for 10 million documents.
7. Screenshots or a small demo recording, licenses/attribution, security/data-handling limitations, and setup/troubleshooting notes for Windows.

This is an engineering research platform, not a SaaS. Optimize for correctness, reproducibility, transparent trade-offs and a convincing interview demonstration.
