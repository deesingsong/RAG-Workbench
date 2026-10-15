# AGENTS.md — Codex instructions for RAG Workbench

## Mission
Implement `PROJECT_SPEC.md` in verifiable milestones. This is a **local-first, zero-mandatory-cost RAG retrieval laboratory**, not a ChatGPT clone. Demonstrate retrieval engineering with reproducible tests, honest benchmarks and source provenance.

## Scope control
- **In the first session, implement only Phase 0 and Phase 1** from `PROJECT_SPEC.md` and stop. Later-phase modules should be created only when their functionality is implemented; do not create empty class hierarchies or TODO-filled files.
- Prefer Python 3.12, `uv`, FastAPI, SQLite/FTS5, SQLAlchemy, pytest, Ruff, PyMuPDF and Streamlit. Do not require Docker, cloud accounts, API keys, Ollama or model downloads for Phases 0-1.
- Use straightforward Python with clear typing, small functions and explicit control flow. Avoid unnecessary wrappers, factory patterns, deep inheritance and extra orchestration frameworks. No LangChain/LlamaIndex unless requested.

## Working process
1. Inspect the repository before editing; preserve working code.
2. State a short implementation plan and any relevant assumptions.
3. Implement a **vertical slice** (one functional ingestion/search path) before adding UI refinements.
4. Add/adjust tests with each behavior; validate the actual program rather than mocking the feature under test.
5. Run formatting, linting and tests. Fix failures before calling a phase complete.
6. Update `README.md` with exact Windows PowerShell setup/run commands and any limitations.
7. At completion, report files touched, commands run, test outcomes, observed behavior, unfinished items and the suggested next milestone. Do not claim tests or benchmarks that were not run.

## Domain boundaries
- `src/rag_workbench/`: all ingestion, persistence, retrieval, configuration and API business logic.
- `ui/`: a thin Streamlit HTTP client of FastAPI; no duplicated retrieval logic.
- `tests/`: deterministic fixtures and behavioral tests. Unit and integration tests must not require internet or an actual LLM.
- `data/`, corpus documents, caches, model weights and generated benchmark results are ignored by Git except tiny redistributable fixtures.
- Keep parsed blocks/chunks and provenance explicit. Never invent a source page, section, quote or citation.

## Correctness and security
- SQLite FTS5 BM25 scores sort **ascending**; provide a single consistent rank convention in application responses.
- Parameterize SQL; sanitize/escape user-provided FTS search terms and handle FTS syntax errors gracefully.
- Preserve source metadata, original document checksum and deterministic profile-specific chunk IDs.
- Ingesting the same file/profile twice must not create duplicate chunks. Deletions must remove FTS index entries as well as relational rows.
- Handle malformed PDFs, empty extracted text, repeated headings and overly large uploads without crashing the API.
- Never call a paid API or upload user documents to external services. Never require a secret for the default local demo.
- Don't add fake results, simulated metrics represented as real tests, arbitrary safety claims or unsupported scale claims.
- If a feature is optional (e.g. Ollama or reranker), a missing dependency must not break unrelated features.

## Architecture and performance
- Implement `parser -> chunker -> SQLite documents/chunks -> FTS5 -> BM25 search` as the first working vertical slice.
- Make configuration explicit, with config hashes/versioned index profiles and no mixing of chunking experiments.
- Use one local writer initially. Document the absence of distributed transactions, concurrent ingestion and public-facing authentication.
- Streaming, external queues, Qdrant, cross-encoder and LLM generation belong to later phases, not Phase 0-1.

## Definition of done for the initial session
- A fresh local install works with documented PowerShell commands.
- `/health` responds; Streamlit displays live API status.
- CLI can ingest a sample TXT/MD/PDF, with consistent document and chunk metadata.
- BM25 search returns relevant excerpts with real filename/page/section metadata *when extractable*.
- Deterministic pytest tests cover chunking, idempotency, FTS5 search/deletion, provenance and invalid inputs.
- Ruff and pytest pass, and the README does not promise unfinished capabilities.
