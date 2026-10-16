# Engineering case studies

Concrete failures and design traps found while building RAG Workbench, with the
observable evidence, the root cause, the fix or decision, and the residual risk.
Each case is grounded in a committed artifact, test or recorded measurement — none
are illustrative fiction.

---

## Case 1 — Chunking strategy, not the token budget, dominates chunk count

**Symptom.** A "bigger chunks should mean fewer chunks" intuition failed. A
`fixed_token` run with a **512**-token budget produced *more* chunks than a
`paragraph` run with a **220**-token budget.

**Evidence** (`benchmarks/recorded_scaling/`, 5,000 documents, seed 0):

| Strategy | Token budget | Chunks | Chunks/doc |
|---|---:|---:|---:|
| `paragraph` | 220 | 38,709 | 7.74 |
| `fixed_token` | 512 | 90,470 | 18.09 |

**Root cause.** Fixed-token windows never cross a parsed block boundary. On
paragraph-structured text each paragraph is its own block (~55 words), so the
512-token window degenerates to one chunk per paragraph and never packs two
paragraphs together. The paragraph chunker, by contrast, packs multiple blocks up
to the 220-token budget.

**Fix / decision.** None in code — the behaviour is correct and documented
(`chunkers.py`). The lesson is operational: **estimate chunk count from the
block/section structure of the real corpus, not from the token budget.** The
capacity model's `chunks_per_document` is the single most influential assumption,
which is why it is a first-class input and why the `small`/`typical`/`large`
scenarios exist.

**Residual risk.** The synthetic corpus is more uniformly paragraph-shaped than a
real corpus, so its chunk-count ratios may not transfer. Measure on a real sample.

---

## Case 2 — `or` queries with common terms drive BM25 cost

**Symptom.** BM25 p50 latency rose from ~4 ms at 687 chunks to ~112 ms at 38,709
chunks (`measured here`), much faster than the ~56× growth in index size would
suggest for a well-behaved index scan.

**Evidence** (`scale-20261016T092334.json`):

| Chunks | @10 p50 | @100 p50 |
|---:|---:|---:|
| 687 | 3.8 ms | 6.5 ms |
| 7,774 | 21.1 ms | 26.3 ms |
| 38,709 | 112.2 ms | 105.0 ms |

**Root cause.** The generated queries combine a document-distinctive token with a
common corpus term. Under `query_mode="or"`, the common term matches a large
fraction of the corpus, so each query scores and ranks a large candidate set. Cost
tracks the number of matching documents, not the index size alone.

**Fix / decision.** Keep `or` as the evaluation default (long natural-language
queries need it) but treat query-time cost as a **function of term selectivity**.
At scale this is one more reason a single-node FTS5 index is untenable: there is no
sharding and no candidate cap that changes the fundamental scan.

**Residual risk.** The exact curve depends on term distribution. A production
corpus with rarer terms would show a gentler slope; the qualitative conclusion —
cost scales with match volume — holds.

---

## Case 3 — A citation that *contains* a valid id is still malformed

**Symptom.** During the Phase 5 answer audit, some answers cited passages correctly
in spirit but failed verification unpredictably.

**Root cause.** The original citation parser accepted any string that *contained*
an `S#` token, so `"S1 and S2"`, `"see S3."` or `"S1,S2"` could be accepted or
rejected depending on incidental formatting. That made the citation metric depend
on model punctuation rather than on whether a claim was actually linked to a
passage.

**Fix.** `normalize_source_id` now requires the **whole** string to be a single
identifier (`^[sS][0-9]{1,3}$`) after stripping at most one bracket pair and a
trailing period. Anything else is recorded as `malformed_citation` and rejected,
and `CitationValidation` separates the mechanical properties
(`all_citations_resolved`, `all_claims_cited`, `citation_valid`) from the
undetermined semantic one (`answer_claims_consistency = not_determined`).

**Residual risk.** Structural citation validity says nothing about whether the
cited passage **supports** the claim. `semantic_support` is never asserted
deterministically; it requires a human or a separate model-based check.

---

## Case 4 — A recorded abstention rate is a draw, not a property

**Symptom.** The first committed answer-quality run recorded a
`correct_abstention_rate` of **0.0**, implying the small local model never
correctly abstained.

**Root cause.** The Ollama model had been re-imported between runs
(`modified_at` moved 2026-10-15 → 2026-10-16), and a 1.5B model's abstention
behaviour is not stable across runs. The 0.0 was one non-reproducible draw.

**Fix / decision.** The original report is preserved **unedited** and annotated as
a single draw; an improved configuration (same prompt variant, `max_evidence=6`,
`seed=0`) was recorded as an **additional** artifact and marked `recommended`.
Reports are never overwritten, and the recommended report is what the demo loads
by default.

**Residual risk.** Even the improved run is a sample. A model's abstention rate on
a small quality set is indicative, not a guarantee; the UI states this explicitly.

---

## Case 5 — Missing optional dependencies must degrade, not break

**Symptom.** Dense, hybrid and reranked retrieval, plus answer generation, all
depend on optional extras (Sentence Transformers/torch, a pulled Ollama model).
The public demo has none of them.

**Root cause / decision.** Rather than fail, each optional stage reports a
**typed, actionable** unavailability: `503 dense_unavailable`,
`503 vector_index_not_built`, `503 reranker_unavailable`,
`503 generator_unavailable`. The core BM25 path, the tests and the demo run with
no model, no download and no keys. The scalability harness follows the same rule:
asking for `dense`/`hybrid` modes without the extra raises an actionable
`dense_unavailable` error instead of crashing mid-run.

**Residual risk.** "Degrades gracefully" is a claim about *availability*, not
*quality*: a user who mistypes a mode gets a clear error, but a user who never
installs the extra gets BM25 only, which may be worse for their corpus.

---

## Case 6 — Candidate-depth semantics must agree across the runner and the service

**Symptom.** Some evaluation configurations failed during execution instead of at
validation, when `candidate_depth` exceeded the number of candidates actually
passed to the reranker.

**Root cause.** The evaluation runner computed a retrieval depth the search
service did not accept (`rerank_candidates >= top_k` was required but not
enforced consistently), so a plausible configuration could retrieve a pool the
reranker stage then silently could not use.

**Fix.** Four quantities were made explicit and consistent — first-stage candidate
depth, reranker input, final result count, and candidate-recall depth — with
validation that rejects unsupported configurations with an actionable `422`
rather than failing mid-run. Candidate recall is always measured on a rerank-free
retrieval, so reranking can never change pool coverage.

**Residual risk.** The semantics are correct but subtle; a single place
(`EvalRunConfig` validation) must remain the source of truth as stages are added.

---

## Case 7 — A benchmark recorded the wrong code revision

**Symptom.** The first committed scaling artifacts named Git revision `5c22ef6`,
which predates the scaling harness itself. The harness recorded `git rev-parse
HEAD` and nothing else, so a run made from a **dirty working tree** silently
attributed its numbers to whatever commit happened to be checked out.

**Root cause.** `HEAD` identifies the last commit, not the code that actually ran.
During Phase 6 the harness code was uncommitted, so `HEAD` pointed at the previous
phase while the measurements came from new, untracked code.

**Fix.** The harness now records both the commit and whether the working tree was
clean (`git_dirty`) plus a human-readable `provenance`. A dirty or unknown tree
adds an explicit limitation and the repository keeps the superseded artifacts under
`benchmarks/recorded_scaling/superseded/`; the live artifacts are regenerated
against committed code. `SCALING_SCHEMA_VERSION` was bumped to 2 to mark the change.

**Residual risk.** Revision provenance depends on `git` being available at record
time; when it is not, the harness records `unknown` rather than claiming a
revision it could not verify.

---

## Case 8 — "Peak RSS" that was one instantaneous reading

**Symptom.** A memory figure in the scaling record was labelled `peak_rss_mb`, but
the harness read `memory_info().rss` exactly once at the end of a workload. That is
an instantaneous value, not a peak, and it silently changed meaning depending on
when the probe ran.

**Root cause.** Three different quantities were collapsed into one name: an
instantaneous process RSS reading, a true sampled peak, and a Python-heap peak from
`tracemalloc`.

**Fix.** They are now separate fields with distinct labels: `rss_mb` (one
instantaneous reading), `sampled_peak_rss_mb` (maximum of periodic readings taken
by a background sampler), and `tracemalloc_peak_mb` (Python-managed allocations
only). When `psutil` is absent, process RSS is reported as **not measured** rather
than substituted, and the record states that `tracemalloc_peak_mb` is not total
process memory. The Capacity/Memory views state the same distinction.

**Residual risk.** Even a sampled peak is not the OS high-water mark; it is the
maximum of whatever the sampler happened to observe.
