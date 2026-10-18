# Grounded-answer quality dataset

`legal-contracts-answers.json` is a small, fully offline grounded-answer evaluation
set over the bundled synthetic `legal-contracts` corpus. It measures the **answer
pipeline**, not legal accuracy:

- can the pipeline produce schema-valid structured answers?
- do the citation ids it emits resolve to passages that were actually selected?
- does it abstain when the passages do not contain the answer?
- was the reference evidence selected into the model context?

Each question carries an `expected_status` (`answered`, `partially_answered`,
`insufficient_evidence`), a manually verified `reference_answer`, `reference_keywords`
and `reference_sections` that define the question's **required evidence units** (the
lexical anchors a grounded answer depends on).

Coverage is measured per unit, not with a single `any()` check, so multi-part
questions cannot hide a missing half. Each question is classified as:

- **`included`** — every required anchor reached the model context;
- **`retrieved_but_excluded`** — every anchor was *retrieved* but at least one was
  dropped from the context (by `max_evidence` or the token budget);
- **`never_retrieved`** — at least one required anchor was absent from the whole
  candidate pool.

Complete and partial coverage are reported separately. Two reference cases are
worth inspecting (`README` §5.4): `aq-002` is a genuine
`retrieved_but_excluded` (fixed by `max_evidence=6`), and `aq-007` has *complete*
coverage yet still receives a wrong answer — showing that coverage is **not**
correctness.

It deliberately does **not** score whether a cited passage truly *supports* a claim:
that is semantic support, which is not deterministically decidable, and is reported
as `not_determined` rather than guessed. It likewise reports whether the free-text
answer matches the structured `claims` as `not_determined`.

Run it (requires a local Ollama and `RAG_WORKBENCH_GENERATION_ENABLED=true`):

```powershell
uv run rag-workbench answer quality --profile <profile-id> `
  --dataset answer-quality/legal-contracts-answers --query-mode or --max-evidence 6 --seed 0
uv run rag-workbench answer record  --profile <profile-id> `
  --dataset answer-quality/legal-contracts-answers --query-mode or --max-evidence 6 --seed 0
```

`answer record` writes the artifacts to `benchmarks/recorded_answers/`; pass
`--no-demos` to write only the aggregate report for a configuration sweep. The
observed metrics and their interpretation are in the README §5.4.

This directory has no `manifest.json`, so the evaluation dataset lister (which
requires the BEIR-style manifest) ignores it.
