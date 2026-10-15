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
used for a deterministic "was reference evidence selected?" check, and the
`reference_sections` it was verified against.

It deliberately does **not** score whether a cited passage truly *supports* a claim:
that is semantic support, which is not deterministically decidable, and is reported
as `not_determined` rather than guessed.

Run it (requires a local Ollama and `RAG_WORKBENCH_GENERATION_ENABLED=true`):

```powershell
uv run rag-workbench answer quality --profile <profile-id> `
  --dataset answer-quality/legal-contracts-answers --query-mode or --max-evidence 4
uv run rag-workbench answer record  --profile <profile-id> `
  --dataset answer-quality/legal-contracts-answers --query-mode or --max-evidence 4
```

`answer record` writes the committed artifacts to `benchmarks/recorded_answers/`.
The observed metrics and their interpretation are in the README §16.

This directory has no `manifest.json`, so the evaluation dataset lister (which
requires the BEIR-style manifest) ignores it.
