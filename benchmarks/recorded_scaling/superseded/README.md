# Superseded scaling artifacts

These two artifacts are **preserved for history only**. They were the first
committed Phase-6 scaling runs, but their `git_commit` field named revision
`5c22ef6`, which predates the scaling harness: the measurements were produced from
an **uncommitted working tree**, so `HEAD` was not the code that actually ran.

They are kept for history rather than deleted, and are explicitly marked
superseded. Their measured numbers and provenance are unchanged; the one
incidental absolute temp path in each `limitations` string was sanitized to a
neutral wording so the repository carries no local filesystem paths.
The live, current artifacts live in the parent directory and were regenerated
against committed code with explicit provenance (`git_commit`, `git_dirty`,
`provenance`). The loader (`rag_workbench.scaling.recorded`) globs `*.json`
non-recursively, so this subdirectory is not read by the API or the demo.

See Case 7 in [`docs/case_studies.md`](../../../docs/case_studies.md).
