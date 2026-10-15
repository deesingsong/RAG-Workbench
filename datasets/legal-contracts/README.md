# Synthetic Legal Contracts (evidence-unit retrieval set)

A small, **fully synthetic** retrieval-evaluation dataset of commercial contracts.
Every sentence of every contract was authored for RAG Workbench. There is **no
third-party copyrighted material** in this directory, and nothing here is derived
from a real agreement. Because the text is original, the dataset is redistributable
under **CC0-1.0**.

This set exists to exercise clause-anchored retrieval: the unit of relevance is a
**clause** (an evidence unit), not a whole contract. It is the reference fixture for
the `clause_aware` chunker and for `EVIDENCE_MAPPING` experiments.

## Contents

```
generate.py       Original authoring script; regenerates every file below.
corpus.jsonl      7 contracts.
queries.jsonl     48 queries.
qrels.tsv         Query -> evidence-unit relevance judgments.
evidence.jsonl    65 evidence units (one per clause).
manifest.json     Counts, checksums, provenance and relevance-mapping policy.
```

Regenerate with the project interpreter:

```powershell
.\.venv\Scripts\python.exe datasets\legal-contracts\generate.py
```

`generate.py` rebuilds the dataset in memory from the inline contract and query
content, validates it, and writes the files with `write_dataset`. `manifest.json`
and all checksums are produced by that call; `manifest.json` is never hand-edited.

## File formats

- **corpus.jsonl** — one JSON object per contract: `{"_id", "title", "text"}`.
  Each `text` is a realistic agreement excerpt with clearly numbered clauses. The
  numbering scheme varies per contract (integers, decimals, roman numerals,
  letters) so clause-boundary detection is tested for real diversity.
- **queries.jsonl** — one JSON object per query: `{"_id", "text"}`, ids `lq-001`…
- **qrels.tsv** — tab-separated `query-id<TAB>corpus-id<TAB>score` with a header
  row. `corpus-id` is an evidence-unit id. Graded `1` for relevant, and `2` for
  the single best/primary unit when a query has more than one relevant clause.
- **evidence.jsonl** — one JSON object per evidence unit:
  `{"_id", "contract", "section", "text"}` where
  - `_id` is `"<contract_id>::<section_number>"` (e.g. `nda-01::5`),
  - `contract` is the corpus id of the contract,
  - `section` is the clause number string,
  - `text` is an exact verbatim passage from that contract's `text` (findable as a
    substring after normalisation).

## Relevance-mapping policy

The dataset uses `EVIDENCE_MAPPING` (`relevance_mapping: "evidence"`). A retrieved
chunk counts as relevant for an evidence unit when **either**:

1. the chunk contains the unit's anchor `text` (after normalisation:
   `" ".join(s.lower().split())`), **or**
2. the chunk carries that unit's section number within the same contract.

This keeps the judgment chunker-neutral: the ground truth is the clause, so a
whole-contract chunk and a clause-sized chunk are measured by the same rule rather
than one chunker being privileged.

**Relevance is judged at the clause / evidence-unit level — never "every chunk of a
relevant contract".** A chunk from the right contract but the wrong clause is not
relevant, and a query is scored against the specific units listed in `qrels.tsv`.

## Query mix

The 48 queries deliberately cover: exact terminology, semantic paraphrase,
multi-clause questions, ambiguous terminology, references to a specific section
number, exception / "subject to" questions, and queries with more than one relevant
evidence unit (14 of them, graded `2` on the primary unit).

## Limitations

- Small and synthetic: it demonstrates retrieval mechanics, not legal accuracy.
- English-only, and the clause styles are those of common-law commercial drafting.
- The evidence anchors are chosen clause passages, not an exhaustive gold standard
  for every conceivable question over these contracts.
