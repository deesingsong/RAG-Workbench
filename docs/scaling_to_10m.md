# Designing RAG Workbench for 10 million documents

This document is a **design case study**, not a claim about the running system. The
local prototype was measured at bounded corpus sizes on a single machine (see
[Measured baselines](#4-measured-baselines)); everything beyond that is explicit
extrapolation, published vendor behaviour, or an assumption that must be replaced
by a production benchmark. Numbers that come from measurement are labelled
`measured here`; numbers that come from arithmetic over assumptions are labelled
`exact arithmetic under assumptions`; numbers that are illustrative defaults with
no measurement behind them are labelled `assumed`; numbers taken from an
identifiable external reference are labelled `documented`; and numbers that cannot
be known without production hardware are labelled `unknown`.

The repository's own assumptions are encoded in
`src/rag_workbench/scaling/capacity.py`, so the tables below can be regenerated
with `rag-workbench scaling capacity`.

---

## 1. Scope and honesty statement

The prototype:

- is a **single-process FastAPI application**;
- stores metadata and the lexical index in **one SQLite file with one writer**;
- keeps vectors in **Qdrant local mode**, which holds an exclusive lock on its
  storage directory;
- ingests **synchronously**, in the request (or the CLI process), with no queue.

It does **not** serve 10 million documents, and nothing here says it does. The
goal is to show the engineering reasoning required to get there: where the current
design breaks, how much storage and compute the target needs, and which
interface-preserving substitutions make the migration tractable.

---

## 2. What was measured, and how

`rag-workbench scaling run` builds a deterministic synthetic corpus, ingests each
prefix into a **fresh, isolated SQLite database**, and measures ingestion, on-disk
footprint and BM25 latency at several corpus sizes. The corpus generator
(`src/rag_workbench/scaling/corpus.py`) is seeded, so the same configuration
regenerates byte-identical documents.

- **Environment (recorded with the artifact):** Windows 11 `10.0.26300`, Python
  `3.13.5`, 16 logical cores, AMD64. `measured here`
- **Corpus:** synthetic, mixed short/typical/large documents; seed `0`.
  `measured here`
- **Retrieval:** BM25 (SQLite FTS5), `query_mode="or"`, 20 generated queries per
  workload/config, candidate depths 10/50/100. `measured here`
- **Not measured in these artifacts:** dense retrieval, hybrid retrieval,
  cross-encoder reranking, and answer generation. Those need the optional `dense`
  extra and are intentionally out of scope for the recorded baseline. `not
  measured`
- **Code revision:** each artifact records the commit and whether the working tree
  was clean (`provenance`). A record produced from a dirty tree is flagged and is
  not an exact source revision.

Recorded artifacts live in [`benchmarks/recorded_scaling/`](../benchmarks/recorded_scaling/)
and are loaded, validated and displayed by the read-only demo.

> The synthetic corpus is designed for *cost* measurement, not *relevance*: it has
> controllable size and vocabulary, but its absolute retrieval quality is not
> meaningful. Every claim below about latency and storage is a cost claim.

---

## 3. Why the prototype cannot scale linearly

Three structural limits, each already visible at 5,000 documents:

1. **One SQLite writer.** Writes serialise through a single process. There is no
   distributed transaction, no concurrent-ingest path, and no queue to absorb
   bursts.
2. **One FTS5 index in one file.** Lexical search cost and index size live on the
   machine that owns the file. BM25 latency already rises from ~2.6 ms at 687
   chunks to ~98 ms at 38,709 chunks (`measured here`) — see §5.
3. **One exclusive vector store.** Qdrant local mode cannot be shared across
   processes or hosts, so dense/hybrid retrieval cannot scale out at all without
   replacing it.

Nothing in the prototype is wrong for its purpose — it is a *laboratory*. The point
is that the purpose changes at 10M documents.

---

## 4. Measured baselines

### 4.1 Paragraph chunking (`paragraph`, 220/40 tokens, seed 0)

| Documents | Chunks | Ingest (docs/s) | DB size | FTS5 index | Source text |
|---:|---:|---:|---:|---:|---:|
| 100 | 687 | 14.3 | 1.8 MB | 177 KB | 743 KB |
| 1,000 | 7,774 | 30.5 | 18.8 MB | 2.0 MB | 8.3 MB |
| 5,000 | 38,709 | 40.5 | 87.2 MB | 9.9 MB | 41.5 MB |

The smallest workload runs cold, so its 14.3 docs/s is a warm-up artefact rather
than the steady-state rate; the 1,000- and 5,000-document workloads (30.5 and 40.5
docs/s) are more representative.

BM25 latency (ms, 20 queries per cell), `measured here`:

| Documents | @10 p50 / p95 | @50 p50 / p95 | @100 p50 / p95 |
|---:|---:|---:|---:|
| 100 | 2.6 / 3.2 | 4.0 / 4.3 | 5.4 / 5.9 |
| 1,000 | 20.6 / 21.6 | 22.9 / 24.6 | 25.0 / 26.4 |
| 5,000 | 98.0 / 101.9 | 101.2 / 114.2 | 102.9 / 106.0 |

### 4.2 Fixed-token chunking (`fixed_token`, 512/64 tokens, seed 0)

| Documents | Chunks | Ingest (docs/s) | DB size | FTS5 index |
|---:|---:|---:|---:|---:|
| 1,000 | 18,154 | 35.0 | 21.2 MB | 2.4 MB |
| 5,000 | 90,470 | 33.8 | 97.3 MB | 11.8 MB |

The larger token budget produced **more** chunks, not fewer, because fixed-token
windows never cross a parsed block boundary: on paragraph-structured text the
window degenerates to one chunk per paragraph. This is a real property of the
chunker, and it is the first case study in
[`case_studies.md`](case_studies.md): **chunking strategy, not the token budget,
dominates chunk count.**

### 4.3 What these numbers tell us

- Indexing throughput is ~30–40 tiny documents per second at steady state
  (`measured here`); the smallest, cold-start workload reports a lower 14.3 docs/s.
- The SQLite file grows to ~2.36 KB per chunk (relational + FTS5) at 5,000
  documents; the FTS5 index alone is ~267 bytes per chunk. `measured here`
- BM25 p50 grows roughly with corpus size for `or` queries that match many
  documents — from ~2.6 ms to ~98 ms across a ~56× increase in chunks.
  `measured here`

### 4.4 Memory measurements (three distinct labels)

The harness records three different things that must not be conflated, and the
read-only demo shows them as separate columns:

- `rss_mb` — **one instantaneous** process-RSS reading at the end of a workload;
- `sampled_peak_rss_mb` — the **maximum of periodic** process-RSS samples taken
  while the workload ran (a *sampled* peak, not an OS high-water mark);
- `tracemalloc_peak_mb` — the peak of **Python-managed allocations**.

The first two require the optional `psutil` extra (`uv sync --extra profiling`).
Without it, process memory is reported as **not measured** — the harness does not
substitute a Python-heap number for RSS, and `tracemalloc_peak_mb` is **never**
extrapolated to total process memory.

---

## 5. Extrapolating the single-node prototype

These are **illustrative linear extrapolations of a measured single-node trend**,
not predictions about a real deployment. They exist to show why the prototype must
be re-architected rather than resized. The largest configuration actually executed
is **5,000 documents / 90,470 chunks** (`measured here`); every figure below that
is a model or an extrapolation, never a run.

Assume the "typical" shape of **10 million documents × 10 chunks = 100M chunks**.

| Quantity | Basis | Value |
|---|---|---:|
| Chunk count | exact arithmetic under assumptions | 100,000,000 |
| SQLite file at 2.36 KB/chunk | linear extrapolation of `measured here` | ~236 GB |
| FTS5 index at 267 B/chunk | linear extrapolation of `measured here` | ~27 GB |
| Naive BM25 p50 at 38,709-chunk p50 × scale | linear extrapolation of `measured here` | ~253 s per query |
| Ingest time at ~40 docs/s, single writer | linear extrapolation of `measured here` | ~69 h |

The last two rows are the argument. A per-query latency in the hundreds of seconds
and a multi-day single-writer bulk load are unacceptable, and adding cores does not
help: the work is inside one process and one file. **Resizing a single node is not
the path to 10M documents.**

---

## 6. Capacity model (storage and ingestion)

`src/rag_workbench/scaling/capacity.py` is a deterministic model, deliberately
separate from any simulator. It labels every line with its basis so a hard-coded
multiplier can never masquerade as a measurement.

Run it directly:

```powershell
rag-workbench scaling capacity --documents 10000000 --scenario typical
rag-workbench scaling capacity --documents 10000000 --scenario typical --documents-per-day 100000
```

### 6.1 Storage, `typical` scenario (10M docs, 10 chunks/doc, 384-d vectors)

Basis key: **E** = exact arithmetic under assumptions, **A** = illustrative
assumption (no measurement behind it), **D** = identifiable external reference.

| Component | Basis | Size |
|---|---|--:|
| Original source documents | E | 229 GB |
| Chunk text | E | 114 GB |
| Chunk metadata | E | 60 GB |
| Raw embedding vectors (384-d fp32) | E | 143 GB |
| Vector index (HNSW/graph) overhead (+50%) | A | 72 GB |
| Payload index overhead (+20%) | A | 12 GB |
| Lexical index (35% of source text) | A | 80 GB |
| **Primary subtotal** | | **~710 GB** |
| Replicas (factor 2) | E | 710 GB |
| Backups (1 copy) | E | 710 GB |
| Reindex working set (1×) | E | 710 GB |
| Operational headroom (+25%) | E | 710 GB |
| **Total** | | **~3.5 TB** |

**This ~3.5 TB total is an assumption-based planning envelope, not a measurement.**
It is the model's arithmetic over illustrative defaults (the three `A` rows plus
the replica, backup, reindex and headroom factors); it is not a size that was ever
produced or observed. Replace the assumed overhead multipliers with your engine's
documented ratios and your own corpus statistics before treating it as a budget.

**Calibration against measurement.** The measured prototype FTS5 index was ~267
B/chunk, i.e. ~27 GB for 100M chunks, against the model's assumed 80 GB lexical
estimate — the model is conservative for this corpus shape. The model is a
*planning envelope*, not a measurement; always override it with your own corpus
statistics (average chunks/document and source size) before trusting the total.

`small` and `large` document scenarios change the chunk count and source size and
are reported alongside the estimate.

### 6.2 Ingestion, continuous stream

The ingestion model applies non-linear worker scaling (adding workers does not
scale linearly) and takes embedding/indexing throughput as **inputs**. Those
inputs default to illustrative values with basis `assumed`; they are only
`measured` when you supply a figure from your own run. Example: 100,000
documents/day at 10 chunks/document = 1M chunks/day.

```powershell
rag-workbench scaling capacity --documents 10000000 ^
  --documents-per-day 100000 ^
  --embedding-chunks-per-second-per-worker 20 --workers 4
```

The model reports the bottleneck stage (embedding vs indexing), utilization and any
projected backlog, and labels the throughput basis. **Replace
`--embedding-chunks-per-second-per-worker` with a number you measured on your own
hardware** and set the basis to `measured`; the model does not invent GPU or
multi-core throughput. Because the default rates are assumptions, `keeps_up=true`
is a **conditional projection under those assumptions**, not a demonstrated
production capability. Initial bulk load and continuous incremental ingestion must
be planned as **two workloads sharing one capacity**, not one.

---

## 7. Target architecture

The proposed architecture is encoded as data in
`src/rag_workbench/scaling/architecture.py` and rendered by the **Architecture
Explorer** in the Streamlit Scalability Laboratory, so the UI and this document
cannot drift.

```text
   upload -> [ API / auth ] -> [ object storage: originals ] -> [ queue ]
                                         |
                              [ embedding workers ]  (independently scaled)
                                         |
        +----------------+---------------+----------------+
        |                |                                |
   [ PostgreSQL ]   [ lexical engine ]              [ Qdrant server/cluster ]
   metadata/qrels   (OpenSearch/ES)                 vectors + payload filters
        |                |                                |
        +----------------+---------------+----------------+
                                         |
                              [ retrieval orchestrator ]
                              BM25Retriever | VectorStore | RRF
                                         |
                              [ bounded rerank workers ]
                                         |
                              [ answer service (optional) ]
```

### 7.1 Component substitutions

| Area | Current | Proposed | Interface preserved | Unknown until benchmarked |
|---|---|---|---|---|
| Original documents | Local files / `byte_size` in SQLite | S3-compatible object storage | `IngestionService.ingest_bytes` | object-store cost and egress |
| Metadata | SQLite (single writer) | PostgreSQL + read replicas | SQLAlchemy engine; error mapping | concurrent-ingest transaction throughput |
| Lexical | SQLite FTS5 | OpenSearch / Elasticsearch | `Bm25Retriever` (query → hits) | ranking parity and analyzers |
| Vectors | Qdrant local mode | Qdrant server/cluster | `VectorStore` protocol | HNSW recall/latency at target scale |
| Ingestion | synchronous, one writer | durable queue + idempotent workers | `IngestionService.ingest_bytes` | queue depth, retry/backoff tuning |
| Retrieval | in-process service | stateless service, per-stage timeouts | `SearchService.search` | tail latency of parallel fan-out |
| Reranking | in-process CPU | bounded worker pool (CPU/GPU) | `Reranker` protocol | GPU throughput for the chosen model |
| Observability | logs + response timings | metrics, tracing, alerting | timing fields become labels | metric cardinality and retention |

The interface column is the point: because retrieval is already expressed as
Protocols (`Bm25Retriever`, `VectorStore`, `Reranker`), most substitutions are
adapter-level changes, not rewrites of the domain logic.

### 7.2 Migration phases

1. **Move reads and metadata off SQLite.** PostgreSQL for documents, chunks,
   qrels and profiles. Keep the adapter boundary so the SQLAlchemy models change,
   not the services.
2. **Introduce a durable ingestion queue.** Parse/chunk/embed become jobs with
   idempotent keys. Freshness becomes a metric, not a request-latency property.
3. **Move the lexical index** to a sharded engine, keeping the `Bm25Retriever`
   contract. Accept that absolute BM25 scores are engine-specific.
4. **Run Qdrant as a server** with replication and payload indexes; retire local
   mode.
5. **Split retrieval stages** into independently scaled services with timeouts and
   partial-result reporting.
6. **Scale reranking and (optionally) generation** as bounded pools.

### 7.3 Lifecycle: deletes, model migrations, index swaps and rollout

These are the parts of scaling that are not about throughput but about not losing
or leaking data. The prototype has simple versions of some of them; the rest are
new work.

- **Access control before retrieval.** In a multi-tenant system, tenant/source
  filters must be applied *inside* the retriever, not after fusion — otherwise a
  passage from another tenant can enter the candidate list before filtering. The
  prototype is single-user and has no such boundary.
- **Delete propagation.** A delete must remove the relational rows, the lexical
  document and the vector points, and must be idempotent across retries and index
  swaps. The prototype already removes the FTS entries alongside the rows
  (`measured here`); the distributed version needs the same guarantee per store,
  plus reconciliation for partial failures.
- **Embedding model version migrations.** Changing the embedding model invalidates
  every vector. The index profile already carries an embedding identity and
  dimension, so a migration means building a **new index beside the old**, switching
  reads atomically to a new `profile_id`, then retiring the old index. Vectors from
  two models must never share one index.
- **Index swaps and rollback.** Lexical and vector rebuilds are built beside the
  live index and swapped by flipping an alias, with rollback to the previous alias
  on error. This makes a reindex a *capacity* event (see the reindex working set in
  §6.1) rather than an outage.
- **Rollout.** Each adapter substitution (PostgreSQL, a distributed lexical engine,
  a Qdrant server, a queue) can be introduced behind the existing interfaces and
  validated with the Phase-4 evaluation platform before traffic is moved.

---

## 8. Failure modes and operational concerns

- **Index staleness.** With asynchronous ingestion, "ingested" and "searchable"
  diverge. Freshness must be measured and surfaced; a query that misses a
  just-uploaded document is expected, not a bug.
- **Poison documents.** A malformed file must be quarantined after bounded retries,
  not retried forever. The prototype already never invents text for empty parses;
  the queue must preserve that property.
- **Reindex storms.** A full rebuild doubles the footprint (see the reindex working
  set in §6.1) and competes with serving. Rebuilds need their own capacity budget.
- **Score incomparability across engines.** The application already keeps one rank
  convention; after the lexical engine changes, ranks remain comparable but raw
  scores do not. Any threshold tuned to FTS5 BM25 must be re-derived.
- **Tail latency.** Fan-out to lexical + vector + rerank multiplies tail latency.
  Partial results must be reported, never silently hidden.
- **Multi-tenant isolation.** Filters, quotas and per-tenant indexes are new
  concerns the single-user prototype does not model.

### 8.1 Observability and SLOs

The prototype reports per-request `timings_ms` and writes structured logs. A
scaled system needs standing metrics and explicit objectives. **No target below is
asserted** — each is a candidate SLI whose SLO must be set by the product, then
measured on production hardware.

| Signal | Why it matters | Basis |
|---|---|---|
| Ingestion throughput (docs/s, chunks/s) | Sizing the worker pool; the model's bottleneck input | measured here (prototype), unknown (target) |
| Queue depth and age | Backlog vs freshness before it reaches users | not measured (no queue) |
| Index freshness (ingest → searchable lag) | The visible effect of asynchronous ingestion | not measured |
| Retrieval latency p50/p95/p99 **per stage** | Tail latency is additive across fan-out and rerank | measured here at small scale |
| Recall@K / nDCG from the evaluation platform | A faster system that retrieves worse is not an improvement | measured (Phase 4 datasets) |
| Index size per store | Directly drives the §6 capacity total | measured here (FTS5) + assumed ratios |
| Error rate (parse failures, poison docs, timeouts) | Distinguishes load problems from data problems | measured here (0 errors at small scale) |
| Cost per 1k queries / per 1M chunks | Ties engineering choices to the bill | unknown |

The evaluation platform (Phase 4) is the reusable piece here: it already turns a
corpus and a query set into reproducible retrieval metrics, so the same harness can
compare engines *before* a migration is trusted.

### 8.2 Local HTTP load smoke test

`rag-workbench scaling load-test` drives a **locally controlled** server over HTTP
and reports throughput and tail latency for read-only endpoints. It uses only
`httpx`; a warm-up phase is excluded and results are broken down **per endpoint**.
One bounded run on the same machine (8 concurrent users, 5 s warm-up, 30 s
measured, mixed read-only workload, `measured here`):

| Metric | Value |
|---|---:|
| Requests (measured window) | 4,626 |
| Throughput | ~154 req/s |
| Errors | 0 |
| Latency p50 / p95 | 36 ms / 132 ms |
| Busiest endpoint (`/v1/search`) | 2,113 req, p50 35 ms, p95 125 ms |

This is a **smoke-level, single-machine observation of HTTP/API throughput across
endpoints — not RAG generation throughput, not retrieval throughput, and not a
latency SLO.** Generation and dense retrieval are separate, opt-in paths with their
own measurements; they are never folded into this number. The harness refuses to
target the public demo.

---

## 9. What is still unknown

These require **production benchmarking** and are not answered here:

- Real embedding throughput on the target GPU/CPU (the model takes it as input).
- Recall/latency of the chosen vector index parameters at 100M vectors.
- Ranking parity between FTS5 BM25 and a distributed lexical engine on
  legal/technical text.
- Queue depth, retry behaviour and poison-document rates under real traffic.
- Reranker GPU throughput for the chosen model.
- Object-store and cross-AZ egress cost at the target ingest and query volume.

---

## 10. Cost envelope (assumptions, not a quote)

At the `typical` scenario, the model's **3.5 TB** total already includes replicas,
one backup, a reindex working set and 25% headroom. A cost estimate is that number
multiplied by the storage price per GB, plus compute for embedding, lexical and
vector serving, plus egress. Because the architecture is not built, this document
gives **no dollar figure**: any number would be an assumption stack, not a
measurement. Use the capacity model with your own prices and your own measured
throughput.

---

## 11. Reproducing this document

```powershell
# Recorded baselines (regenerates benchmarks/recorded_scaling/*.json)
rag-workbench scaling run --documents 100,1000,5000 --depths 10,50,100 --record

# Larger chunking comparison
rag-workbench scaling run --documents 1000,5000 --strategy fixed_token ^
  --max-tokens 512 --overlap-tokens 64 --record

# Capacity tables for any scenario
rag-workbench scaling capacity --documents 10000000 --scenario typical
```

Every recorded artifact stores its own environment, corpus fingerprint, code
revision with its working-tree state (`provenance`), and limitations, so a number
displayed in the demo can always be traced back to the run that produced it — and a
run made from uncommitted code is flagged rather than presented as a clean
revision.
