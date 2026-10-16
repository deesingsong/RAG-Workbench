# Designing RAG Workbench for 10 million documents

This document is a **design case study**, not a claim about the running system. The
local prototype was measured at bounded corpus sizes on a single machine (see
[Measured baselines](#4-measured-baselines)); everything beyond that is explicit
extrapolation, published vendor behaviour, or an assumption that must be replaced
by a production benchmark. Numbers that come from measurement are labelled
`measured here`; numbers that come from arithmetic over assumptions are labelled
`exact arithmetic under assumptions`; numbers taken from documentation are
labelled `documented`; and numbers that cannot be known without production
hardware are labelled `unknown`.

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
  workload, candidate depths 10/50/100. `measured here`
- **Not measured in these artifacts:** dense retrieval, hybrid retrieval,
  cross-encoder reranking, and answer generation. Those need the optional `dense`
  extra and are intentionally out of scope for the recorded baseline. `not
  measured`

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
   machine that owns the file. BM25 latency already rises from ~4 ms at 687 chunks
   to ~112 ms at 38,709 chunks (`measured here`) — see §5.
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
| 100 | 687 | 34.7 | 1.8 MB | 177 KB | 743 KB |
| 1,000 | 7,774 | 38.2 | 18.8 MB | 2.0 MB | 8.3 MB |
| 5,000 | 38,709 | 33.1 | 87.3 MB | 9.9 MB | 41.5 MB |

BM25 latency (ms, 20 queries per cell), `measured here`:

| Documents | @10 p50 / p95 | @50 p50 / p95 | @100 p50 / p95 |
|---:|---:|---:|---:|
| 100 | 3.8 / 4.5 | 5.0 / 5.5 | 6.5 / 7.8 |
| 1,000 | 21.1 / 22.8 | 22.9 / 23.6 | 26.3 / 36.6 |
| 5,000 | 112.2 / 122.0 | 102.6 / 116.5 | 105.0 / 111.5 |

### 4.2 Fixed-token chunking (`fixed_token`, 512/64 tokens, seed 0)

| Documents | Chunks | Ingest (docs/s) | DB size | FTS5 index |
|---:|---:|---:|---:|---:|
| 1,000 | 18,154 | 24.5 | 21.2 MB | 2.4 MB |
| 5,000 | 90,470 | 21.6 | 97.3 MB | 11.8 MB |

The larger token budget produced **more** chunks, not fewer, because fixed-token
windows never cross a parsed block boundary: on paragraph-structured text the
window degenerates to one chunk per paragraph. This is a real property of the
chunker, and it is the first case study in
[`case_studies.md`](case_studies.md): **chunking strategy, not the token budget,
dominates chunk count.**

### 4.3 What these numbers tell us

- Indexing throughput is ~22–38 tiny documents per second, single-writer.
  `measured here`
- The SQLite file grows to ~2.36 KB per chunk (relational + FTS5) at 5,000
  documents; the FTS5 index alone is ~267 bytes per chunk. `measured here`
- BM25 p50 grows roughly with corpus size for `or` queries that match many
  documents — from 3.8 ms to 112 ms across a ~56× increase in chunks. `measured here`

---

## 5. Extrapolating the single-node prototype

These are **illustrative linear extrapolations of a measured single-node trend**,
not predictions about a real deployment. They exist to show why the prototype must
be re-architected rather than resized.

Assume the "typical" shape of **10 documents × 10 chunks = 100M chunks**.

| Quantity | Basis | Value |
|---|---|---:|
| Chunk count | exact arithmetic under assumptions | 100,000,000 |
| SQLite file at 2.36 KB/chunk | linear extrapolation of `measured here` | ~236 GB |
| FTS5 index at 267 B/chunk | linear extrapolation of `measured here` | ~27 GB |
| Naive BM25 p50 at 38,709-chunk p50 × scale | linear extrapolation of `measured here` | ~289 s per query |
| Ingest time at ~35 docs/s, single writer | linear extrapolation of `measured here` | ~79 h |

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

Basis key: **E** = exact arithmetic under assumptions, **D** = documented ratio.

| Component | Basis | Size |
|---|---|--:|
| Original source documents | E | 229 GB |
| Chunk text | E | 114 GB |
| Chunk metadata | E | 60 GB |
| Raw embedding vectors (384-d fp32) | E | 143 GB |
| Vector index (HNSW/graph) overhead (+50%) | D | 72 GB |
| Payload index overhead (+20%) | D | 12 GB |
| Lexical index (35% of source text) | D | 80 GB |
| **Primary subtotal** | | **~710 GB** |
| Replicas (factor 2) | E | 710 GB |
| Backups (1 copy) | E | 710 GB |
| Reindex working set (1×) | E | 710 GB |
| Operational headroom (+25%) | E | 710 GB |
| **Total** | | **~3.5 TB** |

**Calibration against measurement.** The measured prototype FTS5 index was ~267
B/chunk, i.e. ~27 GB for 100M chunks, against the model's documented 80 GB
lexical estimate — the model is conservative for this corpus shape. The model is a
*planning envelope*, not a measurement; always override it with your own corpus
statistics (average chunks/document and source size) before trusting the total.

`small` and `large` document scenarios change the chunk count and source size and
are reported alongside the estimate.

### 6.2 Ingestion, continuous stream

The ingestion model treats embedding throughput as a **measured/user-supplied**
input and applies non-linear worker scaling (adding workers does not scale
linearly). Example: 100,000 documents/day at 10 chunks/document = 1M chunks/day.

```powershell
rag-workbench scaling capacity --documents 10000000 ^
  --documents-per-day 100000 ^
  --embedding-chunks-per-second-per-worker 20 --workers 4
```

The model reports the bottleneck stage (embedding vs indexing), utilization and any
projected backlog. **Replace `--embedding-chunks-per-second-per-worker` with a
number you measured on your own hardware**; the model does not invent GPU or
multi-core throughput. Initial bulk load and continuous incremental ingestion must
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

Every recorded artifact stores its own environment, corpus fingerprint and
limitations, so a number displayed in the demo can always be traced back to the
run that produced it.
