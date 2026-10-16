"""Current-versus-proposed architecture description for the 10M scale design.

This module is *data plus small helpers*, not a diagram engine. It encodes the
migration story the scalability document argues for: which component is replaced,
which existing interface would remain, why the replacement is needed, the
engineering trade-off, and what is still an unknown requiring production
benchmarking. The Streamlit Architecture Explorer and ``docs/scaling_to_10m.md``
are both generated from the same source so the UI and the document cannot drift.

Nothing here is installed or executed; the large-scale components are proposals.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

CURRENT_DIAGRAM = """\
            +-----------------------+
   files -> |  IngestionService      |  parse -> chunk -> SQLite + FTS5
            |  (synchronous, 1 app)  |              \\-> Qdrant local (optional)
            +-----------+-----------+
                        |
            +-----------v-----------+
            |  SearchService         |  BM25 (FTS5) | dense (Qdrant) | RRF hybrid
            |  (in-process)          |  + optional cross-encoder reranking
            +-----------+-----------+
                        |
            +-----------v-----------+
            |  FastAPI  ->  Streamlit |  thin HTTP frontend; recorded artifacts
            +-----------------------+
"""

PROPOSED_DIAGRAM = """\
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
"""


class ArchitectureRow(BaseModel):
    """One migration row: current technology -> proposed replacement."""

    model_config = ConfigDict(frozen=True)

    area: str
    current: str
    proposed: str
    interface: str
    reason: str
    tradeoff: str
    unknowns: str


ARCHITECTURE: tuple[ArchitectureRow, ...] = (
    ArchitectureRow(
        area="Original document storage",
        current="Local filesystem / row byte_size in SQLite",
        proposed="Object storage (S3-compatible) with immutable object keys",
        interface="IngestionService.ingest_bytes(data, source_name, ...)",
        reason="Documents must outlive a single host and be re-readable for "
        "re-parse/re-embed without a database blob.",
        tradeoff="Extra network hop and eventual-consistency handling; checksums "
        "remain the integrity check.",
        unknowns="Object-store cost and egress at expected ingest/query volume.",
    ),
    ArchitectureRow(
        area="Authoritative metadata",
        current="SQLite (single writer)",
        proposed="PostgreSQL (managed), logical replication for read replicas",
        interface="SQLAlchemy session/engine; DomainError mapping unchanged",
        reason="SQLite serialises writes to one process; a shared durable store "
        "is required for concurrent ingestion and multiple API replicas.",
        tradeoff="Operational overhead of a server database; migrations must be "
        "run out-of-band.",
        unknowns="Transaction throughput at peak concurrent ingestion.",
    ),
    ArchitectureRow(
        area="Lexical retrieval",
        current="SQLite FTS5 (BM25)",
        proposed="Scalable lexical engine (OpenSearch/Elasticsearch)",
        interface="Bm25Retriever protocol (query -> ranked hits)",
        reason="FTS5 index and query cost grow with a single local file and one "
        "writer; a distributed engine shards and replicates the index.",
        tradeoff="Ranking-score scale differs from FTS5 BM25; absolute scores are "
        "not comparable across engines.",
        unknowns="Ranking parity and analyzer configuration for legal/technical text.",
    ),
    ArchitectureRow(
        area="Vector storage",
        current="Qdrant local mode (embedded, exclusive lock)",
        proposed="Qdrant server or cluster with replication and payload indexes",
        interface="VectorStore protocol (upsert/search/filter/delete)",
        reason="Local mode holds an exclusive lock and cannot be shared across "
        "processes or hosts.",
        tradeoff="Network round-trips and cluster operations replace an embedded "
        "call.",
        unknowns="Recall/latency of the chosen HNSW parameters at target scale.",
    ),
    ArchitectureRow(
        area="Ingestion pipeline",
        current="Synchronous in-request ingestion; one local writer",
        proposed="Durable queue + idempotent jobs + independently scaled workers",
        interface="IngestionService.ingest_bytes (job becomes asynchronous)",
        reason="Parsing/embedding far exceed a request budget; work must survive "
        "restarts and retry without duplicating chunks.",
        tradeoff="Async status tracking and reconciliation replace a synchronous "
        "result.",
        unknowns="Queue depth and retry/backoff tuning under partial failures.",
    ),
    ArchitectureRow(
        area="Retrieval orchestration",
        current="In-process SearchService",
        proposed="Stateless retrieval service; per-stage timeouts and fan-out",
        interface="SearchService.search(...) signature preserved",
        reason="Dense/lexical calls become network hops that need timeouts, "
        "circuit-breaking and independent scaling.",
        tradeoff="More failure modes (partial results) must be surfaced, not "
        "hidden.",
        unknowns="Tail latency of parallel fan-out at required concurrency.",
    ),
    ArchitectureRow(
        area="Reranking",
        current="In-process cross-encoder, CPU",
        proposed="Bounded reranking worker pool (CPU or GPU) behind the same "
        "Reranker interface",
        interface="Reranker protocol (score pairs -> scores)",
        reason="Cross-encoder inference is the most expensive stage and must be "
        "scaled/bounded separately from retrieval.",
        tradeoff="Added latency and a new scaling dimension; may be skipped when "
        "the pool is saturated.",
        unknowns="GPU throughput for the chosen model (must be benchmarked).",
    ),
    ArchitectureRow(
        area="Observability",
        current="Structured logs + in-response timings",
        proposed="Metrics, tracing and alerting per stage and per index",
        interface="Existing timing fields become metric labels",
        reason="At scale, silent index staleness and per-tenant latency skew are "
        "invisible without stage-level telemetry.",
        tradeoff="Telemetry volume and cost must themselves be managed.",
        unknowns="Cardinality and retention policy for per-tenant metrics.",
    ),
)


def architecture_payload() -> dict[str, object]:
    """Serialise the architecture comparison for the API and Streamlit."""

    return {
        "current_diagram": CURRENT_DIAGRAM.rstrip("\n"),
        "proposed_diagram": PROPOSED_DIAGRAM.rstrip("\n"),
        "rows": [row.model_dump() for row in ARCHITECTURE],
    }


__all__ = [
    "ARCHITECTURE",
    "CURRENT_DIAGRAM",
    "PROPOSED_DIAGRAM",
    "ArchitectureRow",
    "architecture_payload",
]
