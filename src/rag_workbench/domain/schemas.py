"""Domain schemas shared by ingestion, retrieval, the API and the CLI.

These are plain Pydantic models. They carry the provenance contract: a chunk
knows which page(s) and section it came from *when that information was
extractable*, and ``None`` means genuinely unknown rather than a guess.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

DocumentStatus = Literal["indexed", "empty", "failed"]
JobState = Literal["queued", "running", "completed", "partial", "failed", "duplicate"]

#: How to read a hit's ``score``. BM25 over FTS5 is ascending (smaller is
#: better); cosine similarity and RRF are descending (larger is better).
ScoreKind = Literal["bm25", "cosine", "rrf"]


class ParsedBlock(BaseModel):
    """A contiguous unit of parsed text (usually a paragraph) with provenance."""

    model_config = ConfigDict(frozen=True)

    text: str
    order: int
    page: int | None = None
    section_path: str | None = None
    char_start: int | None = None
    char_end: int | None = None


class ParsedDocument(BaseModel):
    """Ordered parsed blocks for one source document, plus parsing diagnostics."""

    source_name: str
    mime_type: str
    checksum: str
    parser_version: str
    blocks: list[ParsedBlock] = Field(default_factory=list)
    diagnostics: list[str] = Field(default_factory=list)
    text_length: int = 0

    @property
    def char_count(self) -> int:
        return sum(len(block.text) for block in self.blocks)


class ChunkData(BaseModel):
    """A chunk ready to be persisted, with deterministic ID and provenance."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    order: int
    text: str
    token_count: int
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    # Clause-aware provenance. Populated only when it could be derived
    # deterministically; ``None`` means genuinely unknown, never a guess.
    section_number: str | None = None
    section_title: str | None = None
    parent_clause: str | None = None
    section_level: int | None = None
    diagnostics: list[str] = Field(default_factory=list)


class RerankBatch(BaseModel):
    """Cross-encoder scores for a batch of query-passage pairs.

    Scores are raw cross-encoder relevance logits: **larger is more relevant**,
    but they are not calibrated probabilities and must not be described as such.
    """

    scores: list[float]
    max_length: int | None = None
    truncated_count: int = 0
    diagnostics: list[str] = Field(default_factory=list)


class EmbeddingBatch(BaseModel):
    """Result of embedding a batch of texts, with truncation diagnostics."""

    vectors: list[list[float]]
    dimension: int
    diagnostics: list[str] = Field(default_factory=list)
    truncated_count: int = 0
    max_tokens_seen: int | None = None


class VectorPoint(BaseModel):
    """A vector ready to be upserted, keyed by its stable chunk ID."""

    id: str
    vector: list[float]
    payload: dict[str, object] = Field(default_factory=dict)


class VectorMatch(BaseModel):
    """A raw vector-store search result before provenance resolution."""

    id: str
    score: float
    payload: dict[str, object] = Field(default_factory=dict)


class SearchHit(BaseModel):
    """A single retrieval result with provenance and a normalized rank.

    ``rank`` is 1-based and always means "1 is best", independent of the
    underlying scoring direction. ``score`` is the score of the selected mode and
    ``score_kind`` documents how to read it:

    * ``bm25``  — FTS5 score, *smaller* (more negative) is better.
    * ``cosine`` — semantic similarity, *larger* is better.
    * ``rrf``   — Reciprocal Rank Fusion score, *larger* is better.

    For hybrid results the original component ranks/scores are preserved in the
    ``bm25_*`` and ``dense_*`` fields so rank movements can be inspected.

    When reranking is enabled ``rank`` is the *final* (post-rerank) position and
    the pre-rerank position is captured in ``retrieval_rank``; ``rerank_score``
    holds the cross-encoder score and ``rank_movement`` is
    ``retrieval_rank - rank`` (positive means the passage moved up).
    """

    rank: int
    chunk_id: str
    document_id: str
    source_name: str
    collection: str
    chunk_order: int
    text: str
    score: float
    score_kind: ScoreKind = "bm25"
    section_path: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    bm25_rank: int | None = None
    bm25_score: float | None = None
    dense_rank: int | None = None
    dense_score: float | None = None
    fused_score: float | None = None
    section_number: str | None = None
    section_title: str | None = None
    parent_clause: str | None = None
    # Reranking diagnostics (populated only when reranking ran).
    rerank_score: float | None = None
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    retrieval_score_kind: ScoreKind | None = None
    rank_movement: int | None = None


class IngestionResult(BaseModel):
    """Outcome of ingesting one document into one index profile."""

    job_id: str
    document_id: str
    profile_id: str
    state: JobState
    is_duplicate: bool
    chunk_count: int
    source_name: str
    checksum: str
    collection: str
    diagnostics: list[str] = Field(default_factory=list)
    timings_ms: dict[str, float] = Field(default_factory=dict)


class DocumentDeleteResult(BaseModel):
    document_id: str
    profile_id: str | None
    chunks_deleted: int
    document_deleted: bool


class SearchResponse(BaseModel):
    """A retrieval response with configuration fingerprint and stage timings."""

    query: str
    mode: str
    profile_id: str
    profile_name: str
    config_hash: str
    collection: str | None = None
    top_k: int
    candidates: int | None = None
    query_mode: str = "and"
    rerank: bool = False
    rerank_candidates: int | None = None
    reranker: dict[str, object] | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    total_ms: float | None = None
    hits: list[SearchHit] = Field(default_factory=list)


class ChunkRecord(BaseModel):
    """A stored chunk with full provenance, for the chunking laboratory."""

    chunk_id: str
    document_id: str
    profile_id: str
    source_name: str
    collection: str
    chunk_order: int
    text: str
    token_count: int
    section_path: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    parent_clause: str | None = None
    section_level: int | None = None
    page_start: int | None = None
    page_end: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    diagnostics: list[str] = Field(default_factory=list)


class VectorIndexStatus(BaseModel):
    """Reconciliation view of one profile's vector index versus SQLite."""

    profile_id: str
    collection: str
    embedding_model: str | None = None
    dimension: int | None = None
    chunk_count: int
    point_count: int
    missing_points: int
    orphan_points: int
    in_sync: bool


class VectorRebuildResult(BaseModel):
    """Outcome of rebuilding (or repairing) one profile's vector index."""

    profile_id: str
    collection: str
    chunks: int
    points_written: int
    orphan_points_deleted: int = 0
    duration_ms: float
    diagnostics: list[str] = Field(default_factory=list)


class DocumentRecord(BaseModel):
    """A document row as returned by the API/CLI."""

    document_id: str
    collection: str
    source_name: str
    source_checksum: str
    mime_type: str
    byte_size: int
    external_id: str | None = None
    imported_at: datetime
    status: DocumentStatus
    diagnostics: list[str] = Field(default_factory=list)
    chunk_count: int


class ProfileRecord(BaseModel):
    profile_id: str
    name: str
    config_hash: str
    parser_version: str
    chunker_strategy: str
    chunker_params: dict[str, object]
    embedding_model: str | None = None
    embedding_dimension: int | None = None
    embedding_revision: str | None = None
    similarity_metric: str = "cosine"
    normalize_embeddings: bool = True
    vector_collection: str | None = None
    index_version: str
    created_at: datetime
    chunk_count: int


# ------------------------------------------------------- answer generation (§8)

#: The explicit states a generated answer may take. ``insufficient_evidence`` is
#: a first-class outcome, not a failure: the model is allowed to abstain.
AnswerStatus = Literal[
    "answered",
    "partially_answered",
    "insufficient_evidence",
    "generation_error",
]


class EvidenceItem(BaseModel):
    """One passage selected into the model context, with a request-scoped id.

    ``source_id`` is the ``[S1]``/``[S2]`` identifier the model must cite; it maps
    to exactly one chunk returned by the retrieval service. ``text`` is the
    passage verbatim — never paraphrased, never invented.
    """

    source_id: str
    chunk_id: str
    document_id: str
    source_name: str
    collection: str | None = None
    section_path: str | None = None
    section_number: str | None = None
    section_title: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    rank: int
    score: float
    score_kind: str = "bm25"
    text: str
    token_estimate: int = 0


class ExcludedEvidence(BaseModel):
    """A retrieved candidate that was not selected, and exactly why."""

    chunk_id: str
    source_name: str
    rank: int
    reason: Literal["duplicate", "overlap", "max_evidence", "context_budget"]
    detail: str = ""


class Claim(BaseModel):
    """One factual claim and the source ids the model cited for it."""

    text: str
    citations: list[str] = Field(default_factory=list)


class AnswerPayload(BaseModel):
    """The structured output the generator must produce (the LLM output schema).

    It carries no numerical confidence: retrieval scores and reranker logits are
    not probabilities of factual correctness and are never requested here.
    """

    status: AnswerStatus
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list)
    reason: str | None = None


class GenerationConfig(BaseModel):
    """Generation parameters for one request (distinct from retrieval config)."""

    model: str
    temperature: float = 0.1
    num_predict: int = 512
    timeout_s: float = 120.0
    seed: int | None = None


class GenerationRequest(BaseModel):
    """Everything the generator needs: trusted instructions, question, evidence."""

    instructions: str
    question: str
    evidence: list[EvidenceItem] = Field(default_factory=list)
    config: GenerationConfig
    response_schema: dict[str, object] = Field(default_factory=dict)


class GenerationResult(BaseModel):
    """A generation outcome, including timing, token usage and failure detail."""

    model_id: str
    payload: AnswerPayload | None = None
    raw_text: str = ""
    structured: bool = True
    done_reason: str | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    timings_ms: dict[str, float] = Field(default_factory=dict)
    error: str | None = None
    diagnostics: list[str] = Field(default_factory=list)


class CitationIssue(BaseModel):
    code: str
    detail: str
    claim_index: int | None = None
    source_id: str | None = None


class CitationValidation(BaseModel):
    """Deterministic citation checks. Several properties are kept strictly apart.

    Mechanical properties (verifiable without human judgement):

    * ``structural_valid``  — the output conformed to the answer schema;
    * ``all_citations_resolved`` — every citation string the model returned is a
      well-formed identifier that maps to a selected passage (no malformed string,
      no unknown id). ``None`` when the model returned no citations at all;
    * ``all_claims_cited`` — every structured claim carries at least one citation.
      ``None`` when there are no structured claims;
    * ``citation_valid``    — the combined mechanical verdict used by callers: the
      two properties above hold, and an answer the model marked as evidence-backed
      actually carries claims and citations.

    Not mechanically verifiable (always reported ``not_determined``):

    * ``answer_claims_consistency`` — whether the free-text answer asserts facts
      that the structured ``claims`` do not represent. Establishing this needs
      human review or a separate evaluator, so it is never asserted here;
    * ``semantic_support``  — whether a cited passage *actually supports* a claim.
    """

    structural_valid: bool
    citation_valid: bool
    semantic_support: str = "not_determined"
    all_citations_resolved: bool | None = None
    all_claims_cited: bool | None = None
    answer_claims_consistency: str = "not_determined"
    cited_source_ids: list[str] = Field(default_factory=list)
    known_source_ids: list[str] = Field(default_factory=list)
    unknown_source_ids: list[str] = Field(default_factory=list)
    malformed_citations: list[str] = Field(default_factory=list)
    uncited_claim_indices: list[int] = Field(default_factory=list)
    issues: list[CitationIssue] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ContextDiagnostics(BaseModel):
    """What evidence selection did: retrieved vs selected vs excluded."""

    retrieved: int
    selected: int
    duplicates_removed: int = 0
    overlaps_removed: int = 0
    excluded_by_budget: int = 0
    excluded_by_max: int = 0
    token_estimate: int = 0
    token_budget: int = 0
    token_count_method: str = "estimated"
    notes: list[str] = Field(default_factory=list)


class AnswerResponse(BaseModel):
    """The complete answer record: answer, evidence, validation and diagnostics.

    ``evidence`` is the *selected context* (what the model saw); ``retrieved_hits``
    is the *full retrieval* (what was available, and could have been excluded).
    They are deliberately separate so a reader can tell whether evidence was never
    retrieved versus retrieved-but-excluded.
    """

    question: str
    status: AnswerStatus
    answer: str = ""
    claims: list[Claim] = Field(default_factory=list)
    unanswered: list[str] = Field(default_factory=list)
    reason: str | None = None
    evidence: list[EvidenceItem] = Field(default_factory=list)
    excluded_evidence: list[ExcludedEvidence] = Field(default_factory=list)
    retrieved_hits: list[SearchHit] = Field(default_factory=list)
    citation_validation: CitationValidation
    context: ContextDiagnostics
    retrieval: dict[str, object] = Field(default_factory=dict)
    generation: dict[str, object] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    diagnostics: list[dict[str, str]] = Field(default_factory=list)
    generator_available: bool = False
    recorded: bool = False
