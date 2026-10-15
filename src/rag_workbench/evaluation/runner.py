"""The evaluation runner.

The runner measures a *retrieval configuration* (a chunking profile plus a
retrieval mode plus optional reranking) over a dataset's queries and judgments.
It deliberately calls :class:`~rag_workbench.retrieval.service.SearchService` — the
same service the API, CLI and Streamlit app use — so a benchmark cannot drift
from live behaviour and no second, simplified retriever exists.

Two relevance-mapping policies are supported, recorded on every experiment:

* ``document`` (BEIR SciFact): a retrieved chunk is mapped to its benchmark
  document ID; repeated chunks from one document are collapsed to the rank of the
  first occurrence, so the evaluation is over a *document* ranking judged at the
  document level.
* ``evidence`` (the legal set): a retrieved chunk is mapped to the clause/evidence
  units it actually contains, so a chunk is never assumed relevant merely because
  it came from a relevant contract.

Candidate depth is tracked separately from the final top-K: ranking metrics are
computed on the de-duplicated ranking truncated to K, while candidate recall is
computed over the whole pooled ranking at depth N.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, model_validator
from sqlalchemy import select

from rag_workbench.config import Settings
from rag_workbench.domain.errors import EvaluationError, ProfileNotFoundError
from rag_workbench.evaluation import diagnostics as diag
from rag_workbench.evaluation.datasets import (
    DOCUMENT_MAPPING,
    EVIDENCE_MAPPING,
    EvaluationDataset,
    EvidenceUnit,
    slugify,
)
from rag_workbench.evaluation.metrics import (
    DEFAULT_K_VALUES,
    aggregate,
    candidate_recall_at_n,
    query_metrics,
    relevant_keys,
)
from rag_workbench.evaluation.runtime import git_commit, latency_stats, runtime_environment
from rag_workbench.persistence.database import Database
from rag_workbench.persistence.models import Document, IndexProfile
from rag_workbench.retrieval.service import SearchService

TIMING_KEYS = ("lexical", "embed_query", "vector_search", "resolve", "fusion")


def _normalize(text: str) -> str:
    return " ".join(text.lower().split())


class EvalRunConfig(BaseModel):
    """A fully explicit, reproducible evaluation configuration."""

    model_config = ConfigDict(frozen=True)

    dataset: str
    profile_id: str
    collection: str | None = None
    mode: str = "bm25"
    query_mode: str = "or"
    candidate_depth: int = Field(default=100, ge=1, le=500)
    top_k: int = Field(default=10, ge=1, le=100)
    k_values: tuple[int, ...] = DEFAULT_K_VALUES
    rerank: bool = False
    rerank_candidates: int | None = Field(default=None, ge=1, le=500)
    reranker_model: str | None = None
    candidates: int | None = Field(default=None, ge=1, le=500)
    max_queries: int | None = Field(default=None, ge=1)
    label: str | None = None
    subset: bool = False
    related_baseline: str | None = None

    @model_validator(mode="after")
    def _validate(self) -> EvalRunConfig:
        if self.query_mode not in ("and", "or"):
            raise ValueError("query_mode must be 'and' or 'or'")
        if self.top_k < max(self.k_values):
            raise ValueError("top_k must cover every requested k value")
        if self.candidate_depth < self.top_k:
            raise ValueError("candidate_depth must be >= top_k")
        if (
            self.rerank
            and self.rerank_candidates is not None
            and self.rerank_candidates < self.top_k
        ):
            raise ValueError("rerank_candidates must be >= top_k")
        return self


class TraceItem(BaseModel):
    """One de-duplicated ranked result, with the evidence needed to explain it."""

    rank: int
    key: str
    chunk_id: str
    document_id: str
    external_document_id: str | None = None
    source_name: str
    section_number: str | None = None
    section_title: str | None = None
    score: float
    score_kind: str
    retrieval_rank: int | None = None
    retrieval_score: float | None = None
    rerank_score: float | None = None
    rank_movement: int | None = None
    relevant: bool = False
    matching_units: list[str] = Field(default_factory=list)
    member_count: int = 1
    excerpt: str = ""


class QueryOutcome(BaseModel):
    """Per-query metrics, diagnostics and trace."""

    query_id: str
    query: str
    num_relevant: int
    status: str  # ok | excluded_no_qrels
    candidate_count: int
    unique_documents: int
    sufficient_candidates: bool
    metrics: dict[str, float] | None = None
    candidate_recall: dict[str, float] = Field(default_factory=dict)
    timings_ms: dict[str, float] = Field(default_factory=dict)
    diagnostics: list[diag.Diagnostic] = Field(default_factory=list)
    trace: list[TraceItem] = Field(default_factory=list)


class ExperimentResult(BaseModel):
    """Everything a completed evaluation run produced."""

    sequence: int | None = None
    experiment_id: str | None = None
    created_at: str | None = None
    git_commit: str | None = None
    dataset: dict[str, object]
    config: dict[str, object]
    runtime: dict[str, object]
    aggregate: dict[str, float]
    candidate_metrics: dict[str, float]
    counts: dict[str, int]
    latency: dict[str, object]
    per_query: list[QueryOutcome]
    notes: list[str] = Field(default_factory=list)


class EvaluationRunner:
    """Run one configuration over one dataset through the real search service."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        search_service: SearchService,
        *,
        git_root: Path | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._search = search_service
        self._git_root = git_root

    # ------------------------------------------------------------------ public

    def run(self, dataset: EvaluationDataset, config: EvalRunConfig) -> ExperimentResult:
        mapping = dataset.manifest.relevance_mapping
        if mapping not in (DOCUMENT_MAPPING, EVIDENCE_MAPPING):
            raise EvaluationError(f"unsupported relevance mapping '{mapping}'")
        if config.collection is not None:
            collection = config.collection
        elif dataset.directory is not None:
            collection = dataset.directory.name
        else:
            collection = slugify(dataset.manifest.name)

        with self._db.session() as session:
            external_by_document, indexed_external = self._document_index(session, collection)
            profile = session.get(IndexProfile, config.profile_id)
            if profile is None:
                raise ProfileNotFoundError(f"index profile not found: {config.profile_id}")
            profile_info = self._profile_info(profile)

        evidence_by_contract: dict[str, list[EvidenceUnit]] = {}
        for unit in dataset.evidence.values():
            evidence_by_contract.setdefault(unit.contract, []).append(unit)

        judged_domain = (
            set(dataset.evidence) if mapping == EVIDENCE_MAPPING else set(dataset.document_index())
        )
        if mapping == DOCUMENT_MAPPING:
            indexed_domain = set(indexed_external)
        else:
            indexed_domain = {
                unit.id
                for unit in dataset.evidence.values()
                if unit.contract in indexed_external
            }

        queries = dataset.queries
        notes: list[str] = []
        if config.max_queries is not None and config.max_queries < len(queries):
            queries = queries[: config.max_queries]
            notes.append(
                f"development subset: {len(queries)} of {len(dataset.queries)} queries "
                "(not directly comparable to full-corpus results)"
            )

        rerank_pool = config.rerank_candidates if config.rerank else None
        outcomes: list[QueryOutcome] = []
        for query in queries:
            outcomes.append(
                self._evaluate_query(
                    query_id=query.id,
                    query_text=query.text,
                    relevance=dataset.relevance_for(query.id),
                    mapping=mapping,
                    external_by_document=external_by_document,
                    evidence_by_contract=evidence_by_contract,
                    indexed_domain=indexed_domain,
                    judged_domain=judged_domain,
                    collection=collection,
                    config=config,
                    rerank_pool=rerank_pool,
                )
            )

        evaluated = [outcome.metrics for outcome in outcomes if outcome.metrics is not None]
        aggregate_metrics = aggregate(evaluated, k_values=config.k_values)
        candidate_metrics = self._candidate_aggregate(outcomes, config.k_values)

        excluded = sum(1 for outcome in outcomes if outcome.status == "excluded_no_qrels")
        insufficient = sum(1 for outcome in outcomes if not outcome.sufficient_candidates)
        if excluded:
            notes.append(f"{excluded} query(ies) had no relevance judgments and were excluded")

        latency = self._latency(outcomes, config)

        return ExperimentResult(
            created_at=None,
            git_commit=git_commit(self._git_root),
            dataset=dataset.manifest.model_dump(mode="json"),
            config={
                **config.model_dump(mode="json"),
                "collection": collection,
                "relevance_mapping": mapping,
                "profile": profile_info,
            },
            runtime=runtime_environment(),
            aggregate=aggregate_metrics,
            candidate_metrics=candidate_metrics,
            counts={
                "queries_total": len(dataset.queries),
                "queries_run": len(outcomes),
                "queries_evaluated": len(evaluated),
                "queries_excluded_no_qrels": excluded,
                "queries_insufficient_candidates": insufficient,
                "documents_in_corpus": dataset.manifest.corpus_count,
                "documents_indexed": len(indexed_external),
            },
            latency=latency,
            per_query=outcomes,
            notes=notes,
        )

    # ----------------------------------------------------------------- queries

    def _evaluate_query(
        self,
        *,
        query_id: str,
        query_text: str,
        relevance: dict[str, int],
        mapping: str,
        external_by_document: dict[str, str],
        evidence_by_contract: dict[str, list[EvidenceUnit]],
        indexed_domain: set[str],
        judged_domain: set[str],
        collection: str,
        config: EvalRunConfig,
        rerank_pool: int | None,
    ) -> QueryOutcome:
        response = self._search.search(
            query=query_text,
            profile_id=config.profile_id,
            mode=config.mode,
            top_k=config.candidate_depth,
            collection=collection,
            candidates=config.candidates or config.candidate_depth,
            rerank=config.rerank,
            reranker_model=config.reranker_model,
            rerank_candidates=rerank_pool,
            query_mode=config.query_mode,
        )

        final_hits = list(response.hits)
        if config.rerank:
            baseline_hits = sorted(
                final_hits, key=lambda hit: (hit.retrieval_rank or hit.rank, hit.chunk_id)
            )
        else:
            baseline_hits = final_hits

        dropped = 0
        final_trace, final_keys = self._map_ranking(
            final_hits, mapping, external_by_document, evidence_by_contract, relevance
        )
        dropped = len(final_hits) - sum(item.member_count for item in final_trace)
        _, baseline_keys = self._map_ranking(
            baseline_hits, mapping, external_by_document, evidence_by_contract, relevance
        )

        relevant = relevant_keys(relevance)
        sufficient = len(final_keys) >= config.top_k
        timing = self._timings(response)

        if not relevant:
            return QueryOutcome(
                query_id=query_id,
                query=query_text,
                num_relevant=0,
                status="excluded_no_qrels",
                candidate_count=len(final_hits),
                unique_documents=len(final_keys),
                sufficient_candidates=sufficient,
                metrics=None,
                timings_ms=timing,
                diagnostics=[
                    diag.Diagnostic(
                        code=diag.UNRESOLVED,
                        detail="no relevance judgments for this query; excluded from aggregates",
                    )
                ],
                trace=final_trace,
            )

        metrics = query_metrics(final_keys, relevance, k_values=config.k_values)
        candidate_recall = {
            f"candidate_recall@{n}": candidate_recall_at_n(final_keys, relevant, n)
            for n in (10, 20, 50, 100)
        }
        diagnostics = diag.diagnose_query(
            relevant=relevant,
            candidate_keys=final_keys,
            final_keys=final_keys,
            baseline_keys=baseline_keys,
            indexed_domain=indexed_domain,
            judged_domain=judged_domain,
            top_k=config.top_k,
            reranked=config.rerank,
            dropped_hits=dropped,
        )
        if not sufficient:
            diagnostics.append(
                diag.Diagnostic(
                    code=diag.BELOW_TOP_K_CUTOFF,
                    detail=(
                        f"only {len(final_keys)} unique units retrieved from a "
                        f"{config.candidate_depth}-candidate pool; top-{config.top_k} "
                        "may be incomplete"
                    ),
                )
            )
        return QueryOutcome(
            query_id=query_id,
            query=query_text,
            num_relevant=len(relevant),
            status="ok",
            candidate_count=len(final_hits),
            unique_documents=len(final_keys),
            sufficient_candidates=sufficient,
            metrics=metrics,
            candidate_recall=candidate_recall,
            timings_ms=timing,
            diagnostics=diagnostics,
            trace=final_trace,
        )

    def _map_ranking(
        self,
        hits: Sequence[object],
        mapping: str,
        external_by_document: dict[str, str],
        evidence_by_contract: dict[str, list[EvidenceUnit]],
        relevance: dict[str, int],
    ) -> tuple[list[TraceItem], list[str]]:
        """Map hits to evaluated keys, collapse repeats, and build the trace."""

        trace: list[TraceItem] = []
        keys: list[str] = []
        position_by_key: dict[str, int] = {}

        for hit in hits:  # type: ignore[assignment]
            external = external_by_document.get(hit.document_id)
            matched = self._keys_for_hit(hit, mapping, external, evidence_by_contract)
            for key in matched:
                if key in position_by_key:
                    trace[position_by_key[key]].member_count += 1
                    continue
                position_by_key[key] = len(trace)
                keys.append(key)
                trace.append(
                    TraceItem(
                        rank=len(trace) + 1,
                        key=key,
                        chunk_id=hit.chunk_id,
                        document_id=hit.document_id,
                        external_document_id=external,
                        source_name=hit.source_name,
                        section_number=hit.section_number,
                        section_title=hit.section_title,
                        score=hit.score,
                        score_kind=hit.score_kind,
                        retrieval_rank=hit.retrieval_rank,
                        retrieval_score=hit.retrieval_score,
                        rerank_score=hit.rerank_score,
                        rank_movement=hit.rank_movement,
                        relevant=relevance.get(key, 0) > 0,
                        matching_units=list(matched),
                        excerpt=" ".join(hit.text.split())[:240],
                    )
                )
        return trace, keys

    @staticmethod
    def _keys_for_hit(
        hit: object,
        mapping: str,
        external: str | None,
        evidence_by_contract: dict[str, list[EvidenceUnit]],
    ) -> list[str]:
        if external is None:
            return []
        if mapping == DOCUMENT_MAPPING:
            return [external]
        units = evidence_by_contract.get(external, [])
        if not units:
            return []
        text = _normalize(hit.text)  # type: ignore[attr-defined]
        matched: list[str] = []
        for unit in units:
            if unit.section and hit.section_number == unit.section or _normalize(unit.text) in text:  # type: ignore[attr-defined]
                matched.append(unit.id)
        return matched

    # ----------------------------------------------------------------- helpers

    @staticmethod
    def _document_index(
        session, collection: str  # noqa: ANN001
    ) -> tuple[dict[str, str], list[str]]:
        rows = session.execute(
            select(Document.id, Document.external_id, Document.source_name).where(
                Document.collection == collection
            )
        ).all()
        by_document: dict[str, str] = {}
        externals: list[str] = []
        for document_id, external_id, source_name in rows:
            resolved = external_id or Path(source_name).stem
            by_document[document_id] = resolved
            externals.append(resolved)
        return by_document, externals

    @staticmethod
    def _profile_info(profile: IndexProfile) -> dict[str, object]:
        return {
            "profile_id": profile.id,
            "name": profile.name,
            "config_hash": profile.config_hash,
            "chunker_strategy": profile.chunker_strategy,
            "chunker_params": dict(profile.chunker_params or {}),
            "embedding_model": profile.embedding_model,
            "embedding_dimension": profile.embedding_dimension,
            "embedding_params": dict(profile.embedding_params or {}),
            "index_version": profile.index_version,
        }

    @staticmethod
    def _timings(response) -> dict[str, float]:  # noqa: ANN001
        timings: dict[str, float] = {}
        retrieval = 0.0
        for key in TIMING_KEYS:
            value = response.timings_ms.get(key)
            if value is not None:
                retrieval += value
        for key, value in response.timings_ms.items():
            if key.startswith("dense_"):
                retrieval += value
        timings["retrieval"] = round(retrieval, 3)
        if "rerank_inference" in response.timings_ms:
            timings["rerank"] = response.timings_ms["rerank_inference"]
        if "rerank_init" in response.timings_ms:
            timings["rerank_init"] = response.timings_ms["rerank_init"]
        if response.total_ms is not None:
            timings["total"] = response.total_ms
        return timings

    @staticmethod
    def _candidate_aggregate(
        outcomes: Sequence[QueryOutcome], k_values: Sequence[int]
    ) -> dict[str, float]:
        del k_values  # candidate recall has its own fixed depths
        values: dict[str, list[float]] = {}
        for outcome in outcomes:
            if outcome.metrics is None:
                continue
            for name, value in outcome.candidate_recall.items():
                values.setdefault(name, []).append(value)
        return {
            name: round(sum(samples) / len(samples), 6)
            for name, samples in values.items()
            if samples
        }

    def _latency(
        self, outcomes: Sequence[QueryOutcome], config: EvalRunConfig
    ) -> dict[str, object]:
        retrieval = [outcome.timings_ms.get("retrieval", 0.0) for outcome in outcomes]
        total = [outcome.timings_ms.get("total", 0.0) for outcome in outcomes]
        latency: dict[str, object] = {
            "retrieval_ms": latency_stats(retrieval),
            "total_ms": latency_stats(total),
        }
        if config.rerank:
            rerank = [
                outcome.timings_ms["rerank"]
                for outcome in outcomes
                if "rerank" in outcome.timings_ms
            ]
            latency["rerank_ms"] = latency_stats(rerank)
            init = [
                outcome.timings_ms["rerank_init"]
                for outcome in outcomes
                if outcome.timings_ms.get("rerank_init")
            ]
            latency["model_init_ms"] = round(max(init), 3) if init else 0.0
        return latency


__all__ = [
    "EvalRunConfig",
    "EvaluationRunner",
    "ExperimentResult",
    "QueryOutcome",
    "TraceItem",
]
