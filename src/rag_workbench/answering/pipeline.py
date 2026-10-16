"""The answer pipeline (Phase 5 §7, §9, §13).

:class:`AnswerService` composes the stages — retrieval (reused, never
re-implemented), evidence selection, context assembly, local generation,
structured-output validation, deterministic citation verification and answer
diagnostics — into one inspectable record.

It abstains honestly:

* **No retrieved evidence** -> ``insufficient_evidence`` without calling the model.
* **No passage fits the budget** -> ``insufficient_evidence`` (never a silent
  truncation).
* **Model output that cannot be recovered** -> ``generation_error`` carrying the
  raw text and the failure reason.
* **Unreachable/disabled generator** -> :class:`GeneratorUnavailableError`, so
  search and evaluation stay usable while generation alone reports 503.

Retrieval and generation latency are measured separately, and evidence that was
retrieved but *excluded from the context* is reported distinctly from evidence
that was never retrieved.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable

from rag_workbench.answering.citations import abstained_validation, verify_citations
from rag_workbench.answering.context import ContextAssembler, estimate_tokens
from rag_workbench.answering.ollama import build_answer_generator
from rag_workbench.answering.prompt import ANSWER_JSON_SCHEMA, build_instructions
from rag_workbench.config import Settings
from rag_workbench.domain.interfaces import AnswerGenerator
from rag_workbench.domain.schemas import (
    AnswerResponse,
    ContextDiagnostics,
    ExcludedEvidence,
    GenerationConfig,
    GenerationRequest,
    SearchHit,
)
from rag_workbench.persistence.database import Database
from rag_workbench.retrieval.service import MAX_TOP_K, SearchService

GeneratorFactory = Callable[[], AnswerGenerator]

_ANSWERED_STATES = ("answered", "partially_answered")
_GENERATION_ERROR_REASON = "the local model could not produce a valid structured answer"


def _dropped_suffix(
    excluded: list[ExcludedEvidence], reason: str, *, limit: int = 6
) -> str:
    """Name the passages dropped for one reason, so the loss is inspectable.

    The excluded list already records every dropped candidate; naming the few
    highest-ranked ones inline makes an ``evidence_excluded_by_max`` diagnostic
    actionable instead of a bare count.
    """

    dropped = [item for item in excluded if item.reason == reason]
    if not dropped:
        return ""
    labels = ", ".join(
        f"{item.source_name} (rank {item.rank})" for item in dropped[:limit]
    )
    more = " …" if len(dropped) > limit else ""
    return f": {labels}{more}"


class AnswerService:
    """Run one grounded answer request end to end."""

    def __init__(
        self,
        database: Database,
        settings: Settings,
        search_service: SearchService,
        *,
        generator_factory: GeneratorFactory | None = None,
    ) -> None:
        self._db = database
        self._settings = settings
        self._search = search_service
        self._generator_factory: GeneratorFactory = generator_factory or (
            lambda: build_answer_generator(settings)
        )

    # ------------------------------------------------------------------- public

    def answer(
        self,
        *,
        question: str,
        profile_id: str | None = None,
        mode: str = "bm25",
        collection: str | None = None,
        query_mode: str = "and",
        candidate_depth: int | None = None,
        candidates: int | None = None,
        rerank: bool = False,
        reranker_model: str | None = None,
        rerank_candidates: int | None = None,
        max_evidence: int | None = None,
        context_tokens: int | None = None,
        temperature: float | None = None,
        num_predict: int | None = None,
        seed: int | None = None,
    ) -> AnswerResponse:
        settings = self._settings
        resolved_max_evidence = max_evidence or settings.answer_max_evidence
        resolved_budget = context_tokens or settings.answer_context_tokens
        depth = self._resolve_depth(candidate_depth, resolved_max_evidence)
        pool = (rerank_candidates or depth) if rerank else None

        generator = self._generator_factory()
        started = time.perf_counter()

        retrieval = self._search.search(
            query=question,
            profile_id=profile_id,
            mode=mode,
            top_k=depth,
            collection=collection,
            candidates=candidates,
            rerank=rerank,
            reranker_model=reranker_model,
            rerank_candidates=pool,
            query_mode=query_mode,
        )
        retrieval_ms = retrieval.total_ms or round((time.perf_counter() - started) * 1000, 3)

        # Reserve room for the trusted instructions, the question, the response
        # schema and the expected completion before admitting passages.
        instructions = build_instructions(settings.answer_prompt_variant)
        reserve = (
            estimate_tokens(instructions)
            + estimate_tokens(question)
            + estimate_tokens(json.dumps(ANSWER_JSON_SCHEMA, separators=(",", ":")))
            + settings.ollama_num_predict
        )
        assembler = ContextAssembler(
            max_evidence=resolved_max_evidence, token_budget=resolved_budget
        )
        context_started = time.perf_counter()
        assembly = assembler.assemble(retrieval.hits, reserve_tokens=reserve)
        context_ms = round((time.perf_counter() - context_started) * 1000, 3)
        context = assembly.diagnostics or ContextDiagnostics(
            retrieved=len(retrieval.hits), selected=len(assembly.evidence)
        )

        retrieval_info = self._retrieval_info(retrieval, depth, pool)
        generation_config = GenerationConfig(
            model=generator.model_id,
            temperature=settings.ollama_temperature if temperature is None else temperature,
            num_predict=settings.ollama_num_predict if num_predict is None else num_predict,
            timeout_s=settings.ollama_timeout_s,
            seed=settings.ollama_seed if seed is None else seed,
        )

        if not retrieval.hits or not assembly.evidence:
            reason = (
                "no passages were retrieved from the corpus"
                if not retrieval.hits
                else "no retrieved passage fitted the context budget"
            )
            return self._abstained(
                question=question,
                reason=reason,
                retrieval_hits=retrieval.hits,
                evidence=assembly.evidence,
                excluded=assembly.excluded,
                context=context,
                retrieval_info=retrieval_info,
                generation_config=generation_config,
                retrieval_ms=retrieval_ms,
                context_ms=context_ms,
                started=started,
            )

        generator.ensure_available()  # raises GeneratorUnavailableError -> 503
        request = GenerationRequest(
            instructions=instructions,
            question=question,
            evidence=assembly.evidence,
            config=generation_config,
            response_schema=ANSWER_JSON_SCHEMA,
        )
        generation_started = time.perf_counter()
        result = generator.generate(request)
        generation_ms = result.timings_ms.get(
            "generation", round((time.perf_counter() - generation_started) * 1000, 3)
        )

        validation_started = time.perf_counter()
        validation = verify_citations(
            result.payload, evidence=assembly.evidence, structural_valid=result.payload is not None
        )
        validation_ms = round((time.perf_counter() - validation_started) * 1000, 3)

        if result.payload is None:
            status = "generation_error"
            answer_text = ""
            claims = []
            unanswered: list[str] = []
            reason = result.error or _GENERATION_ERROR_REASON
        else:
            status = result.payload.status
            answer_text = result.payload.answer
            claims = result.payload.claims
            unanswered = result.payload.unanswered
            reason = result.payload.reason

        diagnostics = self._diagnose(
            retrieval_hits=retrieval.hits,
            excluded=assembly.excluded,
            context=context,
            validation=validation,
            status=status,
            cited_source_ids=validation.cited_source_ids,
            evidence_ids=[item.source_id for item in assembly.evidence],
        )
        generation_info = {
            "model_id": result.model_id,
            "structured": result.structured,
            "done_reason": result.done_reason,
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "temperature": generation_config.temperature,
            "num_predict": generation_config.num_predict,
            "timeout_s": generation_config.timeout_s,
            "prompt_variant": settings.answer_prompt_variant,
            "token_count_method": "model-and-estimate",
        }

        total_ms = round((time.perf_counter() - started) * 1000, 3)
        return AnswerResponse(
            question=question,
            status=status,
            answer=answer_text,
            claims=claims,
            unanswered=unanswered,
            reason=reason,
            evidence=assembly.evidence,
            excluded_evidence=assembly.excluded,
            retrieved_hits=retrieval.hits,
            citation_validation=validation,
            context=context,
            retrieval=retrieval_info,
            generation=generation_info,
            timings_ms={
                "retrieval": retrieval_ms,
                "context_assembly": context_ms,
                "generation": generation_ms,
                "citation_validation": validation_ms,
                "total": total_ms,
            },
            diagnostics=diagnostics,
            generator_available=True,
        )

    # ------------------------------------------------------------------ helpers

    def _resolve_depth(self, candidate_depth: int | None, max_evidence: int) -> int:
        if candidate_depth is not None:
            depth = candidate_depth
        else:
            depth = min(max(max_evidence * 4, 20), MAX_TOP_K)
        depth = max(depth, max_evidence)
        return min(depth, MAX_TOP_K)

    @staticmethod
    def _retrieval_info(response, depth: int, pool: int | None) -> dict[str, object]:
        return {
            "mode": response.mode,
            "query_mode": response.query_mode,
            "profile_id": response.profile_id,
            "profile_name": response.profile_name,
            "config_hash": response.config_hash,
            "collection": response.collection,
            "candidate_depth": depth,
            "rerank": response.rerank,
            "rerank_candidates": pool,
            "reranker": response.reranker,
            "timings_ms": dict(response.timings_ms),
        }

    def _abstained(
        self,
        *,
        question: str,
        reason: str,
        retrieval_hits: list[SearchHit],
        evidence: list,
        excluded: list,
        context: ContextDiagnostics,
        retrieval_info: dict[str, object],
        generation_config: GenerationConfig,
        retrieval_ms: float,
        context_ms: float,
        started: float,
    ) -> AnswerResponse:
        validation = abstained_validation(evidence)
        diagnostics = self._diagnose(
            retrieval_hits=retrieval_hits,
            excluded=excluded,
            context=context,
            validation=None,
            status="insufficient_evidence",
            cited_source_ids=[],
            evidence_ids=[getattr(item, "source_id", "") for item in evidence],
        )
        diagnostics.append(
            {"code": "generation_not_attempted", "detail": reason}
        )
        return AnswerResponse(
            question=question,
            status="insufficient_evidence",
            answer="",
            reason=reason,
            evidence=evidence,
            excluded_evidence=excluded,
            retrieved_hits=retrieval_hits,
            citation_validation=validation,
            context=context,
            retrieval=retrieval_info,
            generation={
                "model_id": generation_config.model,
                "attempted": False,
                "token_count_method": "estimated",
            },
            timings_ms={
                "retrieval": retrieval_ms,
                "context_assembly": context_ms,
                "total": round((time.perf_counter() - started) * 1000, 3),
            },
            diagnostics=diagnostics,
            generator_available=False,
        )

    def _diagnose(
        self,
        *,
        retrieval_hits: list[SearchHit],
        excluded: list[ExcludedEvidence],
        context: ContextDiagnostics,
        validation,
        status: str,
        cited_source_ids: list[str],
        evidence_ids: list[str],
    ) -> list[dict[str, str]]:
        diagnostics: list[dict[str, str]] = []
        if not retrieval_hits:
            diagnostics.append(
                {
                    "code": "no_evidence_retrieved",
                    "detail": "retrieval returned no passages for this question",
                }
            )
        elif context.selected == 0:
            diagnostics.append(
                {
                    "code": "no_evidence_selected",
                    "detail": "passages were retrieved but none fitted the context budget",
                }
            )
        if context.excluded_by_max:
            diagnostics.append(
                {
                    "code": "evidence_excluded_by_max",
                    "detail": (
                        f"{context.excluded_by_max} passage(s) were retrieved but dropped "
                        "because the evidence limit was reached"
                        + _dropped_suffix(excluded, "max_evidence")
                    ),
                }
            )
        if context.excluded_by_budget:
            diagnostics.append(
                {
                    "code": "evidence_excluded_by_budget",
                    "detail": (
                        f"{context.excluded_by_budget} passage(s) were retrieved but "
                        "excluded by the context token budget"
                        + _dropped_suffix(excluded, "context_budget")
                    ),
                }
            )
        if context.duplicates_removed:
            diagnostics.append(
                {
                    "code": "duplicate_evidence",
                    "detail": f"{context.duplicates_removed} duplicate passage(s) removed",
                }
            )
        if context.overlaps_removed:
            diagnostics.append(
                {
                    "code": "overlapping_evidence",
                    "detail": f"{context.overlaps_removed} overlapping passage(s) removed",
                }
            )
        if validation is not None:
            for issue in validation.issues:
                diagnostics.append({"code": issue.code, "detail": issue.detail})
            uncited = [
                source_id for source_id in evidence_ids if source_id not in set(cited_source_ids)
            ]
            if status in _ANSWERED_STATES and uncited:
                diagnostics.append(
                    {
                        "code": "evidence_included_but_uncited",
                        "detail": (
                            f"{len(uncited)} selected passage(s) were not cited: "
                            + ", ".join(uncited)
                        ),
                    }
                )
        if status in _ANSWERED_STATES:
            diagnostics.append(
                {
                    "code": "semantic_support_unverified",
                    "detail": (
                        "citation ids resolved, but whether the cited passages support "
                        "each claim requires human review and is not asserted here"
                    ),
                }
            )
        if status == "generation_error":
            diagnostics.append(
                {
                    "code": "generation_failed",
                    "detail": "the model did not return a usable structured answer",
                }
            )
        return diagnostics


__all__ = ["AnswerService"]
