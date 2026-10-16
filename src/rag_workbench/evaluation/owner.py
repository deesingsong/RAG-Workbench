"""Optional, narrowly scoped owner evaluations (Laboratory Controls §5/§6).

A hosted deployment is read-only for anonymous visitors. The operator can still
run a *small, bounded* retrieval evaluation by presenting a secret token, and
only when the deployment explicitly opts in. This module holds the server-side
policy (the hard bounds) and the service that enforces it; the authorization
itself lives in the FastAPI dependency (``api.deps.require_owner``), so a UI
toggle can never grant access.

The policy is deliberately conservative and independent of the request:

* only bundled datasets on an allowlist;
* BM25 only -- no dense embedding, no reranking (model-dependent operations);
* a small query cap, a shallow candidate pool and a small runtime budget;
* a bounded, trusted-fixture ingestion rather than any bulk load; and
* a cap on concurrently running owner jobs.

Nothing here can reach bulk dataset ingestion, vector rebuilding or unrestricted
model inference, and the results are ordinary live experiments kept separate
from the committed benchmark artifacts.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from rag_workbench.config import Settings
from rag_workbench.domain.errors import OwnerLimitError
from rag_workbench.evaluation.runner import EvalRunConfig
from rag_workbench.evaluation.service import EvaluationService

#: Rank cut-offs offered to a bounded owner evaluation, filtered to the top_k.
_PREFERRED_K_VALUES = (1, 3, 5, 10)


@dataclass(frozen=True)
class OwnerEvaluationPolicy:
    """The hard, server-side bounds for a hosted owner evaluation."""

    allowed_datasets: tuple[str, ...]
    max_queries: int
    max_candidate_depth: int
    max_top_k: int
    max_documents: int
    max_concurrency: int
    max_runtime_s: float

    @classmethod
    def from_settings(cls, settings: Settings) -> OwnerEvaluationPolicy:
        return cls(
            allowed_datasets=tuple(settings.owner_allowed_datasets),
            max_queries=settings.owner_eval_max_queries,
            max_candidate_depth=settings.owner_eval_max_candidate_depth,
            max_top_k=settings.owner_eval_max_top_k,
            max_documents=settings.owner_eval_max_documents,
            max_concurrency=settings.owner_eval_max_concurrency,
            max_runtime_s=settings.owner_eval_max_runtime_s,
        )

    def summary(self) -> dict[str, object]:
        """A non-sensitive description of the enforced bounds (no token)."""

        return {
            "allowed_datasets": list(self.allowed_datasets),
            "max_queries": self.max_queries,
            "max_candidate_depth": self.max_candidate_depth,
            "max_top_k": self.max_top_k,
            "max_documents": self.max_documents,
            "max_concurrency": self.max_concurrency,
            "max_runtime_s": self.max_runtime_s,
            "modes": ["bm25"],
            "rerank": False,
        }


class OwnerEvaluationService:
    """Run one bounded owner evaluation, enforcing the policy server-side."""

    def __init__(self, settings: Settings, evaluation_service: EvaluationService) -> None:
        self._service = evaluation_service
        self._policy = OwnerEvaluationPolicy.from_settings(settings)
        self._lock = threading.Lock()
        self._in_flight = 0

    @property
    def policy(self) -> OwnerEvaluationPolicy:
        return self._policy

    def run(
        self,
        *,
        dataset: str | None = None,
        query_mode: str = "or",
        max_queries: int | None = None,
        candidate_depth: int | None = None,
        top_k: int | None = None,
        label: str | None = None,
    ) -> dict:
        """Validate against the policy, ingest the trusted fixture, evaluate."""

        policy = self._policy
        resolved_dataset = dataset or (
            policy.allowed_datasets[0] if policy.allowed_datasets else ""
        )
        if resolved_dataset not in policy.allowed_datasets:
            allowed = ", ".join(policy.allowed_datasets) or "none"
            raise OwnerLimitError(
                f"dataset '{resolved_dataset}' is not permitted for owner "
                f"evaluations; allowed datasets: {allowed}"
            )

        resolved_top_k = policy.max_top_k if top_k is None else top_k
        if resolved_top_k < 1 or resolved_top_k > policy.max_top_k:
            raise OwnerLimitError(f"top_k must be between 1 and {policy.max_top_k}")

        resolved_depth = (
            policy.max_candidate_depth if candidate_depth is None else candidate_depth
        )
        if resolved_depth < resolved_top_k or resolved_depth > policy.max_candidate_depth:
            raise OwnerLimitError(
                "candidate_depth must be between top_k "
                f"({resolved_top_k}) and {policy.max_candidate_depth}"
            )

        resolved_queries = policy.max_queries if max_queries is None else max_queries
        if resolved_queries < 1 or resolved_queries > policy.max_queries:
            raise OwnerLimitError(
                f"max_queries must be between 1 and {policy.max_queries}"
            )

        if query_mode not in ("and", "or"):
            raise OwnerLimitError("query_mode must be 'and' or 'or'")

        k_values = tuple(
            sorted({k for k in _PREFERRED_K_VALUES if k <= resolved_top_k} | {resolved_top_k})
        )

        self._acquire()
        try:
            # A bounded, trusted-fixture ingest so a fresh deployment has the
            # corpus to evaluate; never a bulk load. Ingestion is idempotent per
            # (document, profile), so repeated owner runs are cheap.
            report = self._service.ingest_dataset(
                resolved_dataset,
                embed=False,
                limit=policy.max_documents,
            )
            config = EvalRunConfig(
                dataset=resolved_dataset,
                profile_id=report.profile_id,
                collection=report.collection,
                mode="bm25",
                query_mode=query_mode,
                candidate_depth=resolved_depth,
                top_k=resolved_top_k,
                k_values=k_values,
                rerank=False,
                max_queries=resolved_queries,
                label=(label or "owner")[:120],
                max_runtime_s=policy.max_runtime_s,
            )
            return self._service.run(config)
        finally:
            self._release()

    def _acquire(self) -> None:
        with self._lock:
            if self._in_flight >= self._policy.max_concurrency:
                raise OwnerLimitError(
                    "too many owner evaluation jobs are already running; at most "
                    f"{self._policy.max_concurrency} may run concurrently"
                )
            self._in_flight += 1

    def _release(self) -> None:
        with self._lock:
            self._in_flight -= 1


__all__ = ["OwnerEvaluationPolicy", "OwnerEvaluationService"]
