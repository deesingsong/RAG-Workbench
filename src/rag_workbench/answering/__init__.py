"""Optional evidence-grounded answer generation (Phase 5).

The pipeline is deliberately staged and inspectable::

    retrieval -> evidence selection -> context assembly -> local LLM generation
    -> structured-output validation -> deterministic citation verification
    -> answer diagnostics

Nothing here re-implements retrieval — it consumes
:class:`~rag_workbench.retrieval.service.SearchService` — and nothing here requires
Ollama until live generation is explicitly enabled and requested. The public
read-only deployment never runs generation; it serves committed *recorded*
demonstrations instead.
"""

from __future__ import annotations

from rag_workbench.answering.citations import verify_citations
from rag_workbench.answering.context import ContextAssembler, estimate_tokens
from rag_workbench.answering.pipeline import AnswerService

__all__ = [
    "AnswerService",
    "ContextAssembler",
    "estimate_tokens",
    "verify_citations",
]
