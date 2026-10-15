"""Import a dataset's corpus into the index through the ordinary ingestion path.

The evaluation engine never writes its own index: it feeds each benchmark
document through :class:`~rag_workbench.ingestion.pipeline.IngestionService`, the
same write path as an uploaded file, preserving the original document ID in the
``external_id`` column so retrieved chunks can be mapped back to benchmark IDs.

Ingestion is idempotent per (document, profile), so re-importing a corpus is a
no-op for documents already indexed. Documents whose *content* collides (identical
bytes) collapse to one row, which is reported rather than hidden — a genuinely
duplicate corpus document cannot be distinguished by content checksum alone.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from rag_workbench.config import IndexProfileSpec
from rag_workbench.evaluation.datasets import CorpusDocument, EvaluationDataset
from rag_workbench.ingestion.pipeline import IngestionService


class CorpusImportReport(BaseModel):
    """Outcome of importing a dataset corpus into one profile/collection."""

    dataset: str
    collection: str
    profile_id: str
    documents: int
    new_documents: int
    duplicates: int
    chunks: int
    mime_type: str = "text/plain"
    diagnostics: list[str] = Field(default_factory=list)


def import_corpus(
    ingestion: IngestionService,
    dataset: EvaluationDataset,
    *,
    collection: str,
    profile: IndexProfileSpec,
    documents: list[CorpusDocument] | None = None,
    limit: int | None = None,
) -> CorpusImportReport:
    """Ingest a dataset's documents and return a summary report."""

    selected = documents if documents is not None else dataset.documents
    if limit is not None:
        selected = selected[:limit]

    new_documents = 0
    duplicates = 0
    chunks = 0
    diagnostics: list[str] = []
    seen_checksums: set[str] = set()

    for document in selected:
        data = document.index_text().encode("utf-8")
        result = ingestion.ingest_bytes(
            data,
            source_name=f"{document.id}.txt",
            collection=collection,
            profile=profile,
            external_id=document.id,
        )
        if result.is_duplicate:
            duplicates += 1
        else:
            new_documents += 1
        chunks += result.chunk_count
        if result.checksum in seen_checksums:
            diagnostics.append(
                f"duplicate corpus content for document '{document.id}' (identical bytes); "
                "it shares a document row with an earlier id"
            )
        seen_checksums.add(result.checksum)

    report = CorpusImportReport(
        dataset=dataset.name,
        collection=collection,
        profile_id=profile.key,
        documents=len(selected),
        new_documents=new_documents,
        duplicates=duplicates,
        chunks=chunks,
        diagnostics=diagnostics,
    )
    return report


__all__ = ["CorpusImportReport", "import_corpus"]
