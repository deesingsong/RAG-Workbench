"""BM25 lexical retrieval over SQLite FTS5.

Correctness notes:

* FTS5's ``bm25()`` returns scores where **smaller is better** (typically
  negative). This module sorts ascending and exposes a 1-based ``rank`` where
  ``1`` is always best, so callers never need to know the raw direction.
* User query text is untrusted. Each whitespace-separated token is stripped of
  double quotes and wrapped in a quoted FTS5 phrase, so operators such as
  ``NEAR``, ``AND``, ``*`` and ``:`` are treated as literal text rather than
  syntax. Tokens with no indexable token characters are dropped.
"""

from __future__ import annotations

import re

from sqlalchemy import text
from sqlalchemy.exc import DatabaseError
from sqlalchemy.orm import Session

from rag_workbench.domain.errors import SearchError, ValidationError
from rag_workbench.domain.schemas import SearchHit

_WORD_CHAR = re.compile(r"\w", re.UNICODE)

# Weighted columns of `chunk_fts`: text (1.0) and section_path (0.5).
BM25_WEIGHTS = (1.0, 0.5)

_SEARCH_SQL = """
SELECT
    c.chunk_id        AS chunk_id,
    c.document_id     AS document_id,
    c.chunk_order     AS chunk_order,
    c.text            AS chunk_text,
    c.section_path    AS section_path,
    c.section_number  AS section_number,
    c.section_title   AS section_title,
    c.parent_clause   AS parent_clause,
    c.page_start      AS page_start,
    c.page_end        AS page_end,
    d.source_name     AS source_name,
    d.collection      AS collection,
    bm25(chunk_fts, :w_text, :w_section) AS score
FROM chunk_fts
JOIN chunks c ON c.id = chunk_fts.rowid
JOIN documents d ON d.id = c.document_id
WHERE chunk_fts MATCH :match
  AND c.profile_id = :profile_id
  {collection_filter}
ORDER BY score ASC, c.id ASC
LIMIT :limit
"""


def build_match_query(raw: str) -> str:
    """Build a safe FTS5 MATCH expression from untrusted user input."""

    if raw is None or not raw.strip():
        raise ValidationError("query must not be empty", code="invalid_query")

    terms: list[str] = []
    for token in raw.split():
        cleaned = token.replace('"', "")
        if not _WORD_CHAR.search(cleaned):
            continue
        terms.append(cleaned)

    if not terms:
        raise ValidationError(
            "query contains no searchable terms", code="invalid_query"
        )

    return " ".join(f'"{term}"' for term in terms)


class Bm25Retriever:
    """Sparse retriever backed by the ``chunk_fts`` FTS5 table."""

    def search(
        self,
        session: Session,
        *,
        query: str,
        profile_id: str,
        top_k: int,
        collection: str | None = None,
    ) -> list[SearchHit]:
        match = build_match_query(query)

        collection_filter = ""
        params: dict[str, object] = {
            "match": match,
            "profile_id": profile_id,
            "w_text": BM25_WEIGHTS[0],
            "w_section": BM25_WEIGHTS[1],
            "limit": top_k,
        }
        if collection is not None:
            collection_filter = "AND d.collection = :collection"
            params["collection"] = collection

        sql = _SEARCH_SQL.format(collection_filter=collection_filter)
        try:
            rows = session.execute(text(sql), params).mappings().all()
        except DatabaseError as exc:
            raise SearchError(f"search failed: {exc}") from exc

        hits: list[SearchHit] = []
        for index, row in enumerate(rows, start=1):
            hits.append(
                SearchHit(
                    rank=index,
                    chunk_id=row["chunk_id"],
                    document_id=row["document_id"],
                    source_name=row["source_name"],
                    collection=row["collection"],
                    chunk_order=row["chunk_order"],
                    text=row["chunk_text"],
                    score=float(row["score"]),
                    section_path=row["section_path"],
                    section_number=row["section_number"],
                    section_title=row["section_title"],
                    parent_clause=row["parent_clause"],
                    page_start=row["page_start"],
                    page_end=row["page_end"],
                )
            )
        return hits


__all__ = ["BM25_WEIGHTS", "Bm25Retriever", "build_match_query"]
