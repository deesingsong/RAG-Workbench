"""Typed error hierarchy.

Each error maps to an actionable HTTP status in the API layer and a non-zero CLI
exit code, so failures are never reported as generic 500s.
"""

from __future__ import annotations


class RagWorkbenchError(Exception):
    """Base class for expected, user-actionable failures.

    ``code`` is a stable machine-readable identifier; ``message`` is human text.
    """

    code = "error"

    def __init__(self, message: str, *, code: str | None = None) -> None:
        super().__init__(message)
        self.message = message
        if code is not None:
            self.code = code


class ValidationError(RagWorkbenchError):
    code = "validation_error"


class UnsupportedFileTypeError(RagWorkbenchError):
    code = "unsupported_file_type"


class FileTooLargeError(RagWorkbenchError):
    code = "file_too_large"


class ParsingError(RagWorkbenchError):
    code = "parsing_error"


class IngestionError(RagWorkbenchError):
    code = "ingestion_error"


class SearchError(RagWorkbenchError):
    code = "search_error"


class NotFoundError(RagWorkbenchError):
    code = "not_found"


class ProfileNotFoundError(NotFoundError):
    code = "profile_not_found"


class DenseUnavailableError(RagWorkbenchError):
    """Dense retrieval was requested but its dependencies or model are missing."""

    code = "dense_unavailable"


class VectorStoreError(RagWorkbenchError):
    """The vector store could not be opened, queried or written."""

    code = "vector_store_error"


class DimensionMismatchError(RagWorkbenchError):
    """A vector's dimension does not match the configured profile/collection."""

    code = "dimension_mismatch"


class ProfileNotDenseError(ValidationError):
    """A dense-only operation was requested for a lexical-only profile."""

    code = "profile_not_dense"


class VectorIndexNotBuiltError(VectorStoreError):
    """A dense search ran against a profile whose vector index does not exist yet."""

    code = "vector_index_not_built"
