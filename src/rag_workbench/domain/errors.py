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
