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


class RerankerUnavailableError(RagWorkbenchError):
    """Reranking was requested but its dependencies or model are missing."""

    code = "reranker_unavailable"


class RerankError(RagWorkbenchError):
    """The reranker ran but the request could not be satisfied."""

    code = "rerank_error"


class DatasetError(RagWorkbenchError):
    """An evaluation dataset is missing, malformed or internally inconsistent."""

    code = "dataset_error"


class DatasetNotFoundError(NotFoundError):
    """A named evaluation dataset does not exist in any configured root."""

    code = "dataset_not_found"


class ExperimentNotFoundError(NotFoundError):
    """A recorded evaluation experiment does not exist."""

    code = "experiment_not_found"


class EvaluationError(RagWorkbenchError):
    """An evaluation run could not be executed or configured."""

    code = "evaluation_error"


class ReadOnlyModeError(RagWorkbenchError):
    """A mutating operation was attempted against a read-only demonstration instance."""

    code = "read_only_mode"


class OwnerAuthError(RagWorkbenchError):
    """An owner-only request lacked a valid, configured owner test token."""

    code = "owner_unauthorized"


class OwnerEvaluationsDisabledError(RagWorkbenchError):
    """Owner evaluations were requested but are not enabled on this deployment."""

    code = "owner_evaluations_disabled"


class OwnerLimitError(RagWorkbenchError):
    """An owner evaluation exceeded one of the server-side bounds."""

    code = "owner_limit_exceeded"


class GenerationError(RagWorkbenchError):
    """Answer generation was requested but could not be completed."""

    code = "generation_error"


class GeneratorUnavailableError(RagWorkbenchError):
    """The local answer generator (Ollama) is disabled, unreachable or missing a model."""

    code = "generator_unavailable"


class AnswerNotFoundError(NotFoundError):
    """A recorded answer demonstration does not exist."""

    code = "answer_not_found"


class ScalingError(RagWorkbenchError):
    """A scalability benchmark, capacity calculation or artifact is invalid."""

    code = "scaling_error"


class ScalingBenchmarkNotFoundError(NotFoundError):
    """A recorded scalability benchmark does not exist."""

    code = "scaling_benchmark_not_found"
