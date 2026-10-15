"""Detection of optional, heavyweight extras.

The default install intentionally excludes the deep-learning stack, so the app
must be able to answer "is dense retrieval / cross-encoder reranking available
here?" without importing (or failing to import) those packages.
"""

from __future__ import annotations

import importlib.util


def dense_feature_available() -> bool:
    """True when ``sentence-transformers`` is importable (the ``dense`` extra)."""

    return importlib.util.find_spec("sentence_transformers") is not None


#: Reranking depends on the same optional extra as dense retrieval.
rerank_feature_available = dense_feature_available

__all__ = ["dense_feature_available", "rerank_feature_available"]
