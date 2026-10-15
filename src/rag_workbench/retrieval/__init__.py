"""Retrieval layer: BM25, dense, hybrid (RRF) and the retrieval service."""

from rag_workbench.retrieval.bm25 import Bm25Retriever, build_match_query
from rag_workbench.retrieval.dense import DenseRetriever
from rag_workbench.retrieval.hybrid import reciprocal_rank_fusion
from rag_workbench.retrieval.service import SearchService

__all__ = [
    "Bm25Retriever",
    "DenseRetriever",
    "SearchService",
    "build_match_query",
    "reciprocal_rank_fusion",
]
