"""检索层: 向量召回 + 关键词召回 + 融合 + 精排."""

from app.services.retrieval.base import RetrievalTrace, RetrievedChunk
from app.services.retrieval.bm25 import BM25Retriever, get_bm25_retriever, tokenize
from app.services.retrieval.fusion import dedupe_by_parent, reciprocal_rank_fusion
from app.services.retrieval.pipeline import RetrievalResult, RetrievedContext, retrieve
from app.services.retrieval.reranker import CrossEncoderReranker, get_reranker, reset_reranker

__all__ = [
    "BM25Retriever",
    "CrossEncoderReranker",
    "RetrievalResult",
    "RetrievalTrace",
    "RetrievedChunk",
    "RetrievedContext",
    "dedupe_by_parent",
    "get_bm25_retriever",
    "get_reranker",
    "reciprocal_rank_fusion",
    "reset_reranker",
    "retrieve",
    "tokenize",
]
