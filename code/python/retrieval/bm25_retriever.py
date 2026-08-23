"""
BM25 稀疏检索器 — 基于 rank_bm25 + jieba 中文分词

升级 (suggestion.md):
  1. 使用 rank_bm25 库替代 TfidfVectorizer 近似方案
  2. jieba 中文分词预处理
  3. 保留 TF-IDF fallback (rank_bm25 不可用时)
"""

from __future__ import annotations

import asyncio
from typing import Any

import numpy as np

from utils.log import logger

# Try importing rank_bm25 for true BM25; fall back to TF-IDF if unavailable
try:
    from rank_bm25 import BM25Okapi as _BM25Impl
    _HAS_RANK_BM25 = True
except ImportError:
    from sklearn.feature_extraction.text import TfidfVectorizer
    _HAS_RANK_BM25 = False

# Try jieba for Chinese word segmentation
try:
    import jieba
    _HAS_JIEBA = True
except ImportError:
    _HAS_JIEBA = False


def _tokenize(text: str) -> list[str]:
    """Tokenize text with jieba if available, otherwise character-level fallback"""
    if _HAS_JIEBA:
        tokens = list(jieba.cut(text))
    else:
        # Character-level unigram + bigram fallback for Chinese
        tokens = list(text.replace(" ", ""))
        # add bigrams
        for i in range(len(tokens) - 1):
            tokens.append(tokens[i] + tokens[i + 1])
    return tokens


class BM25Retriever:
    """BM25 稀疏检索器 (rank_bm25 + jieba)

    若 rank_bm25 未安装则降级为 TF-IDF 近似。
    """

    def __init__(self) -> None:
        self._documents: list[dict] = []
        self._bm25: Any = None  # BM25Okapi instance or TfidfVectorizer
        self._is_indexed = False
        self._using_bm25 = _HAS_RANK_BM25
        if not _HAS_RANK_BM25:
            self._vectorizer = TfidfVectorizer(
                analyzer="char", ngram_range=(1, 3), max_features=5000, sublinear_tf=True,
            )
            self._tfidf_matrix: Any = None

    async def index_documents(self, documents: list[dict]) -> None:
        """从文档列表构建/重建索引"""
        self._documents = list(documents)
        if not self._documents:
            self._is_indexed = False
            return

        texts = [d.get("content", "") for d in self._documents]
        loop = asyncio.get_event_loop()

        if _HAS_RANK_BM25:
            tokenized = await loop.run_in_executor(None, lambda: [_tokenize(t) for t in texts])
            self._bm25 = _BM25Impl(tokenized)
        else:
            self._tfidf_matrix = await loop.run_in_executor(
                None, self._vectorizer.fit_transform, texts
            )

        self._is_indexed = True
        logger.info(
            f"BM25 index built: {len(texts)} docs "
            f"({'rank_bm25 + jieba' if _HAS_RANK_BM25 and _HAS_JIEBA else 'rank_bm25' if _HAS_RANK_BM25 else 'TF-IDF fallback'})"
        )

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """BM25 检索"""
        if not self._is_indexed:
            return []

        loop = asyncio.get_event_loop()

        if _HAS_RANK_BM25 and self._bm25 is not None:
            query_tokens = await loop.run_in_executor(None, _tokenize, query)
            scores = await loop.run_in_executor(None, self._bm25.get_scores, query_tokens)
        elif hasattr(self, '_tfidf_matrix') and self._tfidf_matrix is not None:
            query_vec = await loop.run_in_executor(
                None, self._vectorizer.transform, [query]
            )
            scores = (self._tfidf_matrix @ query_vec.T).toarray().flatten()
        else:
            return []

        if len(scores) == 0:
            return []

        if top_k >= len(scores):
            top_indices = np.argsort(scores)[::-1]
        else:
            top_indices = np.argpartition(scores, -top_k)[-top_k:]
            top_indices = top_indices[np.argsort(scores[top_indices])][::-1]

        results: list[tuple[dict, float]] = []
        for idx in top_indices:
            score = float(scores[idx])
            if score > 0:
                results.append((self._documents[idx], score))
        return results[:top_k]

    def get_document_count(self) -> int:
        return len(self._documents)

    def is_ready(self) -> bool:
        return self._is_indexed
