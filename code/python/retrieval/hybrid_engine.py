"""
混合检索引擎 — 借鉴 MultiAgenticRAG EnsembleRetriever 模式

升级内容 (v2.1):
  1. 真正的并行检索: asyncio.gather 三路并发
  2. RRF 融合 + Ensemble 权重 (借鉴 MultiAgenticRAG 的 0.3/0.3/0.4)
  3. 查询改写生成多条子查询 (借鉴 generate_queries)
  4. 从 ChromaDB 获取 BM25 Corpus (本地实现)
"""

from __future__ import annotations

import asyncio
from typing import Any

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

from config import settings
from retrieval.bm25_retriever import BM25Retriever
from utils.log import logger

QUERY_REWRITE_PROMPT = """你是一个查询改写专家。给定用户的问题，
生成 2~3 个不同角度的检索子查询，以提高召回率。

返回 JSON: {"queries": ["子查询1", "子查询2", "子查询3"]}
只返回 JSON。"""


class HybridRetrievalEngine:
    """三路并行混合检索引擎 (相似度 + MMR + BM25 + 图谱)

    借鉴 MultiAgenticRAG 的 EnsembleRetriever 设计:
      - 多策略并行检索
      - 可配置权重
      - 查询改写增强召回
    """

    SIMILARITY_THRESHOLD = 0.55
    DEFAULT_WEIGHTS = {"vector": 0.35, "bm25": 0.35, "graph": 0.30}  # 向量+BM25等权, 图谱略低

    def __init__(self, vector_store=None, knowledge_graph=None) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.bm25 = BM25Retriever()
        self._llm = None  # lazy init for query rewriting

    @property
    def rewrite_llm(self):
        if self._llm is None:
            self._llm = ChatOpenAI(
                model=settings.openai_model,
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                temperature=0,
            )
        return self._llm

    # ── Public API ──────────────────────────────────────────────

    async def search_all(
        self, query: str, top_k: int = 8, entities: list[dict] | None = None,
    ) -> tuple[list[dict], list[dict], list[dict]]:
        """三路并行检索 (asyncio.gather 真正并发)

        Returns:
            (vector, bm25, graph) 三元组
        """
        # 可选: 查询改写生成子查询 (借鉴 MultiAgenticRAG generate_queries)
        sub_queries = await self._rewrite_query(query)

        vec_task = self._vector_search_multi(sub_queries, top_k)
        bm25_task = self._bm25_search_multi(sub_queries, top_k)
        kg_task = self._graph_search(query, entities or [])  # 图谱用原问题

        vec_results, bm25_results, kg_results = await asyncio.gather(
            vec_task, bm25_task, kg_task
        )

        logger.info(
            f"HybridRetrieval: {len(vec_results)} vec + {len(bm25_results)} BM25 "
            f"+ {len(kg_results)} graph"
        )
        return vec_results, bm25_results, kg_results

    async def fuse_results(
        self,
        vec: list[dict],
        bm25: list[dict],
        graph: list[dict],
        top_k: int = 8,
    ) -> list[dict]:
        """RRF 融合 + 去重

        借鉴 MultiAgenticRAG 的 ensemble_weights 概念,
        使用 RRF (Reciprocal Rank Fusion) 合并多路排名。
        """
        w = self.DEFAULT_WEIGHTS
        k = 60
        scores: dict[str, tuple[dict, float]] = {}

        def _key(doc):
            return doc.get("content", "")[:100]

        for rank, d in enumerate(vec or [], start=1):
            key = _key(d)
            s = w["vector"] / (k + rank)
            scores[key] = (d, scores.get(key, (d, 0))[1] + s)

        for rank, d in enumerate(bm25 or [], start=1):
            key = _key(d)
            s = w["bm25"] / (k + rank)
            scores[key] = (d, scores.get(key, (d, 0))[1] + s)

        for rank, d in enumerate(graph or [], start=1):
            key = _key(d)
            s = w["graph"] / (k + rank)
            scores[key] = (d, scores.get(key, (d, 0))[1] + s)

        ranked = sorted(scores.values(), key=lambda x: x[1], reverse=True)
        return [doc for doc, _ in ranked[:top_k]]

    # ── Internal ────────────────────────────────────────────────

    async def _rewrite_query(self, query: str) -> list[str]:
        """查询改写: 生成多条子查询 (借鉴 generate_queries)"""
        try:
            import json
            messages = [
                SystemMessage(content=QUERY_REWRITE_PROMPT),
                HumanMessage(content=query),
            ]
            resp = await self.rewrite_llm.ainvoke(messages)
            data = json.loads(resp.content.strip().split("```")[0].strip())
            sub = data.get("queries", [query])
            if query not in sub:
                sub.append(query)  # 原问题总是包含
            logger.info(f"Query rewrite: {len(sub)} sub-queries generated")
            return sub[:3]
        except Exception:
            return [query]

    async def _vector_search_multi(self, queries: list[str], top_k: int) -> list[dict]:
        """向量搜索: 多子查询去重合并"""
        if not self.vector_store:
            return []
        seen: set[str] = set()
        results: list[dict] = []
        for q in queries:
            try:
                hits = await self.vector_store.search(q, top_k=max(3, top_k // len(queries)))
                for doc, score in hits:
                    key = doc.get("content", "")[:80]
                    if key not in seen and score >= self.SIMILARITY_THRESHOLD:
                        seen.add(key)
                        results.append({
                            "content": doc.get("content", ""),
                            "source": doc.get("source", ""),
                            "score": score,
                            "type": "vector",
                        })
            except Exception as e:
                logger.warning(f"Vector search failed for sub-query: {e}")
        return results[:top_k]

    async def _bm25_search_multi(self, queries: list[str], top_k: int) -> list[dict]:
        """BM25 搜索: 多子查询去重合并"""
        seen: set[str] = set()
        results: list[dict] = []
        for q in queries:
            try:
                hits = await self.bm25.search(q, top_k=max(3, top_k // len(queries)))
                for doc, score in hits:
                    key = doc.get("content", "")[:80]
                    if key not in seen:
                        seen.add(key)
                        results.append({
                            "content": doc.get("content", ""),
                            "source": doc.get("source", ""),
                            "score": min(score, 1.0),
                            "type": "bm25",
                        })
            except Exception as e:
                logger.warning(f"BM25 search failed for sub-query: {e}")
        return results[:top_k]

    async def _graph_search(self, query: str, entities: list[dict]) -> list[dict]:
        """图谱搜索: 实体多跳遍历"""
        if not self.knowledge_graph:
            return []
        results: list[dict] = []
        entity_names = [e.get("name", "") for e in entities if e.get("name")]
        for name in entity_names[:5]:
            try:
                neighbors = await self.knowledge_graph.get_neighbors(name, hops=2)
                for record in neighbors[:10]:
                    content = (
                        f"{record.get('source', '')} "
                        f"--[{', '.join(record.get('relations', []))}]--> "
                        f"{record.get('target', '')} "
                        f"({record.get('target_type', '')})"
                    )
                    results.append({"content": content, "source": "knowledge_graph", "score": 0.75, "type": "graph"})
            except Exception as e:
                logger.warning(f"Graph search failed for '{name}': {e}")
        return results

    async def refresh_bm25(self) -> None:
        """从向量库全量刷新 BM25 索引"""
        docs: list[dict] = []
        if self.vector_store and hasattr(self.vector_store, '_store') and self.vector_store._store:
            try:
                existing = self.vector_store._store.get(include=["documents", "metadatas"])
                texts = existing.get("documents", [])
                metas = existing.get("metadatas", [])
                for i, text in enumerate(texts):
                    meta = metas[i] if i < len(metas) else {}
                    docs.append({"content": text, "source": meta.get("source", ""), "doc_id": meta.get("doc_id", "")})
            except Exception as e:
                logger.warning(f"BM25 refresh failed: {e}")
        await self.bm25.index_documents(docs)
        logger.info(f"BM25 refreshed: {len(docs)} docs")
