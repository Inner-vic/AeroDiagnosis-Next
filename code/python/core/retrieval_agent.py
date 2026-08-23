"""
检索 Agent — 向量检索 + BM25 稀疏检索 + 图谱检索（三路并行）

职责:
  1. 向量语义检索 (ChromaDB)
  2. BM25 关键词检索 (稀疏表示)
  3. 知识图谱子图遍历
  4. RRF 混合融合
"""

from __future__ import annotations

from typing import Any

from retrieval.bm25_retriever import BM25Retriever
from utils.log import logger


class RetrievalAgent:
    """三路混合检索 Agent"""

    SIMILARITY_THRESHOLD = 0.60

    def __init__(
        self,
        vector_store: Any = None,
        knowledge_graph: Any = None,
    ) -> None:
        self.vector_store = vector_store
        self.knowledge_graph = knowledge_graph
        self.bm25 = BM25Retriever()

    async def vector_search(self, query: str, top_k: int = 5) -> list[dict]:
        """向量语义检索"""
        if not self.vector_store:
            return []
        try:
            results = await self.vector_store.search(query, top_k=top_k)
            return [
                {"content": doc.get("content", ""), "source": doc.get("source", ""), "score": score, "type": "vector"}
                for doc, score in results
                if score >= self.SIMILARITY_THRESHOLD
            ]
        except Exception as e:
            logger.warning(f"Vector search failed: {e}")
            return []

    async def graph_search(self, query: str, entities: list[dict]) -> list[dict]:
        """图谱检索 — 从实体出发多跳遍历"""
        if not self.knowledge_graph:
            return []
        results: list[dict] = []
        entity_names = [e.get("name", "") for e in (entities or []) if e.get("name")]
        for name in entity_names[:5]:
            try:
                neighbors = await self.knowledge_graph.get_neighbors(name, hops=2)
                for record in neighbors[:10]:
                    content = (
                        f"{record.get('source', '')} "
                        f"--[{', '.join(record.get('relations', []))}]--> "
                        f"{record.get('target', '')} "
                        f"({record.get('target_type', '')}): "
                        f"{record.get('target_desc', '')}"
                    )
                    results.append({"content": content, "source": "knowledge_graph", "score": 0.75, "type": "graph"})
            except Exception as e:
                logger.warning(f"Graph search failed for entity '{name}': {e}")
        return results

    async def bm25_search(self, query: str, top_k: int = 5) -> list[dict]:
        """BM25 稀疏检索"""
        try:
            hits = await self.bm25.search(query, top_k=top_k)
            return [
                {"content": doc.get("content", ""), "source": doc.get("source", ""), "score": min(score, 1.0), "type": "bm25"}
                for doc, score in hits
            ]
        except Exception as e:
            logger.warning(f"BM25 search failed: {e}")
            return []

    async def search_all(self, query: str, top_k: int = 6, entities: list[dict] | None = None) -> tuple:
        """三路并行检索 + RRF 融合

        Returns:
            (vec_results, bm25_results, graph_results) 三元组
        """
        vec_results = await self.vector_search(query, top_k=top_k)
        bm25_results = await self.bm25_search(query, top_k=top_k)
        kg_results = await self.graph_search(query, entities or [])

        logger.info(f"Retrieval: {len(vec_results)} vector, {len(bm25_results)} BM25, {len(kg_results)} graph")
        return vec_results, bm25_results, kg_results

    async def refresh_bm25(self) -> None:
        """从向量库全量刷新 BM25 索引"""
        docs: list[dict] = []
        if self.vector_store and hasattr(self.vector_store, '_store') and self.vector_store._store:
            try:
                # 从 ChromaDB 获取所有文档
                existing = self.vector_store._store.get(include=["documents", "metadatas"])
                texts = existing.get("documents", [])
                metas = existing.get("metadatas", [])
                for i, text in enumerate(texts):
                    meta = metas[i] if i < len(metas) else {}
                    docs.append({"content": text, "source": meta.get("source", ""), "doc_id": meta.get("doc_id", "")})
            except Exception as e:
                logger.warning(f"BM25 refresh failed to read from vector store: {e}")
        await self.bm25.index_documents(docs)
        logger.info(f"BM25 index refreshed with {len(docs)} documents")
