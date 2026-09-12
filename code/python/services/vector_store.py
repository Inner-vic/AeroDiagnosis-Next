"""
向量存储服务 — 可配置 ChromaDB 嵌入式 / HTTP 后端

职责:
  1. 文档块向量化 (Embedding)
  2. 向量存储 & 检索
  3. 按 doc_id 删除（支持增量更新）
"""

from __future__ import annotations

from typing import Any

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from agents.doc_parser_agent import DocumentChunk
from config import settings


def _create_embeddings():
    """根据配置创建 Embedding 实例，DeepSeek 等不提供 Embedding API 的用本地模型"""
    if "deepseek" in settings.openai_base_url:
        return HuggingFaceEmbeddings(
            model_name="shibing624/text2vec-base-chinese",
            model_kwargs={"device": "cpu"},
        )
    return OpenAIEmbeddings(
        model=settings.embedding_model,
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )


class VectorStoreService:
    """旧应用的兼容访问层；本机默认使用嵌入式持久化。"""

    COLLECTION_NAME = "knowledge_chunks"

    def __init__(self) -> None:
        self.embeddings = _create_embeddings()
        self._store: Any = None
        self._client: Any = None
        self._backend = settings.vector_store_backend

    # ── initialization ───────────────────────────────────────

    async def _ready(self) -> bool:
        """检查向量存储是否已初始化可用"""
        return self._store is not None

    async def init(self) -> None:
        await self._init_chroma()

    async def _init_chroma(self) -> None:
        import chromadb

        if self._backend == "embedded":
            import os

            os.makedirs(settings.chroma_persist_dir, exist_ok=True)
            self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
        elif self._backend == "chroma_http":
            self._client = chromadb.HttpClient(
                host=settings.chroma_host,
                port=settings.chroma_port,
            )
        else:
            raise ValueError(f"Unsupported vector store backend: {self._backend}")
        self._store = self._client.get_or_create_collection(
            name=self.COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    # ── CRUD ─────────────────────────────────────────────────

    async def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """向量化并存储文档块"""
        if not await self._ready():
            return 0
        if not chunks:
            return 0

        texts = [c.content for c in chunks]
        ids = [c.chunk_id for c in chunks]
        metadatas = [
            {"doc_id": c.doc_id, "doc_type": c.doc_type.value, "source": c.metadata.get("source", ""), "chunk_index": c.chunk_index}
            for c in chunks
        ]

        vectors = await self.embeddings.aembed_documents(texts)
        self._store.upsert(ids=ids, embeddings=vectors, documents=texts, metadatas=metadatas)

        return len(chunks)

    async def search(self, query: str, top_k: int = 5) -> list[tuple[dict, float]]:
        """语义搜索，返回 (文档, 分数) 列表"""
        if not await self._ready():
            return []
        q_vec = await self.embeddings.aembed_query(query)
        results = self._store.query(
            query_embeddings=[q_vec],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )
        out: list[tuple[dict, float]] = []
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for doc, meta, dist in zip(docs, metas, dists):
            score = 1.0 - dist  # cosine distance → similarity
            out.append(({"content": doc, "source": meta.get("source", ""), "metadata": meta}, score))
        return out

    async def delete_by_doc_id(self, doc_id: str) -> int:
        """按 doc_id 删除所有相关向量"""
        if not await self._ready():
            return 0
        existing = self._store.get(where={"doc_id": doc_id}, include=[])
        ids = existing.get("ids", [])
        if ids:
            self._store.delete(ids=ids)
        return len(ids)

    async def get_stats(self) -> dict:
        """获取向量库统计信息"""
        if not await self._ready():
            return {"backend": self._backend, "total_vectors": 0, "collection": self.COLLECTION_NAME}
        count = self._store.count()
        return {
            "backend": self._backend,
            "total_vectors": count,
            "collection": self.COLLECTION_NAME,
        }

    @property
    def backend_name(self) -> str:
        return self._backend
