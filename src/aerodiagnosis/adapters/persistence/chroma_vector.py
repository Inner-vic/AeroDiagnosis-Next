"""Chroma HTTP implementation of the vector-store port."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from aerodiagnosis.ports import EmbeddingProvider, VectorChunk, VectorMatch

from .sqlite_vector import HashingEmbedder


class ChromaHttpVectorStore:
    """Use Chroma as an external replica while preserving v3 chunk identities."""

    def __init__(
        self,
        *,
        host: str,
        port: int,
        ssl: bool,
        collection_name: str,
        embedder: HashingEmbedder | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ) -> None:
        if embedder is not None and embedding_provider is not None:
            raise ValueError("embedder and embedding_provider cannot both be set")
        self._embedder = embedding_provider or embedder or HashingEmbedder()
        self._host = host
        self._port = port
        self._ssl = ssl
        self._collection_name = collection_name
        self._client: Any = None
        self._collection_handle: Any = None

    @property
    def _collection(self) -> Any:
        if self._collection_handle is None:
            import chromadb

            self._client = chromadb.HttpClient(
                host=self._host, port=self._port, ssl=self._ssl
            )
            self._collection_handle = self._client.get_or_create_collection(
                name=self._collection_name,
                metadata={"hnsw:space": "cosine", "aerodiagnosis_schema": "v3"},
            )
        return self._collection_handle

    @property
    def backend_name(self) -> str:
        return "chroma_http_hashing"

    @staticmethod
    def _metadata(chunk: VectorChunk) -> dict[str, str]:
        return {
            "document_id": chunk.document_id,
            "version_id": chunk.version_id,
            "metadata_json": json.dumps(
                dict(chunk.metadata), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ),
        }

    @staticmethod
    def _chunk(
        chunk_id: str,
        document: str,
        metadata: Mapping[str, Any] | None,
    ) -> VectorChunk:
        values = dict(metadata or {})
        metadata_json = str(values.pop("metadata_json", "{}"))
        return VectorChunk(
            chunk_id=chunk_id,
            document_id=str(values.pop("document_id", "")),
            version_id=str(values.pop("version_id", "")),
            content=document,
            metadata=json.loads(metadata_json),
        )

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        if not chunks:
            return 0
        for chunk in chunks:
            if not all(
                value.strip()
                for value in (chunk.chunk_id, chunk.document_id, chunk.version_id, chunk.content)
            ):
                raise ValueError("vector chunk identity and content must not be empty")
        self._collection.upsert(
            ids=[chunk.chunk_id for chunk in chunks],
            embeddings=[list(self._embedder.embed(chunk.content)) for chunk in chunks],
            documents=[chunk.content for chunk in chunks],
            metadatas=[self._metadata(chunk) for chunk in chunks],
        )
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        if active_version_ids is not None and not active_version_ids:
            return []
        count = self.count()
        if count == 0:
            return []
        where: dict[str, Any] | None = None
        if active_version_ids is not None:
            ordered = sorted(active_version_ids)
            where = (
                {"version_id": ordered[0]}
                if len(ordered) == 1
                else {"version_id": {"$in": ordered}}
            )
        result: Mapping[str, Any] = self._collection.query(
            query_embeddings=[list(self._embedder.embed(query))],
            n_results=min(top_k, count),
            where=where,
            include=["documents", "metadatas", "distances"],
        )
        ids = (result.get("ids") or [[]])[0]
        documents = (result.get("documents") or [[]])[0]
        metadatas = (result.get("metadatas") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        matches = [
            VectorMatch(
                chunk=self._chunk(str(chunk_id), str(document), metadata),
                score=1.0 - float(distance),
            )
            for chunk_id, document, metadata, distance in zip(
                ids, documents, metadatas, distances, strict=True
            )
        ]
        matches.sort(key=lambda match: (-match.score, match.chunk.chunk_id))
        return matches

    def delete_document(self, document_id: str) -> int:
        result: Mapping[str, Any] = self._collection.get(
            where={"document_id": document_id}, include=[]
        )
        ids = [str(value) for value in result.get("ids", [])]
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def count(self) -> int:
        return int(self._collection.count())
