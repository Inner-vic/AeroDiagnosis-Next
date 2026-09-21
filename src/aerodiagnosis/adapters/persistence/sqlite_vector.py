"""Dependency-free, persistent vector baseline implemented on SQLite."""

from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Iterable, Sequence
from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.ports import EmbeddingProvider, VectorChunk, VectorMatch

from .outbox import ExternalStoreOutbox
from .sqlite import SQLiteDatabase


class HashingEmbedder:
    """Small deterministic lexical embedding for offline smoke tests and demos.

    This is an honest baseline, not a replacement for a trained domain embedding
    model. Its value is reproducibility and zero network/model downloads.
    """

    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self.dimensions = dimensions

    @staticmethod
    def _tokens(text: str) -> Iterable[str]:
        normalized = unicodedata.normalize("NFKC", text).lower()
        yield from re.findall(r"[a-z0-9_]+", normalized)
        for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
            yield from sequence
            yield from (sequence[index : index + 2] for index in range(len(sequence) - 1))

    def embed(self, text: str) -> tuple[float, ...]:
        values = [0.0] * self.dimensions
        for token in self._tokens(text):
            digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
            bucket = int.from_bytes(digest[:4], "big") % self.dimensions
            sign = 1.0 if digest[4] & 1 else -1.0
            values[bucket] += sign
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return tuple(values)
        return tuple(value / norm for value in values)


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right):
        raise ValueError("embedding dimensions do not match")
    return sum(a * b for a, b in zip(left, right, strict=True))


class SQLiteVectorStore:
    """Portable embedded vector index with an intentionally replaceable port."""

    def __init__(
        self,
        path: Path,
        embedder: HashingEmbedder | None = None,
        embedding_provider: EmbeddingProvider | None = None,
        *,
        outbox: ExternalStoreOutbox | None = None,
    ) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()
        if embedder is not None and embedding_provider is not None:
            raise ValueError("embedder and embedding_provider cannot both be set")
        self._embedder = embedding_provider or embedder or HashingEmbedder()
        self._outbox = outbox

    @property
    def backend_name(self) -> str:
        return "sqlite_hashing"

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        now = datetime.now(UTC).isoformat()
        records = []
        for chunk in chunks:
            if not all(
                value.strip()
                for value in (chunk.chunk_id, chunk.document_id, chunk.version_id, chunk.content)
            ):
                raise ValueError("vector chunk identity and content must not be empty")
            records.append(
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.version_id,
                    chunk.content,
                    json.dumps(self._embedder.embed(chunk.content), separators=(",", ":")),
                    json.dumps(dict(chunk.metadata), ensure_ascii=False, sort_keys=True),
                    now,
                )
            )
        if not records:
            return 0
        with self._database.connect() as connection:
            connection.executemany(
                """
                INSERT INTO vector_chunks (
                    chunk_id, document_id, version_id, content, embedding_json,
                    metadata_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chunk_id) DO UPDATE SET
                    document_id=excluded.document_id,
                    version_id=excluded.version_id,
                    content=excluded.content,
                    embedding_json=excluded.embedding_json,
                    metadata_json=excluded.metadata_json,
                    updated_at=excluded.updated_at
                """,
                records,
            )
            if self._outbox is not None:
                ExternalStoreOutbox.enqueue_in_transaction(
                    connection,
                    stream="vector",
                    operation="upsert_chunks",
                    aggregate_id=hashlib.sha256(
                        "\n".join(chunk.chunk_id for chunk in chunks).encode("utf-8")
                    ).hexdigest(),
                    payload={
                        "chunks": [
                            {
                                "chunk_id": chunk.chunk_id,
                                "document_id": chunk.document_id,
                                "version_id": chunk.version_id,
                                "content": chunk.content,
                                "metadata": dict(chunk.metadata),
                            }
                            for chunk in chunks
                        ]
                    },
                )
            connection.commit()
        return len(records)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[VectorMatch]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        statement = "SELECT * FROM vector_chunks"
        params: tuple[str, ...] = ()
        if active_version_ids is not None:
            if not active_version_ids:
                return []
            ordered_versions = tuple(sorted(active_version_ids))
            placeholders = ",".join("?" for _ in ordered_versions)
            statement += f" WHERE version_id IN ({placeholders})"
            params = ordered_versions
        query_embedding = self._embedder.embed(query)
        with self._database.connect() as connection:
            rows = connection.execute(statement, params).fetchall()
        matches = []
        for row in rows:
            chunk = VectorChunk(
                chunk_id=row["chunk_id"],
                document_id=row["document_id"],
                version_id=row["version_id"],
                content=row["content"],
                metadata=json.loads(row["metadata_json"]),
            )
            stored_embedding = tuple(float(value) for value in json.loads(row["embedding_json"]))
            matches.append(
                VectorMatch(chunk=chunk, score=_cosine(query_embedding, stored_embedding))
            )
        matches.sort(key=lambda match: (-match.score, match.chunk.chunk_id))
        return matches[:top_k]

    def delete_document(self, document_id: str) -> int:
        with self._database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM vector_chunks WHERE document_id = ?", (document_id,)
            )
            if self._outbox is not None:
                ExternalStoreOutbox.enqueue_in_transaction(
                    connection,
                    stream="vector",
                    operation="delete_document",
                    aggregate_id=document_id,
                    payload={"document_id": document_id},
                )
            connection.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._database.connect() as connection:
            return int(connection.execute("SELECT count(*) FROM vector_chunks").fetchone()[0])
