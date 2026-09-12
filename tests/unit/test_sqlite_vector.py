from __future__ import annotations

from pathlib import Path

import pytest

from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder, SQLiteVectorStore
from aerodiagnosis.ports import VectorChunk


def _chunk(chunk_id: str, version_id: str, content: str) -> VectorChunk:
    return VectorChunk(
        chunk_id=chunk_id,
        document_id=f"document-{chunk_id}",
        version_id=version_id,
        content=content,
        metadata={"page": 1},
    )


def test_hashing_embedding_is_deterministic_and_normalized() -> None:
    embedder = HashingEmbedder(64)

    first = embedder.embed("EGT 压气机效率下降")
    second = embedder.embed("EGT 压气机效率下降")

    assert first == second
    assert sum(value * value for value in first) == pytest.approx(1.0)


def test_vector_store_persists_ranks_and_filters_versions(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    store = SQLiteVectorStore(path, HashingEmbedder(64))
    chunks = (
        _chunk("chunk-1", "version-active", "EGT升高与压气机效率下降"),
        _chunk("chunk-2", "version-old", "滑油压力和轴承温度"),
    )

    assert store.upsert(chunks) == 2
    reopened = SQLiteVectorStore(path, HashingEmbedder(64))
    matches = reopened.search("EGT 压气机", top_k=2)

    assert reopened.backend_name == "sqlite_hashing"
    assert reopened.count() == 2
    assert matches[0].chunk.chunk_id == "chunk-1"
    assert (
        reopened.search("压力", active_version_ids=frozenset({"version-active"}))[
            0
        ].chunk.version_id
        == "version-active"
    )
    assert reopened.search("压力", active_version_ids=frozenset()) == []


def test_vector_upsert_is_idempotent_and_delete_is_scoped(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "runtime.db")
    chunk = _chunk("chunk-1", "version-1", "first")
    replacement = VectorChunk(
        chunk_id=chunk.chunk_id,
        document_id=chunk.document_id,
        version_id=chunk.version_id,
        content="replacement",
    )

    store.upsert((chunk,))
    store.upsert((replacement,))

    assert store.count() == 1
    assert store.search("replacement")[0].chunk.content == "replacement"
    assert store.delete_document(chunk.document_id) == 1
    assert store.count() == 0


def test_vector_store_rejects_invalid_requests(tmp_path: Path) -> None:
    store = SQLiteVectorStore(tmp_path / "runtime.db")

    with pytest.raises(ValueError, match="must not be empty"):
        store.upsert((_chunk("", "version", "content"),))
    with pytest.raises(ValueError, match="positive"):
        store.search("query", top_k=0)
    with pytest.raises(ValueError, match="at least 32"):
        HashingEmbedder(8)
