from __future__ import annotations

import sys
from collections.abc import Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import pytest

from aerodiagnosis.adapters.persistence.chroma_vector import ChromaHttpVectorStore
from aerodiagnosis.ports import VectorChunk


class FakeCollection:
    def __init__(self) -> None:
        self.records: dict[str, tuple[str, Mapping[str, Any]]] = {}

    def upsert(
        self,
        *,
        ids: Sequence[str],
        embeddings: Sequence[Sequence[float]],
        documents: Sequence[str],
        metadatas: Sequence[Mapping[str, Any]],
    ) -> None:
        assert len(embeddings) == len(ids)
        self.records.update(
            (chunk_id, (document, metadata))
            for chunk_id, document, metadata in zip(
                ids, documents, metadatas, strict=True
            )
        )

    def query(self, **kwargs: Any) -> Mapping[str, Any]:
        where = kwargs.get("where")
        selected = list(self.records.items())
        if where:
            condition = where["version_id"]
            allowed = (
                set(condition["$in"])
                if isinstance(condition, dict)
                else {condition}
            )
            selected = [
                item for item in selected if item[1][1]["version_id"] in allowed
            ]
        selected = selected[: int(kwargs["n_results"])]
        return {
            "ids": [[item[0] for item in selected]],
            "documents": [[item[1][0] for item in selected]],
            "metadatas": [[item[1][1] for item in selected]],
            "distances": [[0.1 for _item in selected]],
        }

    def get(self, **kwargs: Any) -> Mapping[str, Any]:
        document_id = kwargs["where"]["document_id"]
        return {
            "ids": [
                chunk_id
                for chunk_id, (_content, metadata) in self.records.items()
                if metadata["document_id"] == document_id
            ]
        }

    def delete(self, *, ids: Sequence[str]) -> None:
        for chunk_id in ids:
            del self.records[chunk_id]

    def count(self) -> int:
        return len(self.records)


class FakeClient:
    def __init__(self, collection: FakeCollection, **kwargs: Any) -> None:
        assert kwargs == {"host": "chroma", "port": 8000, "ssl": False}
        self.collection = collection

    def get_or_create_collection(self, **kwargs: Any) -> FakeCollection:
        assert kwargs["name"] == "test-collection"
        return self.collection


def test_chroma_adapter_round_trips_v3_identity(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = FakeCollection()
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(HttpClient=lambda **kwargs: FakeClient(collection, **kwargs)),
    )
    store = ChromaHttpVectorStore(
        host="chroma",
        port=8000,
        ssl=False,
        collection_name="test-collection",
    )
    chunks = (
        VectorChunk("one", "doc", "v1", "compressor fault", {"page": 1}),
        VectorChunk("two", "doc", "v2", "fan fault", {"page": 2}),
    )

    assert store.backend_name == "chroma_http_hashing"
    assert store.upsert(chunks) == 2
    assert store.count() == 2
    matches = store.search("compressor", top_k=2, active_version_ids=frozenset({"v1"}))
    assert matches[0].chunk == chunks[0]
    assert matches[0].score == pytest.approx(0.9)
    assert store.search("none", active_version_ids=frozenset()) == []
    assert store.delete_document("doc") == 2
    assert store.search("empty") == []


def test_chroma_adapter_rejects_invalid_requests(monkeypatch: pytest.MonkeyPatch) -> None:
    collection = FakeCollection()
    monkeypatch.setitem(
        sys.modules,
        "chromadb",
        SimpleNamespace(HttpClient=lambda **kwargs: FakeClient(collection, **kwargs)),
    )
    store = ChromaHttpVectorStore(
        host="chroma", port=8000, ssl=False, collection_name="test-collection"
    )

    with pytest.raises(ValueError, match="must not be empty"):
        store.upsert((VectorChunk("", "doc", "v1", "content"),))
    with pytest.raises(ValueError, match="positive"):
        store.search("query", top_k=0)
