from __future__ import annotations

import hashlib
import sqlite3
from dataclasses import replace

import pytest

from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy


class Encoder:
    name = "test-model@1"
    dimensions = 32

    def __init__(self):
        self.passages = []

    def embed(self, text):
        self.passages.append(text)
        return (1.0, 0.0) + (0.0,) * 30 if "压气机" in text else (0.0, 1.0) + (0.0,) * 30

    def embed_query(self, text):
        return (1.0, 0.0) + (0.0,) * 30


def source():
    text = "Compressor blade damage was found."
    return VectorChunk(
        "c",
        "d",
        "v",
        text,
        {
            "locator": {"kind": "text", "coordinates": {"start": 10, "end": 10 + len(text)}},
            "display_content_zh": "发现压气机叶片损伤。",
            "display_translation_source_sha256": hashlib.sha256(text.encode()).hexdigest(),
        },
    )


def test_chinese_index_encodes_translation_without_changing_evidence(tmp_path):
    encoder = Encoder()
    aligned = SQLiteChineseVectorStore(tmp_path / "aligned.sqlite", embedding_provider=encoder)
    chunk = source()
    aligned.upsert([chunk])
    found = aligned.search("压气机损伤", active_version_ids=frozenset(["v"]))[0]
    assert encoder.passages == [chunk.metadata["display_content_zh"]]
    assert found.chunk.content == chunk.content
    assert found.chunk.metadata["locator"] == chunk.metadata["locator"]
    assert aligned.search("压气机", active_version_ids=frozenset(["other"])) == []
    with pytest.raises(ValueError, match="policy mismatch"):
        SQLiteVectorStore(tmp_path / "aligned.sqlite", embedding_provider=encoder).search("query")


def test_unrebuilt_or_changed_translation_is_rejected(tmp_path):
    path = tmp_path / "index.sqlite"
    encoder = Encoder()
    SQLiteVectorStore(path, embedding_provider=encoder).upsert([source()])
    aligned = SQLiteChineseVectorStore(path, embedding_provider=encoder)
    with pytest.raises(ValueError, match="rebuild"):
        aligned.search("query")
    stale = replace(source(), content="Changed source")
    with pytest.raises(ValueError, match="stale"):
        aligned.upsert([stale])
    aligned.upsert([source()])
    with sqlite3.connect(path) as db:
        db.execute(
            "UPDATE vector_chunks SET metadata_json=replace(metadata_json, ?, ?)",
            ("发现压气机叶片损伤。", "发现压气机结冰。"),
        )
    db.close()
    with pytest.raises(ValueError, match="stale"):
        aligned.search("query")


def test_model_identity_and_missing_translation_guard(tmp_path):
    path = tmp_path / "index.sqlite"
    a = Encoder()
    store = SQLiteChineseVectorStore(path, embedding_provider=a)
    with pytest.raises(ValueError, match="requires"):
        store.upsert([VectorChunk("x", "d", "v", "Untranslated English", {})])
    store.upsert([source()])
    b = Encoder()
    b.name = "different-model"
    with pytest.raises(ValueError, match="rebuild"):
        SQLiteChineseVectorStore(path, embedding_provider=b).search("query")


class Route:
    backend_name = "test-route"
    embedding_identity = "test-route"

    def __init__(self, ids):
        self.ids = ids
        self.calls = 0

    def search(self, query, *, top_k=5, active_version_ids=None):
        self.calls += 1
        return [VectorMatch(VectorChunk(k, "d" + k, "v", k, {}), 1.0) for k in self.ids][:top_k]


def test_weighted_fusion_changes_order_and_zero_weight_does_not_dispatch():
    vector, sparse = Route(["a", "b"]), Route(["b", "c"])

    def run(weight):
        return ManualEvidenceRetriever(
            vector_store=vector,
            lexical_store=sparse,
            policy=ManualRetrievalPolicy(
                vector_weight=weight, sparse_weight=1, max_per_document=0, expand_context=False
            ),
        ).search("query", top_k=3)

    low, high = run(0.25), run(2)
    assert [m.chunk.chunk_id for m in low] == ["b", "c", "a"]
    assert [m.chunk.chunk_id for m in high] == ["b", "a", "c"]
    before = vector.calls
    assert [m.chunk.chunk_id for m in run(0)] == ["b", "c"]
    assert vector.calls == before


@pytest.mark.parametrize("weight", [float("nan"), float("inf"), -0.1, 5.1])
def test_invalid_weights_rejected(weight):
    with pytest.raises(ValueError, match="weights"):
        ManualRetrievalPolicy(vector_weight=weight)
