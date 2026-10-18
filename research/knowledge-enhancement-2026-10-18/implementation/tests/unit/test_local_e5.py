from pathlib import Path

import pytest

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.ports import VectorChunk
from aerodiagnosis.retrieval.reranker import E5SemanticReranker, RankedText


def test_sqlite_vector_search_uses_query_role_if_supported(tmp_path):
    class Roles:
        name = "role-test"
        dimensions = 32

        def __init__(self):
            self.roles = []

        def embed(self, text):
            self.roles.append("passage")
            return (1.0,) + (0.0,) * 31

        def embed_query(self, text):
            self.roles.append("query")
            return (1.0,) + (0.0,) * 31

    provider = Roles()
    store = SQLiteVectorStore(tmp_path / "index.sqlite", embedding_provider=provider)
    store.upsert([VectorChunk("a", "d", "v", "source", {})])
    assert store.search("question")[0].chunk.chunk_id == "a"
    assert provider.roles == ["passage", "query"]


def test_real_local_e5_cross_language_relevance_and_normalization():
    root = Path(__file__).resolve().parents[2] / ".runtime/models/multilingual-e5-small"
    if not (root / "model_qint8_avx512_vnni.onnx").exists():
        pytest.skip("optional local trained model is not installed")
    provider = LocalE5EmbeddingProvider(root, threads=2)
    vector = provider.embed_query("ice ingestion can cause engine flameout")
    assert len(vector) == 384 and sum(x * x for x in vector) == pytest.approx(1, abs=1e-5)
    assert provider.name.startswith("local-e5-onnx-maskmean-windows@3:")
    candidates = [
        RankedText("unrelated", "The aircraft cabin seats were replaced.", 1),
        RankedText("engine", "Soft ice entering the engine can cause a flame-out.", 0),
    ]
    assert (
        E5SemanticReranker(provider).rerank("结冰吸入会使发动机熄火吗", candidates)[0].key
        == "engine"
    )
    long_text = "Fuel pressure was stable. " * 190 + "Ice ingestion caused flameout."
    long_vector = provider.embed(long_text)
    assert provider.windowed_encodings == 1
    assert provider.truncated_encodings == 0
    assert sum(x * x for x in long_vector) == pytest.approx(1, abs=1e-5)
    with pytest.raises(ValueError):
        provider.embed(" ")
