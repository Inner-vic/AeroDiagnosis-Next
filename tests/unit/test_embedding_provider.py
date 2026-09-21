from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.embeddings import (
    HashingEmbeddingProvider,
    OpenAICompatibleEmbeddingProvider,
)
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.ports import VectorChunk


def test_hashing_embedding_provider_is_reproducible_and_named() -> None:
    provider = HashingEmbeddingProvider(dimensions=128)

    first = provider.embed("compressor stall")
    second = provider.embed("compressor stall")

    assert provider.name == "hashing@128"
    assert provider.dimensions == 128
    assert len(first) == 128
    assert first == second


def test_hashing_embedding_provider_rejects_small_dimensions() -> None:
    import pytest

    with pytest.raises(ValueError, match="dimensions"):
        HashingEmbeddingProvider(dimensions=16)


def test_sqlite_vector_store_accepts_embedding_provider(tmp_path: Path) -> None:
    store = SQLiteVectorStore(
        tmp_path / "runtime.db",
        embedding_provider=HashingEmbeddingProvider(dimensions=128),
    )
    store.upsert(
        (
            VectorChunk(
                "chunk-1",
                "doc-1",
                "version-1",
                "Compressor stall causes EGT rise.",
            ),
        )
    )

    matches = store.search(
        "compressor stall",
        active_version_ids=frozenset({"version-1"}),
    )

    assert matches[0].chunk.chunk_id == "chunk-1"


def test_openai_compatible_embedding_provider_calls_embeddings_endpoint(
    monkeypatch: object,
) -> None:
    class FakeResponse:
        def __enter__(self) -> object:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def read(self) -> bytes:
            return (
                b'{"data":[{"embedding":[0.1,0.2,0.3]}]}'
            )

    def fake_urlopen(_request: object, timeout: object = None) -> FakeResponse:
        del timeout
        return FakeResponse()

    monkeypatch.setattr(
        "urllib.request.urlopen",
        fake_urlopen,
    )
    provider = OpenAICompatibleEmbeddingProvider(
        base_url="https://provider.example/v1",
        api_key="secret",
        model="embedding-model",
        dimensions=3,
    )

    assert provider.name == "openai_compatible:embedding-model:3"
    assert provider.embed("compressor") == (0.1, 0.2, 0.3)
