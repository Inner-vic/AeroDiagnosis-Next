from __future__ import annotations

from aerodiagnosis.adapters.embeddings import HashingEmbeddingProvider


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
