from __future__ import annotations

from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder


class HashingEmbeddingProvider:
    def __init__(self, dimensions: int = 256) -> None:
        if dimensions < 32:
            raise ValueError("dimensions must be at least 32")
        self._impl = HashingEmbedder(dimensions=dimensions)
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return f"hashing@{self.dimensions}"

    def embed(self, text: str) -> tuple[float, ...]:
        return self._impl.embed(text)
