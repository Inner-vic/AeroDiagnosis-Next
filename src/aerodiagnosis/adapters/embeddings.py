from __future__ import annotations

import json
import urllib.request

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


class OpenAICompatibleEmbeddingProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        dimensions: int,
    ) -> None:
        if dimensions < 1:
            raise ValueError("dimensions must be positive")
        self._endpoint = base_url.rstrip("/") + "/embeddings"
        self._api_key = api_key
        self._model = model
        self.dimensions = dimensions

    @property
    def name(self) -> str:
        return f"openai_compatible:{self._model}:{self.dimensions}"

    def embed(self, text: str) -> tuple[float, ...]:
        payload = json.dumps({"model": self._model, "input": text}).encode()
        request = urllib.request.Request(
            self._endpoint,
            data=payload,
            headers={
                "Authorization": f"Bearer {self._api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(request, timeout=30) as response:
            body = json.loads(response.read().decode())
        vector = tuple(float(value) for value in body["data"][0]["embedding"])
        if len(vector) != self.dimensions:
            raise ValueError(
                f"embedding returned {len(vector)} dimensions, expected {self.dimensions}"
            )
        return vector
