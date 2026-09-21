"""Vector retrieval port and transport-free records."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class VectorChunk:
    chunk_id: str
    document_id: str
    version_id: str
    content: str
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class VectorMatch:
    chunk: VectorChunk
    score: float


class VectorStore(Protocol):
    @property
    def backend_name(self) -> str: ...

    @property
    def embedding_identity(self) -> str: ...

    def upsert(self, chunks: Sequence[VectorChunk]) -> int: ...

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[VectorMatch]: ...

    def delete_document(self, document_id: str) -> int: ...

    def count(self) -> int: ...


class EmbeddingProvider(Protocol):
    @property
    def name(self) -> str: ...

    @property
    def dimensions(self) -> int: ...

    def embed(self, text: str) -> tuple[float, ...]: ...
