"""Chinese passage vectors with original evidence content and strict translation provenance."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Sequence
from dataclasses import replace
from pathlib import Path

from aerodiagnosis.ports import EmbeddingProvider, VectorChunk

from .sqlite_vector import SQLiteVectorStore

POLICY = "chinese_display@1"


class SQLiteChineseVectorStore(SQLiteVectorStore):
    """Local index variant; rebuild before use, never reinterpret source-language vectors."""

    def __init__(self, path: Path, *, embedding_provider: EmbeddingProvider) -> None:
        super().__init__(path, embedding_provider=embedding_provider)

    @property
    def embedding_identity(self) -> str:
        return self._embedder.name + ":" + POLICY

    def _embedding_input(self, chunk: VectorChunk) -> str:
        display = chunk.metadata.get("display_content_zh")
        if isinstance(display, str) and display.strip():
            if (
                chunk.metadata.get("display_translation_source_sha256")
                != hashlib.sha256(chunk.content.encode()).hexdigest()
            ):
                raise ValueError("Chinese vector input has stale/unverified translation")
            return display
        if re.search(r"[\u3400-\u9fff]", chunk.content):
            return chunk.content
        raise ValueError("Chinese vector input requires a source-verified translation")

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        aligned = [
            replace(
                chunk,
                metadata={
                    **chunk.metadata,
                    "vector_embedding_input": {
                        "policy": POLICY,
                        "provider": self._embedder.name,
                        "sha256": hashlib.sha256(self._embedding_input(chunk).encode()).hexdigest(),
                    },
                },
            )
            for chunk in chunks
        ]
        return super().upsert(aligned)

    def _validate_embedding_row(self, chunk: VectorChunk) -> None:
        expected = {
            "policy": POLICY,
            "provider": self._embedder.name,
            "sha256": hashlib.sha256(self._embedding_input(chunk).encode()).hexdigest(),
        }
        if chunk.metadata.get("vector_embedding_input") != expected:
            raise ValueError("Chinese vector index is missing/stale; rebuild before searching")
