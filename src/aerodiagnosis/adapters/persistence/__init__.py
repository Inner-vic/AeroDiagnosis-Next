"""Persistence adapter factories."""

from __future__ import annotations

from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import GraphStore, VectorStore

from .sqlite_cases import SQLiteCaseStore
from .sqlite_checkpoint import SQLiteCheckpointStore
from .sqlite_graph import SQLiteGraphStore
from .sqlite_knowledge_enhancement import SQLiteKnowledgeEnhancementStore
from .sqlite_memory import SQLiteConversationMemory
from .sqlite_vector import SQLiteVectorStore


class UnsupportedBackendError(ValueError):
    """Raised when optional infrastructure was selected but is not installed."""


def create_vector_store(settings: RuntimeSettings) -> VectorStore:
    if settings.vector_backend == "sqlite":
        return SQLiteVectorStore(settings.database_path)
    raise UnsupportedBackendError(
        "chroma_http is an optional deployment adapter and is not part of the "
        "local foundation runtime; use sqlite or install the external-store adapter"
    )


def create_graph_store(settings: RuntimeSettings) -> GraphStore:
    if settings.graph_backend == "sqlite":
        return SQLiteGraphStore(settings.database_path)
    raise UnsupportedBackendError(
        "neo4j is an optional deployment adapter and is not part of the local "
        "foundation runtime; use sqlite or install the external-store adapter"
    )


__all__ = [
    "SQLiteCaseStore",
    "SQLiteCheckpointStore",
    "SQLiteConversationMemory",
    "SQLiteGraphStore",
    "SQLiteKnowledgeEnhancementStore",
    "SQLiteVectorStore",
    "UnsupportedBackendError",
    "create_graph_store",
    "create_vector_store",
]
