"""Persistence adapter factories."""

from __future__ import annotations

from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import GraphStore, VectorStore

from .chroma_vector import ChromaHttpVectorStore
from .neo4j_graph import Neo4jGraphStore
from .outbox import ExternalStoreOutbox
from .replicated import ExternalStoreSync, ReplicatedGraphStore, ReplicatedVectorStore
from .sqlite_cases import SQLiteCaseStore
from .sqlite_checkpoint import SQLiteCheckpointStore
from .sqlite_graph import SQLiteGraphStore
from .sqlite_knowledge_enhancement import SQLiteKnowledgeEnhancementStore
from .sqlite_memory import SQLiteConversationMemory
from .sqlite_vector import SQLiteVectorStore


class UnsupportedBackendError(ValueError):
    """Raised when optional infrastructure was selected but is not installed."""


def _chroma_vector_store(settings: RuntimeSettings) -> ChromaHttpVectorStore:
    return ChromaHttpVectorStore(
        host=settings.chroma_host,
        port=settings.chroma_port,
        ssl=settings.chroma_ssl,
        collection_name=settings.chroma_collection,
    )


def _neo4j_graph_store(settings: RuntimeSettings) -> Neo4jGraphStore:
    if settings.neo4j_password is None:
        raise UnsupportedBackendError(
            "AERODIAGNOSIS_NEO4J_PASSWORD is required when graph backend is neo4j"
        )
    return Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )


def create_vector_store(settings: RuntimeSettings) -> VectorStore:
    if settings.vector_backend == "sqlite":
        return SQLiteVectorStore(settings.database_path)
    if settings.external_store_mode == "external-primary":
        return _chroma_vector_store(settings)
    outbox = ExternalStoreOutbox(settings.database_path)
    primary = SQLiteVectorStore(settings.database_path, outbox=outbox)
    replica = _chroma_vector_store(settings)
    sync = ExternalStoreSync(
        database_path=settings.database_path,
        outbox=outbox,
        vector_replica=replica,
        max_attempts=settings.sync_max_attempts,
    )
    sync.enqueue_backfill(("vector",))
    return ReplicatedVectorStore(
        primary, replica, sync, batch_size=settings.sync_batch_size
    )


def create_graph_store(settings: RuntimeSettings) -> GraphStore:
    if settings.graph_backend == "sqlite":
        return SQLiteGraphStore(settings.database_path)
    if settings.external_store_mode == "external-primary":
        return _neo4j_graph_store(settings)
    outbox = ExternalStoreOutbox(settings.database_path)
    primary = SQLiteGraphStore(settings.database_path, outbox=outbox)
    replica = _neo4j_graph_store(settings)
    sync = ExternalStoreSync(
        database_path=settings.database_path,
        outbox=outbox,
        graph_replica=replica,
        max_attempts=settings.sync_max_attempts,
    )
    sync.enqueue_backfill(("graph",))
    return ReplicatedGraphStore(primary, replica, sync, batch_size=settings.sync_batch_size)


def create_external_sync(settings: RuntimeSettings) -> ExternalStoreSync:
    if settings.external_store_mode == "external-primary":
        raise UnsupportedBackendError(
            "external sync is disabled in external-primary mode"
        )
    outbox = ExternalStoreOutbox(settings.database_path)
    vector_replica: VectorStore | None = None
    graph_replica: GraphStore | None = None
    if settings.vector_backend == "chroma_http":
        vector_replica = _chroma_vector_store(settings)
    if settings.graph_backend == "neo4j":
        graph_replica = _neo4j_graph_store(settings)
    return ExternalStoreSync(
        database_path=settings.database_path,
        outbox=outbox,
        vector_replica=vector_replica,
        graph_replica=graph_replica,
        max_attempts=settings.sync_max_attempts,
    )


__all__ = [
    "ChromaHttpVectorStore",
    "ExternalStoreOutbox",
    "ExternalStoreSync",
    "Neo4jGraphStore",
    "ReplicatedGraphStore",
    "ReplicatedVectorStore",
    "SQLiteCaseStore",
    "SQLiteCheckpointStore",
    "SQLiteConversationMemory",
    "SQLiteGraphStore",
    "SQLiteKnowledgeEnhancementStore",
    "SQLiteVectorStore",
    "UnsupportedBackendError",
    "create_external_sync",
    "create_graph_store",
    "create_vector_store",
]
