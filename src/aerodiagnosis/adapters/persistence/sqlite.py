"""SQLite connection and forward-only schema migration support."""

from __future__ import annotations

import sqlite3
from collections.abc import Iterator
from contextlib import closing, contextmanager
from dataclasses import dataclass
from pathlib import Path

LATEST_SCHEMA_VERSION = 6


@dataclass(frozen=True, slots=True)
class Migration:
    version: int
    name: str
    sql: str


MIGRATIONS = (
    Migration(
        1,
        "runtime_ledger",
        """
        CREATE TABLE IF NOT EXISTS documents (
            document_id TEXT PRIMARY KEY,
            display_name TEXT NOT NULL,
            active_version_id TEXT,
            desired_revision INTEGER NOT NULL DEFAULT 0,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS document_versions (
            version_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL REFERENCES documents(document_id) ON DELETE CASCADE,
            revision INTEGER NOT NULL,
            parser_version TEXT NOT NULL,
            content_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK (status IN (
                'received', 'parsed', 'indexing', 'staged', 'active', 'failed',
                'superseded', 'deleting', 'deleted'
            )),
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            UNIQUE(document_id, revision)
        );
        CREATE TABLE IF NOT EXISTS jobs (
            job_id TEXT PRIMARY KEY,
            idempotency_key TEXT NOT NULL UNIQUE,
            status TEXT NOT NULL,
            fencing_token INTEGER NOT NULL DEFAULT 0,
            lease_owner TEXT,
            lease_expires_at TEXT,
            heartbeat_at TEXT,
            attempt INTEGER NOT NULL DEFAULT 0,
            max_attempts INTEGER NOT NULL DEFAULT 3,
            not_before TEXT,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS cases (
            case_id TEXT PRIMARY KEY,
            version INTEGER NOT NULL,
            payload_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS checkpoints (
            run_id TEXT NOT NULL,
            checkpoint_seq INTEGER NOT NULL,
            workflow_version TEXT NOT NULL,
            state_schema_version TEXT NOT NULL,
            snapshot_id TEXT NOT NULL,
            state_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            PRIMARY KEY(run_id, checkpoint_seq)
        );
        CREATE TABLE IF NOT EXISTS tool_ledger (
            operation_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL,
            call_id TEXT NOT NULL,
            tool_name TEXT NOT NULL,
            status TEXT NOT NULL,
            request_hash TEXT NOT NULL,
            result_json TEXT,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        """,
    ),
    Migration(
        2,
        "embedded_graph",
        """
        CREATE TABLE IF NOT EXISTS graph_nodes (
            node_id TEXT PRIMARY KEY,
            name TEXT NOT NULL,
            kind TEXT NOT NULL,
            description TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            version_id TEXT NOT NULL,
            properties_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_name ON graph_nodes(name);
        CREATE INDEX IF NOT EXISTS idx_graph_nodes_version ON graph_nodes(version_id);
        CREATE TABLE IF NOT EXISTS graph_edges (
            edge_id TEXT PRIMARY KEY,
            source_id TEXT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
            target_id TEXT NOT NULL REFERENCES graph_nodes(node_id) ON DELETE CASCADE,
            relation TEXT NOT NULL,
            source_ref TEXT NOT NULL,
            version_id TEXT NOT NULL,
            confidence REAL NOT NULL CHECK(confidence >= 0.0 AND confidence <= 1.0),
            properties_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_graph_edges_source ON graph_edges(source_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_target ON graph_edges(target_id);
        CREATE INDEX IF NOT EXISTS idx_graph_edges_version ON graph_edges(version_id);
        """,
    ),
    Migration(
        3,
        "embedded_vector",
        """
        CREATE TABLE IF NOT EXISTS vector_chunks (
            chunk_id TEXT PRIMARY KEY,
            document_id TEXT NOT NULL,
            version_id TEXT NOT NULL,
            content TEXT NOT NULL,
            embedding_json TEXT NOT NULL,
            metadata_json TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_vector_document ON vector_chunks(document_id);
        CREATE INDEX IF NOT EXISTS idx_vector_version ON vector_chunks(version_id);
        """,
    ),
    Migration(
        4,
        "conversation_memory",
        """
        CREATE TABLE IF NOT EXISTS conversation_sessions (
            session_id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS conversation_messages (
            message_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL REFERENCES conversation_sessions(session_id) ON DELETE CASCADE,
            ordinal INTEGER NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user', 'assistant', 'tool')),
            content TEXT NOT NULL,
            created_at TEXT NOT NULL,
            UNIQUE(session_id, ordinal)
        );
        CREATE INDEX IF NOT EXISTS idx_conversation_messages_session
            ON conversation_messages(session_id, ordinal);
        """,
    ),
    Migration(
        5,
        "legacy_migration_ledger",
        """
        CREATE TABLE IF NOT EXISTS legacy_migration_ledger (
            source_key TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            source_hash TEXT NOT NULL,
            document_id TEXT NOT NULL,
            version_id TEXT,
            status TEXT NOT NULL CHECK(status IN ('imported', 'failed')),
            error TEXT,
            migrated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_legacy_migration_status
            ON legacy_migration_ledger(status);
        """,
    ),
    Migration(
        6,
        "knowledge_enhancement",
        """
        CREATE TABLE IF NOT EXISTS model_plugins (
            model_id TEXT PRIMARY KEY,
            version TEXT NOT NULL,
            dataset_id TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN ('active', 'disabled')),
            manifest_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_model_plugins_dataset
            ON model_plugins(dataset_id, status);
        CREATE TABLE IF NOT EXISTS root_cause_sessions (
            rca_id TEXT PRIMARY KEY,
            model_id TEXT NOT NULL,
            dataset_hash TEXT NOT NULL,
            status TEXT NOT NULL CHECK(status IN (
                'investigating', 'ready_for_report', 'completed'
            )),
            session_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL
        );
        CREATE INDEX IF NOT EXISTS idx_root_cause_sessions_updated
            ON root_cause_sessions(updated_at DESC);
        """,
    ),
)


class SchemaVersionError(RuntimeError):
    pass


class SQLiteDatabase:
    """Open short-lived configured connections and keep migrations explicit."""

    def __init__(self, path: Path) -> None:
        self.path = path

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        connection.execute("PRAGMA busy_timeout = 10000")
        try:
            yield connection
        finally:
            connection.close()

    def migrate(self) -> int:
        with self.connect() as connection:
            current = int(connection.execute("PRAGMA user_version").fetchone()[0])
            if current > LATEST_SCHEMA_VERSION:
                raise SchemaVersionError(
                    f"database schema {current} is newer than supported {LATEST_SCHEMA_VERSION}"
                )
            for migration in MIGRATIONS:
                if migration.version <= current:
                    continue
                try:
                    connection.executescript(
                        "BEGIN IMMEDIATE;\n"
                        f"{migration.sql}\n"
                        f"PRAGMA user_version = {migration.version};\n"
                        "COMMIT;"
                    )
                except sqlite3.DatabaseError as exc:
                    if connection.in_transaction:
                        connection.rollback()
                    raise SchemaVersionError(
                        f"migration {migration.version} ({migration.name}) failed"
                    ) from exc
                current = migration.version
            return current

    def schema_version(self) -> int:
        with self.connect() as connection:
            return int(connection.execute("PRAGMA user_version").fetchone()[0])

    def backup_to(self, destination: Path) -> Path:
        """Create a consistent SQLite backup without mutating the source database."""

        resolved = destination.resolve()
        if resolved == self.path.resolve():
            raise ValueError("backup destination must differ from the live database")
        resolved.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as source, closing(sqlite3.connect(resolved)) as target:
            source.backup(target)
        return resolved
