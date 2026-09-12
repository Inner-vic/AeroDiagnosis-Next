from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from aerodiagnosis.adapters.persistence.sqlite import (
    LATEST_SCHEMA_VERSION,
    MIGRATIONS,
    SchemaVersionError,
    SQLiteDatabase,
)


def test_empty_database_migrates_to_latest_and_is_idempotent(tmp_path: Path) -> None:
    database = SQLiteDatabase(tmp_path / "runtime.db")

    assert database.migrate() == LATEST_SCHEMA_VERSION
    assert database.migrate() == LATEST_SCHEMA_VERSION
    assert database.schema_version() == LATEST_SCHEMA_VERSION


def test_existing_v1_database_migrates_without_losing_manifest_data(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.executescript(MIGRATIONS[0].sql)
        connection.execute("PRAGMA user_version = 1")
        connection.execute(
            """
            INSERT INTO documents (
                document_id, display_name, desired_revision, created_at, updated_at
            ) VALUES ('doc-1', 'manual.pdf', 0, 'now', 'now')
            """
        )
        connection.commit()

    database = SQLiteDatabase(path)
    database.migrate()

    with database.connect() as connection:
        display_name = connection.execute("SELECT display_name FROM documents").fetchone()[0]
        assert display_name == "manual.pdf"
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type = 'table'"
            ).fetchall()
        }
    assert {"graph_nodes", "graph_edges", "vector_chunks"} <= tables


def test_newer_database_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "future.db"
    with closing(sqlite3.connect(path)) as connection:
        connection.execute(f"PRAGMA user_version = {LATEST_SCHEMA_VERSION + 1}")

    with pytest.raises(SchemaVersionError, match="newer than supported"):
        SQLiteDatabase(path).migrate()
