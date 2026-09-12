"""SQLite model registry and root-cause session store."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.domain import (
    ModelPluginManifest,
    ModelPluginRecord,
    PluginStatus,
    RootCauseSession,
)

from .sqlite import SQLiteDatabase


class SQLiteKnowledgeEnhancementStore:
    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    def register_model(self, manifest: ModelPluginManifest) -> ModelPluginRecord:
        now = datetime.now(UTC).isoformat()
        payload = manifest.model_dump_json()
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO model_plugins (
                    model_id, version, dataset_id, status, manifest_json, created_at, updated_at
                ) VALUES (?, ?, ?, 'active', ?, ?, ?)
                ON CONFLICT(model_id) DO UPDATE SET
                    version=excluded.version,
                    dataset_id=excluded.dataset_id,
                    status='active',
                    manifest_json=excluded.manifest_json,
                    updated_at=excluded.updated_at
                """,
                (
                    manifest.model_id,
                    manifest.version,
                    manifest.dataset_id,
                    payload,
                    now,
                    now,
                ),
            )
            connection.commit()
        record = self.get_model(manifest.model_id)
        assert record is not None
        return record

    def list_models(self) -> tuple[ModelPluginRecord, ...]:
        with self._database.connect() as connection:
            rows = connection.execute(
                "SELECT * FROM model_plugins ORDER BY updated_at DESC, model_id"
            ).fetchall()
        return tuple(self._model_record(row) for row in rows)

    def get_model(self, model_id: str) -> ModelPluginRecord | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM model_plugins WHERE model_id = ?",
                (model_id,),
            ).fetchone()
        return None if row is None else self._model_record(row)

    def save_session(self, session: RootCauseSession) -> None:
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO root_cause_sessions (
                    rca_id, model_id, dataset_hash, status, session_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(rca_id) DO UPDATE SET
                    status=excluded.status,
                    session_json=excluded.session_json,
                    updated_at=excluded.updated_at
                """,
                (
                    session.rca_id,
                    session.model_result.model_id,
                    session.dataset.content_hash,
                    session.status.value,
                    session.model_dump_json(),
                    session.created_at,
                    session.updated_at,
                ),
            )
            connection.commit()

    def get_session(self, rca_id: str) -> RootCauseSession | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT session_json FROM root_cause_sessions WHERE rca_id = ?",
                (rca_id,),
            ).fetchone()
        return None if row is None else RootCauseSession.model_validate_json(row["session_json"])

    def list_sessions(self, *, limit: int = 50) -> tuple[RootCauseSession, ...]:
        if limit < 1 or limit > 500:
            raise ValueError("root-cause session limit must be between 1 and 500")
        with self._database.connect() as connection:
            rows = connection.execute(
                """
                SELECT session_json FROM root_cause_sessions
                ORDER BY updated_at DESC, rca_id LIMIT ?
                """,
                (limit,),
            ).fetchall()
        return tuple(RootCauseSession.model_validate_json(row["session_json"]) for row in rows)

    @staticmethod
    def _model_record(row: sqlite3.Row) -> ModelPluginRecord:
        values = dict(row)
        return ModelPluginRecord(
            manifest=ModelPluginManifest.model_validate(json.loads(values["manifest_json"])),
            status=PluginStatus(values["status"]),
            created_at=values["created_at"],
            updated_at=values["updated_at"],
        )
