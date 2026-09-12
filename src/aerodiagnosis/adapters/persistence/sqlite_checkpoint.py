"""SQLite implementation of durable workflow checkpoints."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.ports import CheckpointRecord

from .sqlite import SQLiteDatabase


class SQLiteCheckpointStore:
    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    def save(
        self,
        *,
        run_id: str,
        workflow_version: str,
        state_schema_version: str,
        snapshot_id: str,
        state_json: str,
    ) -> CheckpointRecord:
        if not all(value.strip() for value in (run_id, workflow_version, state_json)):
            raise ValueError("checkpoint identity and state must not be empty")
        with self._database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                "SELECT max(checkpoint_seq) FROM checkpoints WHERE run_id = ?",
                (run_id,),
            ).fetchone()
            sequence = int(row[0] or 0) + 1
            connection.execute(
                """
                INSERT INTO checkpoints (
                    run_id, checkpoint_seq, workflow_version, state_schema_version,
                    snapshot_id, state_json, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    run_id,
                    sequence,
                    workflow_version,
                    state_schema_version,
                    snapshot_id,
                    state_json,
                    datetime.now(UTC).isoformat(),
                ),
            )
            connection.commit()
        return CheckpointRecord(
            run_id=run_id,
            sequence=sequence,
            workflow_version=workflow_version,
            state_schema_version=state_schema_version,
            snapshot_id=snapshot_id,
            state_json=state_json,
        )

    def latest(self, run_id: str) -> CheckpointRecord | None:
        with self._database.connect() as connection:
            row = connection.execute(
                """
                SELECT * FROM checkpoints
                WHERE run_id = ?
                ORDER BY checkpoint_seq DESC
                LIMIT 1
                """,
                (run_id,),
            ).fetchone()
        if row is None:
            return None
        return CheckpointRecord(
            run_id=row["run_id"],
            sequence=int(row["checkpoint_seq"]),
            workflow_version=row["workflow_version"],
            state_schema_version=row["state_schema_version"],
            snapshot_id=row["snapshot_id"],
            state_json=row["state_json"],
        )
