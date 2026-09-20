"""Durable transactional outbox for external vector and graph replicas."""

from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal

from .sqlite import SQLiteDatabase

OutboxStream = Literal["vector", "graph"]


@dataclass(frozen=True, slots=True)
class ExternalStoreEvent:
    event_id: str
    stream: OutboxStream
    operation: str
    aggregate_id: str
    payload: Mapping[str, Any]
    attempts: int


class ExternalStoreOutbox:
    """Record changes with local writes and lease them to one or more workers."""

    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    @staticmethod
    def enqueue_in_transaction(
        connection: sqlite3.Connection,
        *,
        stream: OutboxStream,
        operation: str,
        aggregate_id: str,
        payload: Mapping[str, Any],
        deduplicate: bool = False,
    ) -> str:
        payload_json = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        )
        identity = (
            f"outbox-v1:{stream}:{operation}:{aggregate_id}:{payload_json}"
            if deduplicate
            else f"outbox-v1:{uuid.uuid4().hex}"
        )
        event_id = hashlib.sha256(identity.encode("utf-8")).hexdigest()
        now = datetime.now(UTC).isoformat()
        connection.execute(
            """
            INSERT OR IGNORE INTO external_store_outbox (
                event_id, stream, operation, aggregate_id, payload_json, status,
                attempts, available_at, locked_at, last_error, created_at, updated_at
            ) VALUES (?, ?, ?, ?, ?, 'pending', 0, ?, NULL, NULL, ?, ?)
            """,
            (event_id, stream, operation, aggregate_id, payload_json, now, now, now),
        )
        return event_id

    def claim(
        self,
        *,
        streams: Iterable[OutboxStream],
        limit: int,
        stale_after: timedelta = timedelta(minutes=5),
    ) -> tuple[ExternalStoreEvent, ...]:
        selected_streams = tuple(sorted(set(streams)))
        if not selected_streams or limit < 1:
            return ()
        now = datetime.now(UTC)
        stale_before = (now - stale_after).isoformat()
        now_text = now.isoformat()
        placeholders = ",".join("?" for _ in selected_streams)
        with self._database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            connection.execute(
                """
                UPDATE external_store_outbox
                SET status = 'pending', locked_at = NULL, updated_at = ?
                WHERE status = 'processing' AND locked_at < ?
                """,
                (now_text, stale_before),
            )
            rows = connection.execute(
                f"""
                SELECT * FROM external_store_outbox
                WHERE status = 'pending' AND available_at <= ?
                  AND stream IN ({placeholders})
                ORDER BY created_at, event_id
                LIMIT ?
                """,
                (now_text, *selected_streams, limit),
            ).fetchall()
            event_ids = tuple(str(row["event_id"]) for row in rows)
            if event_ids:
                id_placeholders = ",".join("?" for _ in event_ids)
                connection.execute(
                    f"""
                    UPDATE external_store_outbox
                    SET status = 'processing', attempts = attempts + 1,
                        locked_at = ?, updated_at = ?
                    WHERE event_id IN ({id_placeholders})
                    """,
                    (now_text, now_text, *event_ids),
                )
            connection.commit()
        return tuple(
            ExternalStoreEvent(
                event_id=str(row["event_id"]),
                stream=str(row["stream"]),  # type: ignore[arg-type]
                operation=str(row["operation"]),
                aggregate_id=str(row["aggregate_id"]),
                payload=json.loads(str(row["payload_json"])),
                attempts=int(row["attempts"]) + 1,
            )
            for row in rows
        )

    def mark_applied(self, event_id: str) -> None:
        now = datetime.now(UTC).isoformat()
        with self._database.connect() as connection:
            connection.execute(
                """
                UPDATE external_store_outbox
                SET status = 'applied', locked_at = NULL, last_error = NULL, updated_at = ?
                WHERE event_id = ? AND status = 'processing'
                """,
                (now, event_id),
            )
            connection.commit()

    def mark_failed(
        self,
        event: ExternalStoreEvent,
        error: Exception,
        *,
        max_attempts: int,
    ) -> None:
        now = datetime.now(UTC)
        is_dead = event.attempts >= max_attempts
        delay = min(2 ** min(event.attempts, 8), 300)
        available_at = (now + timedelta(seconds=delay)).isoformat()
        with self._database.connect() as connection:
            connection.execute(
                """
                UPDATE external_store_outbox
                SET status = ?, available_at = ?, locked_at = NULL,
                    last_error = ?, updated_at = ?
                WHERE event_id = ? AND status = 'processing'
                """,
                (
                    "dead" if is_dead else "pending",
                    available_at,
                    f"{type(error).__name__}: {error}"[:1000],
                    now.isoformat(),
                    event.event_id,
                ),
            )
            connection.commit()

    def pending_count(self, stream: OutboxStream | None = None) -> int:
        statement = "SELECT count(*) FROM external_store_outbox WHERE status != 'applied'"
        params: tuple[str, ...] = ()
        if stream is not None:
            statement += " AND stream = ?"
            params = (stream,)
        with self._database.connect() as connection:
            return int(connection.execute(statement, params).fetchone()[0])

    def counts(self) -> dict[str, int]:
        with self._database.connect() as connection:
            rows = connection.execute(
                """
                SELECT status, count(*) AS total
                FROM external_store_outbox GROUP BY status ORDER BY status
                """
            ).fetchall()
        return {str(row["status"]): int(row["total"]) for row in rows}
