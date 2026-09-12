"""SQLite short-term conversation memory."""

from __future__ import annotations

import sqlite3
from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.ports import ConversationMessage, MessageRole

from .sqlite import SQLiteDatabase


class SQLiteConversationMemory:
    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    def create_session(self, session_id: str) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        now = datetime.now(UTC).isoformat()
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO conversation_sessions (session_id, created_at, updated_at)
                VALUES (?, ?, ?)
                ON CONFLICT(session_id) DO NOTHING
                """,
                (session_id, now, now),
            )
            connection.commit()

    def append(
        self,
        *,
        session_id: str,
        message_id: str,
        role: MessageRole,
        content: str,
    ) -> ConversationMessage:
        if not message_id.strip() or not content.strip():
            raise ValueError("message_id and content must not be empty")
        now = datetime.now(UTC).isoformat()
        with self._database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            session = connection.execute(
                "SELECT 1 FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"unknown conversation session: {session_id}")
            existing = connection.execute(
                "SELECT * FROM conversation_messages WHERE message_id = ?",
                (message_id,),
            ).fetchone()
            if existing is not None:
                connection.rollback()
                return self._record(existing)
            row = connection.execute(
                "SELECT max(ordinal) FROM conversation_messages WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            ordinal = int(row[0] or 0) + 1
            connection.execute(
                """
                INSERT INTO conversation_messages (
                    message_id, session_id, ordinal, role, content, created_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (message_id, session_id, ordinal, role, content, now),
            )
            connection.execute(
                "UPDATE conversation_sessions SET updated_at = ? WHERE session_id = ?",
                (now, session_id),
            )
            connection.commit()
        return ConversationMessage(message_id, session_id, ordinal, role, content, now)

    def recent(self, session_id: str, *, limit: int = 20) -> tuple[ConversationMessage, ...]:
        if limit < 1 or limit > 100:
            raise ValueError("memory limit must be between 1 and 100")
        with self._database.connect() as connection:
            session = connection.execute(
                "SELECT 1 FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if session is None:
                raise KeyError(f"unknown conversation session: {session_id}")
            rows = connection.execute(
                """
                SELECT * FROM (
                    SELECT * FROM conversation_messages
                    WHERE session_id = ? ORDER BY ordinal DESC LIMIT ?
                ) ORDER BY ordinal ASC
                """,
                (session_id, limit),
            ).fetchall()
        return tuple(self._record(row) for row in rows)

    def delete_session(self, session_id: str) -> bool:
        with self._database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM conversation_sessions WHERE session_id = ?",
                (session_id,),
            )
            connection.commit()
        return cursor.rowcount > 0

    def counts(self) -> tuple[int, int]:
        with self._database.connect() as connection:
            sessions = int(
                connection.execute("SELECT count(*) FROM conversation_sessions").fetchone()[0]
            )
            messages = int(
                connection.execute("SELECT count(*) FROM conversation_messages").fetchone()[0]
            )
        return sessions, messages

    @staticmethod
    def _record(row: sqlite3.Row) -> ConversationMessage:
        return ConversationMessage(
            message_id=row["message_id"],
            session_id=row["session_id"],
            ordinal=int(row["ordinal"]),
            role=MessageRole(row["role"]),
            content=row["content"],
            created_at=row["created_at"],
        )
