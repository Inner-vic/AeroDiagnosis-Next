"""Immutable, deduplicated parent text; never resolved across document versions."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.domain import make_chunk_id
from aerodiagnosis.ports.context import ParentContext


class SQLiteParentContextStore:
    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    def upsert(self, contexts: Sequence[ParentContext]) -> int:
        with self._database.connect() as connection:
            for context in contexts:
                if not context.content.strip() or context.context_id != make_chunk_id(
                    context.version_id, context.locator, context.content
                ):
                    raise ValueError("parent context identity does not match its source")
                payload = (
                    context.context_id,
                    context.document_id,
                    context.version_id,
                    context.content,
                    json.dumps(dict(context.locator), ensure_ascii=False, sort_keys=True),
                )
                existing = connection.execute(
                    "SELECT * FROM knowledge_parent_contexts WHERE context_id=?",
                    (context.context_id,),
                ).fetchone()
                if existing is not None and tuple(existing) != payload:
                    raise ValueError("immutable parent context collision")
                connection.execute(
                    "INSERT OR IGNORE INTO knowledge_parent_contexts VALUES (?,?,?,?,?)", payload
                )
            connection.commit()
        return len(contexts)

    def get(self, context_id: str, *, version_id: str) -> ParentContext | None:
        with self._database.connect() as connection:
            row = connection.execute(
                "SELECT * FROM knowledge_parent_contexts WHERE context_id=? AND version_id=?",
                (context_id, version_id),
            ).fetchone()
        return (
            None
            if row is None
            else ParentContext(
                row["context_id"],
                row["document_id"],
                row["version_id"],
                row["content"],
                json.loads(row["locator_json"]),
            )
        )

    def delete_document(self, document_id: str) -> int:
        with self._database.connect() as connection:
            cursor = connection.execute(
                "DELETE FROM knowledge_parent_contexts WHERE document_id=?", (document_id,)
            )
            connection.commit()
            return cursor.rowcount
