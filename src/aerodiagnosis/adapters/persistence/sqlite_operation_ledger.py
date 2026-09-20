from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aerodiagnosis.ports import OperationRecord

from .sqlite import SQLiteDatabase


class SQLiteOperationLedger:
    """Persist model and tool operations in the existing tool_ledger table."""

    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    def record_model(
        self,
        *,
        run_id: str,
        task: str,
        payload: dict[str, Any],
        result: dict[str, Any],
        error: str | None = None,
    ) -> OperationRecord:
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        operation_id = hashlib.sha256(
            f"model:{run_id}:{task}:{canonical}".encode()
        ).hexdigest()
        result_json = None if error else json.dumps(result, ensure_ascii=False, sort_keys=True)
        return self._record(
            operation_id=operation_id,
            run_id=run_id,
            call_id=operation_id,
            name=f"model:{task}",
            status="error" if error else "ok",
            request_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            result_json=result_json,
            error=error,
        )

    def record_tool(
        self,
        *,
        run_id: str,
        call_id: str,
        tool_name: str,
        query: str,
        status: str,
        evidence_count: int,
        error: str | None = None,
    ) -> OperationRecord:
        canonical = json.dumps(
            {"tool_name": tool_name, "query": query},
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        operation_id = hashlib.sha256(
            f"tool:{run_id}:{call_id}:{canonical}".encode()
        ).hexdigest()
        result: dict[str, Any] = {"evidence_count": evidence_count}
        if error is not None:
            result["error"] = error
        return self._record(
            operation_id=operation_id,
            run_id=run_id,
            call_id=call_id,
            name=tool_name,
            status=status,
            request_hash=hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
            result_json=json.dumps(result, ensure_ascii=False, sort_keys=True),
            error=error,
        )

    def list_run(self, run_id: str) -> tuple[OperationRecord, ...]:
        with self._database.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tool_ledger
                WHERE run_id = ?
                ORDER BY created_at, operation_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def _record(
        self,
        *,
        operation_id: str,
        run_id: str,
        call_id: str,
        name: str,
        status: str,
        request_hash: str,
        result_json: str | None,
        error: str | None,
    ) -> OperationRecord:
        now = datetime.now(UTC).isoformat()
        if result_json is None and error is not None:
            result_json = json.dumps({"error": error}, ensure_ascii=False, sort_keys=True)
        with self._database.connect() as connection:
            connection.execute(
                """
                INSERT INTO tool_ledger (
                    operation_id, run_id, call_id, tool_name, status,
                    request_hash, result_json, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(operation_id) DO NOTHING
                """,
                (
                    operation_id,
                    run_id,
                    call_id,
                    name,
                    status,
                    request_hash,
                    result_json,
                    now,
                    now,
                ),
            )
            connection.commit()
            row = connection.execute(
                "SELECT * FROM tool_ledger WHERE operation_id = ?",
                (operation_id,),
            ).fetchone()
        return self._record_from_row(row)

    @staticmethod
    def _record_from_row(row: Any) -> OperationRecord:
        name = str(row["tool_name"])
        return OperationRecord(
            operation_id=str(row["operation_id"]),
            run_id=str(row["run_id"]),
            call_id=str(row["call_id"]),
            kind="model" if name.startswith("model:") else "tool",
            name=name.removeprefix("model:") if name.startswith("model:") else name,
            status=str(row["status"]),
            request_hash=str(row["request_hash"]),
            result_json=(
                str(row["result_json"]) if row["result_json"] is not None else None
            ),
            created_at=str(row["created_at"]),
            updated_at=str(row["updated_at"]),
        )
