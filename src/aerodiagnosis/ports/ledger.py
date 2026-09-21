from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class OperationRecord:
    operation_id: str
    run_id: str
    call_id: str
    kind: str
    name: str
    status: str
    request_hash: str
    result_json: str | None
    created_at: str
    updated_at: str


class OperationLedger(Protocol):
    def record_model(
        self,
        *,
        run_id: str,
        task: str,
        payload: dict[str, Any],
        result: dict[str, Any],
        error: str | None = None,
    ) -> OperationRecord: ...

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
    ) -> OperationRecord: ...

    def list_run(self, run_id: str) -> tuple[OperationRecord, ...]: ...
