# Agent Operation Ledger Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Persist an audit record for every model call and tool execution in the diagnostic workflow, using the existing SQLite `tool_ledger` table without a schema migration.

**Architecture:** Add an `OperationLedger` port, a `SQLiteOperationLedger` adapter, and inject it into `DiagnosticWorkflow`. Model calls are recorded after structured validation; tool calls are recorded in the retrieval loop.

**Tech Stack:** Python 3.13, Pydantic, SQLite, pytest.

## Global Constraints

- Do not change `LATEST_SCHEMA_VERSION`.
- Do not expose API keys or raw local database contents in ledger payloads.
- Keep operation IDs deterministic for retried identical model calls.
- Preserve all existing diagnostic workflow behavior and fail-closed semantics.

---

## Task 1: Add Operation Ledger Port and Adapter

**Files:**
- Create: `src/aerodiagnosis/ports/ledger.py`
- Modify: `src/aerodiagnosis/ports/__init__.py`
- Create: `src/aerodiagnosis/adapters/persistence/sqlite_operation_ledger.py`
- Test: `tests/unit/test_sqlite_operation_ledger.py`

**Interfaces:**
- Produces: `OperationRecord`, `OperationLedger`, `SQLiteOperationLedger`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_sqlite_operation_ledger.py`:

```python
from pathlib import Path

from aerodiagnosis.adapters.persistence.sqlite_operation_ledger import (
    SQLiteOperationLedger,
)


def test_ledger_records_model_and_tool_operations(tmp_path: Path) -> None:
    ledger = SQLiteOperationLedger(tmp_path / "runtime.db")

    ledger.record_model(
        run_id="run-1",
        task="plan_evidence",
        payload={"question": "EGT high"},
        result={"tools": ["search_manual_chunks"]},
    )
    ledger.record_tool(
        run_id="run-1",
        call_id="tool-1",
        tool_name="search_manual_chunks",
        query="EGT high",
        status="ok",
        evidence_count=1,
    )

    operations = ledger.list_run("run-1")
    assert [item.kind for item in operations] == ["model", "tool"]
    assert operations[0].name == "plan_evidence"
    assert operations[1].name == "search_manual_chunks"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sqlite_operation_ledger.py -v`
Expected: FAIL with import error.

- [ ] **Step 3: Add the port**

Create `src/aerodiagnosis/ports/ledger.py`:

```python
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
```

Export it from `src/aerodiagnosis/ports/__init__.py`.

- [ ] **Step 4: Add the SQLite adapter**

Create `src/aerodiagnosis/adapters/persistence/sqlite_operation_ledger.py`:

```python
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from aerodiagnosis.ports import OperationLedger, OperationRecord

from .sqlite import SQLiteDatabase


class SQLiteOperationLedger:
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
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        operation_id = hashlib.sha256(
            f"model:{run_id}:{task}:{canonical}".encode()
        ).hexdigest()
        result_json = json.dumps(result, ensure_ascii=False, sort_keys=True)
        return self._record(
            operation_id=operation_id,
            run_id=run_id,
            call_id=operation_id,
            name=f"model:{task}",
            kind="model",
            status="error" if error else "ok",
            request_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            result_json=None if error else result_json,
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
        result = {"evidence_count": evidence_count, "error": error}
        return self._record(
            operation_id=operation_id,
            run_id=run_id,
            call_id=call_id,
            name=tool_name,
            kind="tool",
            status=status,
            request_hash=hashlib.sha256(canonical.encode()).hexdigest(),
            result_json=json.dumps(result, ensure_ascii=False, sort_keys=True),
            error=error,
        )

    def list_run(self, run_id: str) -> tuple[OperationRecord, ...]:
        with self._database.connect() as connection:
            rows = connection.execute(
                """
                SELECT * FROM tool_ledger
                WHERE run_id = ? ORDER BY created_at, operation_id
                """,
                (run_id,),
            ).fetchall()
        return tuple(self._record_from_row(row) for row in rows)

    def _record(self, **values: Any) -> OperationRecord:
        ...

    def _record_from_row(self, row: Any) -> OperationRecord:
        ...
```

Implement `_record` with `INSERT ... ON CONFLICT(operation_id) DO NOTHING` and return the existing or inserted row. Implement `_record_from_row` to infer `kind` from the `name` prefix.

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sqlite_operation_ledger.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aerodiagnosis/ports/ledger.py src/aerodiagnosis/ports/__init__.py src/aerodiagnosis/adapters/persistence/sqlite_operation_ledger.py tests/unit/test_sqlite_operation_ledger.py
git commit -m "feat: add operation ledger port and adapter"
```

---

## Task 2: Integrate Ledger into Diagnostic Workflow

**Files:**
- Modify: `src/aerodiagnosis/bootstrap.py`
- Modify: `src/aerodiagnosis/application/diagnostic_workflow.py`
- Test: `tests/unit/test_diagnostic_workflow.py`

**Interfaces:**
- Consumes: `SQLiteOperationLedger`
- Produces: persisted model and tool operations for every `DiagnosticWorkflow` run

- [ ] **Step 1: Write the failing assertion**

Add to an existing workflow test or a new test:

```python
def test_workflow_records_operation_ledger(tmp_path: Path) -> None:
    settings = RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "runtime.db")
    ledger = SQLiteOperationLedger(settings.database_path)
    workflow = DiagnosticWorkflow(
        tools=...,
        language_model=...,
        active_versions=lambda: frozenset(),
        checkpoints=...,
        memory=...,
        ledger=ledger,
    )
    ...
    assert ledger.list_run(report.run_id)
```

Use the existing test helper and a fake model that returns a valid plan, diagnosis, and verification.

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_diagnostic_workflow.py -v`
Expected: FAIL because `DiagnosticWorkflow` does not accept `ledger`.

- [ ] **Step 3: Add ledger injection**

In `bootstrap.py`, import `SQLiteOperationLedger`, create it, and pass `ledger=ledger` to the `DiagnosticWorkflow` factory. In `DiagnosticWorkflow.__init__`, accept `ledger: OperationLedger` and store it.

- [ ] **Step 4: Record model and tool operations**

Change `_model_output` to accept `run_id` and call `self._ledger.record_model` after a successful validation or with `error` on failure. Change `_plan`, `_generate`, and `_verify` to pass `state.run_id`.

In `_retrieve`, call `self._ledger.record_tool` after every tool result or exception.

- [ ] **Step 5: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_diagnostic_workflow.py tests/unit/test_sqlite_operation_ledger.py -v`
Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add src/aerodiagnosis/bootstrap.py src/aerodiagnosis/application/diagnostic_workflow.py tests/unit/test_diagnostic_workflow.py
git commit -m "feat: record agent operations in durable ledger"
```

---

## Self-Review

1. **Spec coverage:** model calls and tool calls are recorded in SQLite.
2. **Placeholder scan:** no TBD or unsupported interfaces.
3. **Type consistency:** `OperationLedger` methods match the workflow calls.
