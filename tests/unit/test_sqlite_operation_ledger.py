from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.persistence.sqlite_operation_ledger import (
    SQLiteOperationLedger,
)
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings


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
    assert operations[1].status == "ok"
    assert operations[1].request_hash


def test_application_exposes_operation_history(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
    )
    application = bootstrap(settings)
    SQLiteOperationLedger(settings.database_path).record_model(
        run_id="run-1",
        task="plan_evidence",
        payload={"question": "EGT"},
        result={"tools": ["search_manual_chunks"]},
    )

    operations = application.browse_operations.execute("run-1")

    assert [operation.kind for operation in operations] == ["model"]
    assert operations[0].name == "plan_evidence"
