from __future__ import annotations

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
    assert operations[1].status == "ok"
    assert operations[1].request_hash
