from __future__ import annotations

from pathlib import Path

import pytest

from aerodiagnosis.adapters.persistence import SQLiteCheckpointStore


def test_checkpoint_store_appends_and_loads_latest_state(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runtime.db")

    first = store.save(
        run_id="run-1",
        workflow_version="workflow@1",
        state_schema_version="1.0",
        snapshot_id="a" * 64,
        state_json='{"stage":"created"}',
    )
    second = store.save(
        run_id="run-1",
        workflow_version="workflow@1",
        state_schema_version="1.0",
        snapshot_id="a" * 64,
        state_json='{"stage":"completed"}',
    )

    assert first.sequence == 1
    assert second.sequence == 2
    assert store.latest("run-1") == second
    assert store.latest("missing") is None


def test_checkpoint_store_rejects_empty_identity(tmp_path: Path) -> None:
    store = SQLiteCheckpointStore(tmp_path / "runtime.db")

    with pytest.raises(ValueError, match="must not be empty"):
        store.save(
            run_id="",
            workflow_version="workflow@1",
            state_schema_version="1.0",
            snapshot_id="a" * 64,
            state_json="{}",
        )
