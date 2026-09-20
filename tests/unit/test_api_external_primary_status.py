from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.api.routers import _external_sync_status
from aerodiagnosis.config import RuntimeSettings


def test_external_primary_status_does_not_report_cdc(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
        vector_backend="chroma_http",
        graph_backend="neo4j",
        external_store_mode="external-primary",
    )

    assert _external_sync_status(settings) == {
        "mode": "direct_external_primary",
        "events": {"applied": 0, "pending": 0, "dead": 0},
    }
