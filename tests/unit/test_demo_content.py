from __future__ import annotations

from pathlib import Path

from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings


def test_demo_content_is_seeded_idempotently_when_enabled(tmp_path: Path) -> None:
    settings = RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
        seed_demo_content=True,
    )

    first = bootstrap(settings)
    second = bootstrap(settings)
    status = second.get_runtime_status.execute()

    assert status.document_count == 1
    assert status.version_count == 1
    assert status.vector_chunks >= 1
    assert status.graph_nodes == 10
    assert status.graph_edges == 10
    assert status.case_count == 4
    assert {case.case_id for case in first.browse_cases.execute()} == {
        "DEMO-CASE-EGT-001",
        "DEMO-CASE-SURGE-002",
        "DEMO-CASE-FAN-003",
        "DEMO-CASE-FUEL-004",
    }
    assert second.search_evidence.execute("知识增强诊断")[0].chunk.document_id == (
        "demo-common-aero-knowledge"
    )
    cases = second.browse_cases.execute()
    assert all("教学" not in case.summary for case in cases)
    assert all("教学" not in str(case.attributes) for case in cases)
    document = second.browse_knowledge.execute()[0]
    assert document.display_name == "航空发动机气路故障通用知识.md"
