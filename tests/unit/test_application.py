from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.persistence import SQLiteCaseStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.cli import ingest_file, main
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import CaseRecord, GraphEdge, GraphNode


def _settings(tmp_path: Path, *, token: str | None = None) -> RuntimeSettings:
    return RuntimeSettings(
        runtime_dir=tmp_path,
        database_path=tmp_path / "runtime.db",
        operator_token=token,
    )


def test_composition_root_drives_ingestion_search_and_status(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))
    ingested = application.ingest_document.execute(
        display_name="manual.txt",
        content=b"EGT rise indicates turbine distress",
    )

    hits = application.search_evidence.execute("turbine distress")
    status = application.get_runtime_status.execute()

    assert hits[0].chunk.version_id == ingested.version_id
    assert status.document_count == 1
    assert status.vector_chunks == 1
    assert status.graph_backend == "sqlite_graph"
    assert application.browse_knowledge.execute()[0].active_version_id == ingested.version_id


def test_application_browses_active_graph_and_case_catalog(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    application = bootstrap(settings)
    ingested = application.ingest_document.execute(
        display_name="manual.txt",
        content=b"Compressor stall causes an EGT rise.",
    )
    graph = SQLiteGraphStore(settings.database_path)
    graph.upsert_nodes(
        (
            GraphNode(
                "stall",
                "Compressor stall",
                "fault_mode",
                "Unstable flow",
                "manual",
                ingested.version_id,
            ),
            GraphNode(
                "egt",
                "EGT rise",
                "symptom",
                "Temperature increase",
                "manual",
                ingested.version_id,
            ),
        )
    )
    graph.upsert_edges(
        (GraphEdge("causes", "stall", "egt", "CAUSES", "manual", ingested.version_id),)
    )
    SQLiteCaseStore(settings.database_path).upsert(
        CaseRecord("case-1", 1, "Compressor stall with an EGT rise")
    )

    overview = application.browse_graph.execute()
    cases = application.browse_cases.execute(query="compressor")

    assert {node.node_id for node in overview.nodes} == {"stall", "egt"}
    assert overview.edges[0].relation == "CAUSES"
    assert cases[0].case_id == "case-1"


def test_search_rejects_empty_query_and_empty_corpus(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))

    assert application.search_evidence.execute("compressor") == ()
    try:
        application.search_evidence.execute("  ")
    except ValueError as exc:
        assert str(exc) == "query must not be empty"
    else:  # pragma: no cover - assertion guard
        raise AssertionError("empty query was accepted")


def test_cli_file_adapter_uses_the_same_application_use_case(tmp_path: Path) -> None:
    source = tmp_path / "egt.csv"
    source.write_text("parameter,value\negt,710\n", encoding="utf-8")
    settings = _settings(tmp_path / "runtime")

    result = ingest_file(source, settings=settings)

    assert result.chunk_count == 1
    indexed = bootstrap(settings).search_evidence.execute("egt")[0]
    assert indexed.chunk.version_id == result.version_id


def test_cli_returns_machine_readable_error_for_missing_file(capsys: object) -> None:
    assert main(["definitely-missing.txt"]) == 1
    assert '"error"' in capsys.readouterr().err  # type: ignore[attr-defined]
