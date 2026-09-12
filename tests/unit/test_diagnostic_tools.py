from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.persistence import SQLiteCaseStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.domain import DiagnosisCommand, ParameterObservation
from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest
from aerodiagnosis.ports import CaseRecord, GraphEdge, GraphNode
from aerodiagnosis.tools.diagnostic import (
    CASE_TOOL,
    GRAPH_TOOL,
    HYBRID_TOOL,
    MANUAL_TOOL,
    PARAMETER_TOOL,
    DiagnosticToolset,
)


def _tools(path: Path) -> DiagnosticToolset:
    return DiagnosticToolset(
        vector_store=SQLiteVectorStore(path),
        graph_store=SQLiteGraphStore(path),
        case_store=SQLiteCaseStore(path),
    )


def test_four_domain_tools_return_typed_provenance(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    manifest = DocumentManifest(path)
    ingested = DocumentIngestionService(manifest, SQLiteVectorStore(path)).ingest(
        display_name="manual.txt",
        content=b"Compressor stall raises exhaust gas temperature.",
    )
    graph = SQLiteGraphStore(path)
    graph.upsert_nodes(
        [
            GraphNode(
                "compressor",
                "Compressor stall",
                "fault",
                "Unstable compressor flow.",
                "manual",
                ingested.version_id,
            ),
            GraphNode(
                "egt",
                "EGT rise",
                "symptom",
                "Exhaust temperature increases.",
                "manual",
                ingested.version_id,
            ),
        ]
    )
    graph.upsert_edges(
        [
            GraphEdge(
                "causes",
                "compressor",
                "egt",
                "CAUSES",
                "manual",
                ingested.version_id,
            )
        ]
    )
    cases = SQLiteCaseStore(path)
    cases.upsert(CaseRecord("case-1", 1, "Compressor stall case with EGT rise."))
    assert cases.list_cases()[0].case_id == "case-1"
    tools = _tools(path)
    command = DiagnosisCommand(
        session_id="session",
        question="Find a similar compressor case and analyze EGT.",
        min_relevance=0,
        parameters=(
            ParameterObservation(
                name="EGT",
                value=760,
                expected_min=600,
                expected_max=720,
                unit="C",
            ),
        ),
    )
    active = frozenset({ingested.version_id})

    manual = tools.execute(
        MANUAL_TOOL,
        command=command,
        query="compressor stall EGT",
        active_version_ids=active,
    )
    graph_items = tools.execute(
        GRAPH_TOOL,
        command=command,
        query="Compressor stall",
        active_version_ids=active,
    )
    case_items = tools.execute(
        CASE_TOOL,
        command=command,
        query="similar compressor case",
        active_version_ids=active,
    )
    parameter_items = tools.execute(
        PARAMETER_TOOL,
        command=command,
        query="EGT",
        active_version_ids=active,
    )

    assert manual[0].source_kind == "document_chunk"
    assert graph_items[0].source_kind == "knowledge_graph_path"
    assert case_items[0].source_kind == "case"
    assert parameter_items[0].source_kind == "parameter_analysis"
    assert "above" in parameter_items[0].excerpt
    all_items = (*manual, *graph_items, *case_items, *parameter_items)
    assert len({item.evidence_id for item in all_items}) == 4

    hybrid = tools.hybrid_search(
        command=command,
        query="Compressor stall EGT",
        active_version_ids=active,
    )
    assert hybrid.algorithm == "weighted_rrf@1"
    assert {route.route for route in hybrid.routes} == {"manual", "graph", "case"}
    assert all(route.included_count >= 1 for route in hybrid.routes)
    assert {hit.evidence.source_kind for hit in hybrid.hits} >= {
        "document_chunk",
        "knowledge_graph_path",
        "case",
    }
    assert all(hit.evidence.locator["retrieval"] for hit in hybrid.hits)

    fused_items = tools.execute(
        HYBRID_TOOL,
        command=command,
        query="Compressor stall EGT",
        active_version_ids=active,
    )
    assert fused_items == tuple(hit.evidence for hit in hybrid.hits)


def test_toolset_rejects_unknown_tool(tmp_path: Path) -> None:
    command = DiagnosisCommand(session_id="session", question="Unknown tool request")

    try:
        _tools(tmp_path / "runtime.db").execute(
            "arbitrary_tool",
            command=command,
            query="query",
            active_version_ids=frozenset(),
        )
    except KeyError as exc:
        assert "unknown diagnostic tool" in str(exc)
    else:  # pragma: no cover - assertion guard
        raise AssertionError("unknown tool was accepted")
