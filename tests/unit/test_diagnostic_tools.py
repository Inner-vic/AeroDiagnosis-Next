from __future__ import annotations

from pathlib import Path

from aerodiagnosis.adapters.persistence import SQLiteCaseStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.domain import (
    DiagnosisCommand,
    EvidenceItem,
    ParameterObservation,
    RetrievalStrategy,
)
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


def test_toolset_snapshot_identity_includes_vector_embedding_identity(
    tmp_path: Path,
) -> None:
    tools = _tools(tmp_path / "runtime.db")

    assert "sqlite_hashing@256" in tools.backend_identity


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
    cases.upsert(CaseRecord("case-2", 1, "A second compressor stall investigation."))
    assert {item.case_id for item in cases.list_cases()} == {"case-1", "case-2"}
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
    assert len({item.evidence_id for item in all_items}) == 5

    hybrid = tools.hybrid_search(
        command=command,
        query="Compressor stall EGT",
        active_version_ids=active,
    )
    assert hybrid.algorithm == "weighted_rrf@2"
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

    manual_only = tools.hybrid_search(
        command=command,
        query="Compressor stall EGT",
        active_version_ids=active,
        strategy=RetrievalStrategy.MANUAL_ONLY,
    )
    assert manual_only.algorithm == "manual_only@1"
    assert {hit.evidence.source_kind for hit in manual_only.hits} == {"document_chunk"}
    assert not manual_only.route_coverage_enabled

    rrf_k10 = tools.hybrid_search(
        command=command,
        query="Compressor stall EGT",
        active_version_ids=active,
        strategy=RetrievalStrategy.RRF,
        rrf_k=10,
        enforce_route_coverage=False,
    )
    rrf_k90 = tools.hybrid_search(
        command=command,
        query="Compressor stall EGT",
        active_version_ids=active,
        strategy=RetrievalStrategy.RRF,
        rrf_k=90,
        enforce_route_coverage=False,
    )
    assert rrf_k10.rrf_k == 10
    assert rrf_k90.rrf_k == 90
    rank_two_k10 = next(
        part.reciprocal_rank_score
        for hit in rrf_k10.hits
        for part in hit.contributions
        if part.route_rank == 2
    )
    rank_two_k90 = next(
        part.reciprocal_rank_score
        for hit in rrf_k90.hits
        for part in hit.contributions
        if part.route_rank == 2
    )
    assert rank_two_k10 < rank_two_k90


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


def test_graph_tool_uses_canonical_causal_paths(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    graph = SQLiteGraphStore(path)
    version = "v1"
    graph.upsert_nodes(
        (
            GraphNode("engine", "Aero Engine", "Equipment", "Engine", "manual", version),
            GraphNode("compressor", "Compressor", "Component", "Compressor", "manual", version),
            GraphNode("egt", "EGT high", "Symptom", "High exhaust temperature", "manual", version),
            GraphNode("stall", "Compressor stall", "Cause", "Stability loss", "manual", version),
            GraphNode(
                "inspect",
                "Inspect blades",
                "Solution",
                "Borescope inspection",
                "manual",
                version,
            ),
        )
    )
    graph.upsert_edges(
        (
            GraphEdge("e1", "engine", "compressor", "HAS_COMPONENT", "manual", version),
            GraphEdge("e2", "compressor", "egt", "HAS_SYMPTOM", "manual", version),
            GraphEdge("e3", "egt", "stall", "CAUSED_BY", "manual", version, 0.9),
            GraphEdge("e4", "stall", "inspect", "SOLVED_BY", "manual", version, 0.8),
        )
    )
    tools = _tools(path)
    command = DiagnosisCommand(session_id="session", question="Why is EGT high?")

    evidence = tools.execute(
        GRAPH_TOOL,
        command=command,
        query="EGT high",
        active_version_ids=frozenset({version}),
    )

    assert evidence[0].locator["kind"] == "causal_graph_path"
    assert "Inspect blades" in evidence[0].excerpt


def test_external_tools_are_exposed_and_executed(tmp_path: Path) -> None:
    external_evidence = EvidenceItem(
        evidence_id="e" * 64,
        source_kind="external_tool",
        source_ref="external-ping",
        document_id="external",
        version_id="v1",
        content_hash="f" * 64,
        excerpt="External tool evidence",
        locator={"kind": "external", "coordinates": {"tool": "external_ping"}},
        score=1.0,
    )
    tools = DiagnosticToolset(
        vector_store=SQLiteVectorStore(tmp_path / "runtime.db"),
        graph_store=SQLiteGraphStore(tmp_path / "runtime.db"),
        case_store=SQLiteCaseStore(tmp_path / "runtime.db"),
        external_tools={
            "external_ping": lambda _command, _query, _active: (external_evidence,)
        },
    )
    command = DiagnosisCommand(session_id="session", question="External tool")

    assert "external_ping" in tools.available_tools
    assert tools.execute(
        "external_ping",
        command=command,
        query="ping",
        active_version_ids=frozenset(),
    )[0].source_kind == "external_tool"
