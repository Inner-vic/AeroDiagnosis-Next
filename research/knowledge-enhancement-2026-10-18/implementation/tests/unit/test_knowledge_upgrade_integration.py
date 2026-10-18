from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from aerodiagnosis.adapters.api.routers import hybrid_retrieval, search_evidence
from aerodiagnosis.adapters.api.schemas import HybridRetrievalRequest, SearchEvidenceRequest
from aerodiagnosis.adapters.persistence.sqlite_cases import SQLiteCaseStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.application.fault_graph import FaultGraphReasoner
from aerodiagnosis.application.knowledge_applicability import assess_applicability, permits
from aerodiagnosis.bootstrap import Application, bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import DiagnosisCommand, DocumentApplicability, DocumentMetadata
from aerodiagnosis.domain.knowledge_query import KnowledgeQueryContext
from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest, VersionStatus
from aerodiagnosis.ports import CaseRecord, GraphEdge, GraphNode, VectorChunk, VectorMatch
from aerodiagnosis.tools import GRAPH_TOOL, MANUAL_TOOL
from aerodiagnosis.tools.diagnostic import DiagnosticToolset


def metadata(model: str = "teaching-a", phase: str = "cruise") -> DocumentMetadata:
    return DocumentMetadata(
        source_type="synthetic_test",
        applicability=DocumentApplicability(
            scope_status="known",
            engine_models=(model,),
            flight_phases=(phase,),
        ),
    )


def application(tmp_path: Path) -> Application:
    return bootstrap(RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "test.db"))


def test_query_normalization_and_strict_missing_conditions() -> None:
    query = KnowledgeQueryContext(engine_model=" Teaching-\uff21 ", flight_phase=" CRUISE ")
    assert query.engine_model == "teaching-a"
    assert assess_applicability(metadata().model_dump(), query) == "matched"
    with pytest.raises(ValidationError, match="needs an engine model"):
        KnowledgeQueryContext(require_known=True)
    with pytest.raises(ValidationError, match="blank"):
        KnowledgeQueryContext(engine_model="   ")


@pytest.mark.parametrize(
    "scope,expected",
    [
        ({}, "unknown"),
        ({"invalid": "data"}, "unknown"),
        ({"applicability": {"scope_status": "generic"}}, "generic"),
        (metadata().model_dump(), "matched"),
        (metadata("teaching-b").model_dump(), "conflict"),
        (metadata(phase="takeoff").model_dump(), "conflict"),
        ({"applicability": {"scope_status": "known", "engine_models": ["teaching-a"]}}, "unknown"),
    ],
)
def test_partial_conditions_never_establish_full_applicability(scope: dict, expected: str) -> None:
    context = KnowledgeQueryContext(engine_model="teaching-a", flight_phase="cruise")
    status = assess_applicability(scope, context)
    assert status == expected
    assert permits(status, context) is (expected != "conflict")
    strict = context.model_copy(update={"require_known": True})
    assert permits(status, strict) is (expected == "matched")
    assert assess_applicability(scope, KnowledgeQueryContext()) == "not_checked"


def test_prefilter_before_top_k_and_http_round_trip(tmp_path: Path) -> None:
    app = application(tmp_path)
    wrong = app.ingest_document.execute(
        display_name="wrong.txt",
        content=b"EGT rise compressor fouling",
        metadata=metadata("teaching-b"),
    )
    right = app.ingest_document.execute(
        display_name="right.txt",
        content=b"Compressor fouling may cause EGT rise. This is a longer synthetic context.",
        metadata=metadata(),
    )
    raw = app.search_evidence.execute("EGT rise compressor fouling", top_k=1)
    assert raw[0].chunk.document_id == wrong.document_id
    context = KnowledgeQueryContext(
        engine_model="teaching-a", flight_phase="cruise", require_known=True
    )
    found = search_evidence(
        SearchEvidenceRequest(
            query="EGT rise compressor fouling",
            top_k=1,
            knowledge_context=context,
        ),
        app,
    )
    assert len(found) == 1
    assert found[0].document_id == right.document_id
    assert found[0].applicability_status == "matched"
    hybrid = hybrid_retrieval(
        HybridRetrievalRequest(
            query="EGT rise",
            min_relevance=0,
            knowledge_context=context,
        ),
        app,
    )
    assert hybrid.hits
    assert all(hit.evidence.version_id == right.version_id for hit in hybrid.hits)
    assert all(hit.evidence.applicability_status == "matched" for hit in hybrid.hits)
    assert hybrid.knowledge_context == context


def test_graph_and_case_routes_share_scope_policy(tmp_path: Path) -> None:
    app = application(tmp_path)
    for model in ("teaching-a", "teaching-b"):
        app.ingest_document.execute(
            display_name=f"{model}.txt",
            content=b"Compressor fouling may cause EGT rise; should perform borescope.",
            metadata=metadata(model),
        )
    cases = SQLiteCaseStore(tmp_path / "test.db")
    for model in ("teaching-a", "teaching-b"):
        cases.upsert(
            CaseRecord(
                model,
                1,
                "EGT rise due to compressor fouling",
                {
                    "document_metadata": metadata(model).model_dump(mode="json"),
                },
            )
        )
    cases.upsert(CaseRecord("unknown", 1, "EGT rise due to compressor fouling"))
    strict = KnowledgeQueryContext(
        engine_model="teaching-a", flight_phase="cruise", require_known=True
    )
    result = app.hybrid_retrieval.execute(
        "EGT rise", top_k=20, min_relevance=0, knowledge_context=strict
    )
    assert {part.route for hit in result.hits for part in hit.contributions} == {
        "manual",
        "graph",
        "case",
    }
    assert all(hit.evidence.applicability_status == "matched" for hit in result.hits)
    graph_hits = [hit for hit in result.hits if hit.evidence.source_kind == "knowledge_graph_path"]
    assert graph_hits
    edge_provenance = graph_hits[0].evidence.locator["coordinates"]["evidence_edges"]
    assert any(edge["properties"].get("chunk_id") for edge in edge_provenance)
    assert "Suggested inspection" in graph_hits[0].evidence.excerpt
    permissive = app.hybrid_retrieval.execute(
        "EGT rise",
        top_k=20,
        min_relevance=0,
        knowledge_context=KnowledgeQueryContext(engine_model="teaching-a"),
    )
    assert any(hit.evidence.applicability_status == "unknown" for hit in permissive.hits)
    assert not any(hit.evidence.document_id == "teaching-b" for hit in permissive.hits)


def test_source_occurrences_survive_other_source_revision_and_deletion(tmp_path: Path) -> None:
    app = application(tmp_path)
    first = app.ingest_document.execute(
        display_name="a.txt", content=b"Compressor fouling causes EGT rise."
    )
    second = app.ingest_document.execute(
        display_name="b.txt", content=b"Compressor fouling causes EGT rise."
    )
    graph = SQLiteGraphStore(tmp_path / "test.db")
    nodes, edges = graph.snapshot(
        active_version_ids=frozenset({first.version_id, second.version_id})
    )
    assert {edge.version_id for edge in edges if edge.relation == "CAUSED_BY"} == {
        first.version_id,
        second.version_id,
    }
    symptoms = [node for node in nodes if node.kind == "Symptom"]
    assert len({node.node_id for node in symptoms}) == 2
    assert len({node.properties["semantic_id"] for node in symptoms}) == 1
    updated = app.ingest_document.execute(
        display_name="a.txt",
        document_id=first.document_id,
        content=b"Compressor fouling causes EGT rise.",
        metadata=DocumentMetadata(source_revision="R2"),
    )
    assert updated.version_id != first.version_id
    old_nodes, _ = graph.snapshot(active_version_ids=frozenset({first.version_id}))
    assert old_nodes
    graph.delete_source(f"document:{first.document_id}")
    _, remaining = graph.snapshot(active_version_ids=frozenset({second.version_id}))
    assert any(edge.relation == "CAUSED_BY" for edge in remaining)


class FailingGraph(SQLiteGraphStore):
    def upsert_edges(self, edges: object) -> int:
        raise OSError("synthetic graph failure")


def test_graph_failure_does_not_publish_or_replace_old_version(tmp_path: Path) -> None:
    app = application(tmp_path)
    old = app.ingest_document.execute(
        display_name="manual.txt", content=b"Compressor fouling causes EGT rise."
    )
    path = tmp_path / "test.db"
    manifest = DocumentManifest(path)
    ingestion = DocumentIngestionService(manifest, SQLiteVectorStore(path), FailingGraph(path))
    with pytest.raises(OSError, match="synthetic graph failure"):
        ingestion.ingest(
            display_name="manual.txt",
            document_id=old.document_id,
            content=b"Compressor erosion may cause EGT rise.",
        )
    assert manifest.active_version_ids() == frozenset({old.version_id})
    records = manifest.list_documents()[0].versions
    assert len(records) == 2
    assert sum(record.status == VersionStatus.FAILED for record in records) == 1
    visible = app.hybrid_retrieval.execute("EGT rise", min_relevance=0)
    assert all(hit.evidence.version_id == old.version_id for hit in visible.hits)


def test_negation_mentions_cannot_become_positive_graph_evidence(tmp_path: Path) -> None:
    app = application(tmp_path)
    result = app.ingest_document.execute(
        display_name="negative.txt",
        content=b"Compressor fouling does not cause EGT rise.",
    )
    tools = DiagnosticToolset(
        vector_store=SQLiteVectorStore(tmp_path / "test.db"),
        graph_store=SQLiteGraphStore(tmp_path / "test.db"),
        case_store=SQLiteCaseStore(tmp_path / "test.db"),
    )
    assert (
        tools.execute(
            GRAPH_TOOL,
            command=DiagnosisCommand(session_id="test", question="EGT rise"),
            query="EGT rise",
            active_version_ids=frozenset({result.version_id}),
        )
        == ()
    )


def test_inspection_is_separate_from_mitigation_and_cause_direction(tmp_path: Path) -> None:
    app = application(tmp_path)
    result = app.ingest_document.execute(
        display_name="check.txt",
        content=b"Compressor fouling may cause EGT rise; should perform borescope.",
    )
    graph = SQLiteGraphStore(tmp_path / "test.db")
    path = FaultGraphReasoner(graph).reason(
        "EGT rise", active_version_ids=frozenset({result.version_id})
    )[0]
    assert path.checks and not path.solutions
    assert path.causes
    causes = FaultGraphReasoner(graph).reason(
        "fouling", active_version_ids=frozenset({result.version_id})
    )
    assert all(not item.causes for item in causes)


def test_document_plugin_cannot_override_manifest_or_snapshot(tmp_path: Path) -> None:
    path = tmp_path / "test.db"
    raw = DiagnosticToolset._manual_item(
        VectorMatch(
            VectorChunk(
                "chunk",
                "doc",
                "v1",
                "EGT rise",
                {
                    "document_metadata": metadata().model_dump(mode="json"),
                },
            ),
            1.0,
        ),
        0,
    )
    assert raw is not None
    tools = DiagnosticToolset(
        vector_store=SQLiteVectorStore(path),
        graph_store=SQLiteGraphStore(path),
        case_store=SQLiteCaseStore(path),
        version_metadata=lambda _: metadata("teaching-b").model_dump(mode="json"),
        external_tools={"plugin_search": lambda *_: (raw,)},
    )
    command = DiagnosisCommand(
        session_id="test",
        question="EGT rise",
        knowledge_context=KnowledgeQueryContext(
            engine_model="teaching-a", flight_phase="cruise", require_known=True
        ),
    )
    assert (
        tools.execute(
            "plugin_search", command=command, query="EGT rise", active_version_ids=frozenset({"v1"})
        )
        == ()
    )
    assert (
        tools.execute(
            "plugin_search",
            command=command.model_copy(update={"knowledge_context": KnowledgeQueryContext()}),
            query="EGT rise",
            active_version_ids=frozenset(),
        )
        == ()
    )


def test_full_structured_evidence_keeps_late_condition(tmp_path: Path) -> None:
    app = application(tmp_path)
    text = "EGT rise. " + "Synthetic explanatory text. " * 24 + "Only during cruise."
    assert 600 < len(text) < 800
    result = app.ingest_document.execute(display_name="late.txt", content=text.encode())
    evidence = app.query_diagnostic_tool.execute(MANUAL_TOOL, query=text)
    assert evidence
    assert evidence[0].version_id == result.version_id
    assert evidence[0].excerpt == text
    assert "Only during cruise." in evidence[0].excerpt
    oversized = DiagnosticToolset._manual_item(
        VectorMatch(VectorChunk("large", "doc", "v1", "x" * 8001), 1),
        0,
    )
    assert oversized is None


def test_incoming_graph_preserves_direction_and_provenance(
    tmp_path: Path,
) -> None:
    path = tmp_path / "test.db"
    graph = SQLiteGraphStore(path)
    graph.upsert_nodes(
        [
            GraphNode("symptom", "EGT rise", "Symptom", "Temperature rise.", "doc", "v1"),
            GraphNode("cause", "fouling", "Cause", "Deposit.", "doc", "v1"),
        ]
    )
    graph.upsert_edges(
        [
            GraphEdge(
                "edge",
                "symptom",
                "cause",
                "CAUSED_BY",
                "doc",
                "v1",
                properties={
                    "conditions": "Only during cruise",
                    "evidence_text": "fouling causes EGT",
                },
            )
        ]
    )
    tools = DiagnosticToolset(
        vector_store=SQLiteVectorStore(path), graph_store=graph, case_store=SQLiteCaseStore(path)
    )
    evidence = tools.execute(
        GRAPH_TOOL,
        command=DiagnosisCommand(session_id="test", question="fouling"),
        query="fouling",
        active_version_ids=frozenset({"v1"}),
    )
    assert evidence[0].excerpt.startswith("EGT rise --CAUSED_BY--> fouling")
    coordinates = evidence[0].locator["coordinates"]
    assert coordinates["source"] == "symptom"
    assert coordinates["target"] == "cause"
    assert coordinates["properties"]["conditions"] == "Only during cruise"


def test_source_revision_is_filtered_before_http_search_top_k(tmp_path: Path) -> None:
    app = application(tmp_path)
    base = metadata()
    app.ingest_document.execute(
        display_name="old.txt",
        content=b"EGT rise compressor fouling",
        metadata=base.model_copy(update={"source_revision": "R1"}),
    )
    right = app.ingest_document.execute(
        display_name="new.txt",
        content=b"Compressor fouling may cause EGT rise.",
        metadata=base.model_copy(update={"source_revision": "R2"}),
    )
    request = SearchEvidenceRequest.model_validate(
        {
            "query": "EGT rise compressor fouling",
            "top_k": 1,
            "knowledge_context": {
                "engine_model": "teaching-a",
                "flight_phase": "cruise",
                "source_revision": " r2 ",
                "require_known": True,
            },
        }
    )
    rows = search_evidence(request, app)
    assert len(rows) == 1
    assert rows[0].document_id == right.document_id
    assert rows[0].document_metadata.source_revision == "R2"
    assert rows[0].applicability_status == "matched"
