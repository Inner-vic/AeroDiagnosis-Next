# ruff: noqa: RUF001

from __future__ import annotations

from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor


def test_knowledge_graph_extractor_builds_typed_causal_relations() -> None:
    extractor = KnowledgeGraphExtractor()

    nodes, edges = extractor.extract(
        source_ref="document:doc-1",
        version_id="version-1",
        chunks=(
            "压气机喘振可能引起排气温度升高，应检查压气机叶片并进行孔探。",
        ),
    )

    by_kind = {node.kind: node for node in nodes}
    assert "Equipment" in by_kind
    assert "Component" in by_kind
    assert "Symptom" in by_kind
    assert "Cause" in by_kind
    assert "Solution" in by_kind
    assert {edge.relation for edge in edges} >= {
        "HAS_COMPONENT",
        "HAS_SYMPTOM",
        "CAUSED_BY",
        "SOLVED_BY",
    }


def test_knowledge_graph_extractor_is_idempotent() -> None:
    extractor = KnowledgeGraphExtractor()
    chunks = ("风扇振动升高，可能是叶片积垢，需要检查和清洗。",)

    first = extractor.extract(
        source_ref="document:doc-1",
        version_id="version-1",
        chunks=chunks,
    )
    second = extractor.extract(
        source_ref="document:doc-1",
        version_id="version-1",
        chunks=chunks,
    )

    assert first == second
