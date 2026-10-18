# ruff: noqa: RUF001
from __future__ import annotations

import pytest

from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor


def _extract(text: str, *, source: str = "doc:1", version: str = "v1"):
    return KnowledgeGraphExtractor().extract(source_ref=source, version_id=version, chunks=(text,))


def test_graph_preserves_explicit_causality_modality_and_check_semantics() -> None:
    nodes, edges = _extract("压气机喘振可能引起排气温度升高，应检查压气机叶片并进行孔探。")
    assert {node.kind for node in nodes} >= {"Component", "Symptom", "Cause", "Check"}
    assert {edge.relation for edge in edges} >= {"CAUSED_BY", "CHECKED_BY", "HAS_SYMPTOM"}
    assert "SOLVED_BY" not in {edge.relation for edge in edges}
    cause = next(edge for edge in edges if edge.relation == "CAUSED_BY")
    names = {node.node_id: node.name for node in nodes}
    assert names[cause.source_id] == "EGT 异常升高"
    assert names[cause.target_id] == "喘振"
    assert cause.properties["modality"] == "possible"
    assert cause.properties["confidence_kind"] == "heuristic"
    assert cause.properties["evidence_text"].startswith("压气机喘振")


@pytest.mark.parametrize(
    "text",
    [
        "压气机污染、排气温度升高均有记录。",
        "压气机污染不会导致排气温度升高。",
        "Compressor fouling does not cause EGT rise.",
        "排除压气机污染导致排气温度升高。",
        "压气机污染和侵蚀可能导致排气温度升高。",
        "污染导致排气温度升高和振动升高。",
    ],
)
def test_graph_does_not_publish_cooccurrence_negation_or_ambiguous_causality(text: str) -> None:
    _, edges = _extract(text)
    assert not any(edge.relation == "CAUSED_BY" for edge in edges)


def test_graph_reverse_causality_and_conditions() -> None:
    nodes, edges = _extract("在巡航阶段，排气温度升高可能是压气机污染。")
    cause = next(edge for edge in edges if edge.relation == "CAUSED_BY")
    by_id = {node.node_id: node for node in nodes}
    assert by_id[cause.source_id].kind == "Symptom"
    assert by_id[cause.target_id].name == "叶片积垢"
    assert "巡航" in cause.properties["conditions"]


def test_stall_surge_and_english_word_boundaries_are_distinct() -> None:
    nodes, _ = _extract("Compressor surge and stall; infant, turbine.")
    symptom_keys = {node.properties["semantic_key"] for node in nodes if node.kind == "Symptom"}
    assert symptom_keys == {"symptom:surge", "symptom:stall"}
    assert "风扇" not in {node.name for node in nodes}


def test_graph_source_instances_keep_shared_semantic_identity() -> None:
    first = _extract("Compressor surge can cause EGT rise.", source="doc:english", version="en")
    second = _extract("压气机喘振可能引起排气温度升高。", source="doc:chinese", version="cn")
    english = next(
        node for node in first[0] if node.properties["semantic_key"] == "symptom:egt_high"
    )
    chinese = next(
        node for node in second[0] if node.properties["semantic_key"] == "symptom:egt_high"
    )
    assert english.node_id != chinese.node_id
    assert english.properties["semantic_id"] == chinese.properties["semantic_id"]
    assert first == _extract(
        "Compressor surge can cause EGT rise.", source="doc:english", version="en"
    )


def test_graph_claim_spans_and_multiple_sentence_evidence_are_preserved() -> None:
    text = "  压气机污染导致排气温度升高。\n压气机污染可能引起排气温度升高。  "
    _, edges = KnowledgeGraphExtractor().extract(
        source_ref="doc",
        version_id="v",
        chunks=(text,),
        chunk_ids=("chunk-a",),
        chunk_locators=({"kind": "character_range", "coordinates": {"start": 30, "end": 90}},),
    )
    claims = [edge for edge in edges if edge.relation == "CAUSED_BY"]
    assert len(claims) == 2
    assert len({edge.edge_id for edge in claims}) == 2
    for claim in claims:
        props = claim.properties
        assert text[props["sentence_start"] : props["sentence_end"]] == props["evidence_text"]
        assert props["chunk_id"] == "chunk-a"
        assert props["locator"]["kind"] == "character_range"


def test_graph_sqlite_preserves_sources_revisions_and_deletion(tmp_path) -> None:
    store = SQLiteGraphStore(tmp_path / "graph.db")
    for source, version in (("doc:1", "v1"), ("doc:1", "v2"), ("doc:2", "v3")):
        nodes, edges = _extract("压气机污染导致排气温度升高。", source=source, version=version)
        store.upsert_nodes(nodes)
        store.upsert_edges(edges)
    nodes, edges = store.snapshot(active_version_ids=frozenset({"v1"}))
    assert nodes and edges and {node.version_id for node in nodes} == {"v1"}
    assert edges[0].properties["evidence_text"] == "压气机污染导致排气温度升高。"
    store.delete_source("doc:1")
    nodes, edges = store.snapshot()
    assert nodes and edges and {node.source_ref for node in nodes} == {"doc:2"}


def test_graph_extraction_requires_aligned_optional_provenance() -> None:
    with pytest.raises(ValueError, match="align"):
        KnowledgeGraphExtractor().extract(
            source_ref="doc", version_id="v", chunks=("a",), chunk_ids=()
        )
    with pytest.raises(ValueError, match="align"):
        KnowledgeGraphExtractor().extract(
            source_ref="doc", version_id="v", chunks=("a",), chunk_locators=()
        )


def test_graph_explicit_mitigation_is_distinct_from_inspection() -> None:
    _, edges = _extract("压气机清洗可去除叶片积垢。")
    assert "MITIGATED_BY" in {edge.relation for edge in edges}
    assert "SOLVED_BY" not in {edge.relation for edge in edges}


def test_multiple_explicit_clauses_do_not_cross_causal_targets() -> None:
    nodes, edges = _extract("压气机污染导致排气温度升高，侵蚀导致振动升高。")
    names = {node.node_id: node.name for node in nodes}
    pairs = {
        (names[edge.source_id], names[edge.target_id])
        for edge in edges
        if edge.relation == "CAUSED_BY"
    }
    assert pairs == {("EGT 异常升高", "叶片积垢"), ("振动升高", "叶片侵蚀")}


def test_reverse_english_causal_trigger_tolerates_spacing() -> None:
    _, edges = _extract("EGT rise may be caused   by compressor fouling.")
    assert any(edge.relation == "CAUSED_BY" for edge in edges)


def test_skipped_relation_reason_is_recorded_on_mentions() -> None:
    nodes, _ = _extract("压气机污染和排气温度升高同时记录。")
    cause = next(node for node in nodes if node.kind == "Cause")
    assert cause.properties["skipped_relation_reasons"] == ["no_explicit_causal_trigger"]


def test_graph_does_not_infer_mitigation_from_an_unrelated_remove_clause() -> None:
    _, edges = _extract("压气机污染导致排气温度升高，可清洗传感器去除读数噪音。")
    assert not any(edge.relation == "MITIGATED_BY" for edge in edges)


def test_checks_for_another_component_are_not_assigned_to_causal_component() -> None:
    _, edges = _extract("压气机污染导致排气温度升高，应对涡轮进行孔探检查。")
    assert not any(edge.relation == "CHECKED_BY" for edge in edges)
