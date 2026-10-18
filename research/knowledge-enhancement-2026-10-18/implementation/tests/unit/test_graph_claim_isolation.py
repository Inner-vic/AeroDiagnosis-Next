# ruff: noqa: RUF001
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from typing import cast

import pytest

from aerodiagnosis.application.fault_graph import FaultGraphReasoner
from aerodiagnosis.application.knowledge_graph_extractor import KnowledgeGraphExtractor
from aerodiagnosis.ports import GraphEdge, GraphNode, GraphStore


class SnapshotStore:
    """Deliberately unfiltered snapshot verifies the reasoner's boundary checks."""

    def __init__(self, nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> None:
        self.nodes = list(nodes)
        self.edges = list(edges)

    def snapshot(self, **kwargs: object) -> tuple[list[GraphNode], list[GraphEdge]]:
        return self.nodes, self.edges


def _reason(chunks: tuple[str, ...]):
    nodes, edges = KnowledgeGraphExtractor().extract(
        source_ref="doc",
        version_id="v",
        chunks=chunks,
        chunk_ids=tuple(f"chunk-{index}" for index in range(len(chunks))),
    )
    paths = FaultGraphReasoner(cast(GraphStore, SnapshotStore(nodes, edges))).reason(
        "EGT",
        active_version_ids=frozenset({"v"}),
        limit=20,
    )
    return [path for path in paths if path.causes]


@pytest.mark.parametrize("split_chunks", [False, True])
def test_checks_do_not_cross_flight_phase_sentence_or_chunk(split_chunks: bool) -> None:
    cruise = "在巡航阶段，压气机污染导致排气温度升高，应进行孔探检查。"
    takeoff = "在起飞阶段，压气机污染导致振动升高，应进行趋势复核。"
    chunks = (cruise, takeoff) if split_chunks else (cruise + "\n" + takeoff,)
    paths = _reason(chunks)
    assert len(paths) == 1
    assert {node.name for node in paths[0].checks} == {"孔探检查"}
    assert all("起飞" not in edge.properties["evidence_text"] for edge in paths[0].evidence_edges)
    assert any(edge.relation == "CHECKED_BY" for edge in paths[0].evidence_edges)


def test_same_symptom_claims_remain_separate_with_individual_proofs() -> None:
    paths = _reason(
        (
            "在巡航阶段，压气机污染导致排气温度升高，应进行孔探检查。\n"
            "在起飞阶段，压气机污染导致排气温度升高，应进行趋势复核。",
        )
    )
    assert len(paths) == 2
    assert [{node.name for node in path.checks} for path in paths] == [
        {"孔探检查"},
        {"趋势复核"},
    ]
    assert len({tuple(edge.edge_id for edge in path.evidence_edges) for path in paths}) == 2
    for path in paths:
        assert len({edge.properties["sentence_index"] for edge in path.evidence_edges}) == 1


def test_mitigation_in_another_sentence_does_not_become_symptom_advice() -> None:
    paths = _reason(("压气机污染导致排气温度升高。\n压气机清洗可去除污染。",))
    assert len(paths) == 1
    assert not paths[0].solutions
    assert not any(edge.relation == "MITIGATED_BY" for edge in paths[0].evidence_edges)


def test_multiple_causal_clauses_do_not_infer_which_inspection_targets_which_symptom() -> None:
    _, edges = KnowledgeGraphExtractor().extract(
        source_ref="doc",
        version_id="v",
        chunks=("压气机污染导致排气温度升高，应进行孔探检查；污染导致振动升高，应进行趋势复核。",),
    )
    assert sum(edge.relation == "CAUSED_BY" for edge in edges) == 2
    assert not any(edge.relation == "CHECKED_BY" for edge in edges)


def test_additional_unlinked_symptom_makes_inspection_target_ambiguous() -> None:
    _, edges = KnowledgeGraphExtractor().extract(
        source_ref="doc",
        version_id="v",
        chunks=("压气机污染导致排气温度升高；另外记录振动升高，应进行孔探检查。",),
    )
    assert any(edge.relation == "CAUSED_BY" for edge in edges)
    assert not any(edge.relation == "CHECKED_BY" for edge in edges)


@pytest.mark.parametrize(
    "bad",
    [
        "negative",
        "deprecated",
        "expired",
        "missing_endpoint",
        "inactive_version",
        "source_mismatch",
        "missing_sentence",
    ],
)
def test_invalid_machine_edges_cannot_contribute_to_path_or_score(bad: str) -> None:
    nodes, edges = KnowledgeGraphExtractor().extract(
        source_ref="doc",
        version_id="v",
        chunks=("压气机污染导致排气温度升高。",),
    )
    cause = next(edge for edge in edges if edge.relation == "CAUSED_BY")
    changes: dict[str, object] = {}
    properties = dict(cause.properties)
    if bad == "negative":
        properties["polarity"] = "negative"
    elif bad == "deprecated":
        properties["deprecated"] = True
    elif bad == "expired":
        properties["valid_to"] = "2000-01-01T00:00:00+00:00"
    elif bad == "missing_endpoint":
        changes["target_id"] = "absent"
    elif bad == "inactive_version":
        changes["version_id"] = "inactive"
    elif bad == "source_mismatch":
        changes["source_ref"] = "another-document"
    else:
        properties.pop("sentence_index")
    invalid = replace(cause, properties=properties, **changes)
    reasoner = FaultGraphReasoner(cast(GraphStore, SnapshotStore(nodes, (invalid,))))
    paths = reasoner.reason("EGT", active_version_ids=frozenset({"v"}))
    assert paths
    assert all(not path.causes and not path.evidence_edges for path in paths)
    assert all(path.score == 0.55 for path in paths)


def test_inactive_snapshot_nodes_are_not_candidates_even_if_adapter_ignores_filter() -> None:
    nodes, edges = KnowledgeGraphExtractor().extract(
        source_ref="doc",
        version_id="old",
        chunks=("压气机污染导致排气温度升高。",),
    )
    reasoner = FaultGraphReasoner(cast(GraphStore, SnapshotStore(nodes, edges)))
    assert reasoner.reason("EGT", active_version_ids=frozenset({"new"})) == []


def test_source_and_version_instances_remain_separate_claim_paths() -> None:
    all_nodes: list[GraphNode] = []
    all_edges: list[GraphEdge] = []
    for source, version in (("doc:a", "v1"), ("doc:b", "v2"), ("doc:a", "v3")):
        nodes, edges = KnowledgeGraphExtractor().extract(
            source_ref=source,
            version_id=version,
            chunks=("压气机污染导致排气温度升高，应进行孔探检查。",),
        )
        all_nodes.extend(nodes)
        all_edges.extend(edges)
    paths = FaultGraphReasoner(cast(GraphStore, SnapshotStore(all_nodes, all_edges))).reason(
        "EGT",
        active_version_ids=frozenset({"v2", "v3"}),
        limit=20,
    )
    assert len(paths) == 2
    assert {path.symptom.version_id for path in paths} == {"v2", "v3"}
    for path in paths:
        assert {edge.source_ref for edge in path.evidence_edges} == {path.symptom.source_ref}
        assert {edge.version_id for edge in path.evidence_edges} == {path.symptom.version_id}
