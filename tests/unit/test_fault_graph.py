from __future__ import annotations

from collections.abc import Sequence

from aerodiagnosis.application.fault_graph import FaultGraphReasoner
from aerodiagnosis.ports import GraphEdge, GraphNode


class MemoryGraphStore:
    def __init__(self, nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]) -> None:
        self.nodes = tuple(nodes)
        self.edges = tuple(edges)

    @property
    def backend_name(self) -> str:
        return "memory_graph"

    def snapshot(
        self,
        *,
        active_version_ids: frozenset[str] | None = None,
        limit: int = 300,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        del active_version_ids, limit
        return list(self.nodes), list(self.edges)


def test_fault_graph_reasoner_extracts_complete_causal_path() -> None:
    equipment = GraphNode("engine", "Engine", "Equipment", "Aero engine", "manual", "v1")
    compressor = GraphNode(
        "compressor", "Compressor", "Component", "Compressor section", "manual", "v1"
    )
    symptom = GraphNode(
        "egt-high", "EGT high", "Symptom", "Exhaust temperature is elevated", "manual", "v1"
    )
    cause = GraphNode(
        "compressor-stall",
        "Compressor stall",
        "Cause",
        "Compressor operation crosses the stability boundary",
        "manual",
        "v1",
    )
    solution = GraphNode(
        "inspect-blades",
        "Inspect compressor blades",
        "Solution",
        "Perform a borescope inspection",
        "manual",
        "v1",
    )
    store = MemoryGraphStore(
        (equipment, compressor, symptom, cause, solution),
        (
            GraphEdge("e1", equipment.node_id, compressor.node_id, "HAS_COMPONENT", "manual", "v1"),
            GraphEdge("e2", compressor.node_id, symptom.node_id, "HAS_SYMPTOM", "manual", "v1"),
            GraphEdge("e3", symptom.node_id, cause.node_id, "CAUSED_BY", "manual", "v1", 0.9),
            GraphEdge("e4", cause.node_id, solution.node_id, "SOLVED_BY", "manual", "v1", 0.8),
        ),
    )

    paths = FaultGraphReasoner(store).reason("EGT high", active_version_ids=frozenset({"v1"}))

    assert paths
    assert paths[0].symptom.node_id == "egt-high"
    assert paths[0].equipment is not None
    assert paths[0].equipment.node_id == "engine"
    assert paths[0].component is not None
    assert paths[0].component.node_id == "compressor"
    assert [cause.node_id for cause in paths[0].causes] == ["compressor-stall"]
    assert [solution.node_id for solution in paths[0].solutions] == ["inspect-blades"]
