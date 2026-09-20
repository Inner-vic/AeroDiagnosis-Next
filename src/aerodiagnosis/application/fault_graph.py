from __future__ import annotations

import re
import unicodedata
from collections import defaultdict
from dataclasses import dataclass

from aerodiagnosis.ports import GraphEdge, GraphNode, GraphStore

_CAUSE_RELATIONS = {"CAUSED_BY", "caused_by", "HAS_CAUSE", "has_cause", "可能关联", "可能诱发"}
_SOLUTION_RELATIONS = {"SOLVED_BY", "solved_by", "建议检查", "建议核验", "修复"}
_COMPONENT_RELATIONS = {"HAS_SYMPTOM", "has_symptom", "发生于", "影响"}
_EQUIPMENT_RELATIONS = {"HAS_COMPONENT", "has_component", "OCCURS_ON", "occurs_on"}
_ROOT_CAUSE_RELATIONS = {"RELATED_TO", "related_to"}


@dataclass(frozen=True, slots=True)
class FaultGraphPath:
    symptom: GraphNode
    equipment: GraphNode | None
    component: GraphNode | None
    causes: tuple[GraphNode, ...]
    solutions: tuple[GraphNode, ...]
    score: float
    hops: int


def _keywords(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).casefold()
    words = re.findall(r"[a-z0-9_]{2,}|[\u3400-\u9fff]{2,}", normalized)
    return tuple(dict.fromkeys((normalized.strip(), *words)))


def _matches(node: GraphNode, keywords: tuple[str, ...]) -> bool:
    haystack = unicodedata.normalize("NFKC", f"{node.name} {node.description}").casefold()
    return any(keyword in haystack for keyword in keywords if keyword)


class FaultGraphReasoner:
    """Traverse typed graph snapshots into explainable symptom-cause-solution paths."""

    def __init__(self, graph_store: GraphStore) -> None:
        self._graph_store = graph_store

    def reason(
        self,
        query: str,
        *,
        active_version_ids: frozenset[str] | None = None,
        limit: int = 10,
    ) -> list[FaultGraphPath]:
        if limit < 1:
            raise ValueError("limit must be positive")
        nodes, edges = self._graph_store.snapshot(
            active_version_ids=active_version_ids,
            limit=2000,
        )
        nodes_by_id = {node.node_id: node for node in nodes}
        outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
        incoming: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            outgoing[edge.source_id].append(edge)
            incoming[edge.target_id].append(edge)

        query_keywords = _keywords(query)
        candidates = [
            node for node in nodes if _matches(node, query_keywords)
        ]
        paths: list[FaultGraphPath] = []

        for symptom in candidates:
            causes = self._linked_nodes(
                symptom,
                outgoing,
                nodes_by_id,
                _CAUSE_RELATIONS,
            )
            solutions: list[GraphNode] = []
            for cause in causes:
                solutions.extend(
                    self._linked_nodes(
                        cause,
                        outgoing,
                        nodes_by_id,
                        _SOLUTION_RELATIONS,
                    )
                )
            components = self._linked_nodes(
                symptom,
                incoming,
                nodes_by_id,
                _COMPONENT_RELATIONS,
                incoming=True,
            )
            equipment: list[GraphNode] = []
            for component in components:
                equipment.extend(
                    self._linked_nodes(
                        component,
                        incoming,
                        nodes_by_id,
                        _EQUIPMENT_RELATIONS,
                        incoming=True,
                    )
                )
            if not equipment:
                equipment = self._linked_nodes(
                    symptom,
                    outgoing,
                    nodes_by_id,
                    _EQUIPMENT_RELATIONS,
                )

            hops = len(causes) + len(solutions) + len(components) + len(equipment)
            edge_confidence = self._path_confidence(
                symptom,
                causes,
                solutions,
                outgoing,
                incoming,
            )
            score = min(1.0, 0.55 + 0.05 * min(hops, 5) + edge_confidence * 0.2)
            paths.append(
                FaultGraphPath(
                    symptom=symptom,
                    equipment=equipment[0] if equipment else None,
                    component=components[0] if components else None,
                    causes=tuple(causes),
                    solutions=tuple(solutions),
                    score=score,
                    hops=hops,
                )
            )

        paths.sort(key=lambda path: (-path.score, path.symptom.node_id))
        return paths[:limit]

    @staticmethod
    def _linked_nodes(
        source: GraphNode,
        adjacency: dict[str, list[GraphEdge]],
        nodes_by_id: dict[str, GraphNode],
        relations: set[str],
        *,
        incoming: bool = False,
    ) -> list[GraphNode]:
        found: list[GraphNode] = []
        for edge in adjacency.get(source.node_id, []):
            other_id = edge.source_id if incoming else edge.target_id
            target = nodes_by_id.get(other_id)
            if target is not None and edge.relation in relations:
                found.append(target)
        return found

    @staticmethod
    def _path_confidence(
        symptom: GraphNode,
        causes: list[GraphNode],
        solutions: list[GraphNode],
        outgoing: dict[str, list[GraphEdge]],
        incoming: dict[str, list[GraphEdge]],
    ) -> float:
        related_ids = {symptom.node_id, *(node.node_id for node in causes + solutions)}
        confidences: list[float] = []
        for edge in [*outgoing[symptom.node_id], *incoming[symptom.node_id]]:
            if edge.source_id in related_ids or edge.target_id in related_ids:
                confidences.append(edge.confidence)
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
