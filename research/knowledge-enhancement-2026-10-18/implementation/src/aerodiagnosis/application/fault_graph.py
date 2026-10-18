from __future__ import annotations

import re
import unicodedata
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import UTC, datetime

from aerodiagnosis.ports import GraphEdge, GraphNode, GraphStore

_CAUSE_RELATIONS = {"CAUSED_BY", "caused_by", "HAS_CAUSE", "has_cause", "可能关联", "可能诱发"}
_FORWARD_CAUSE_RELATIONS = {"CAUSES", "CONTRIBUTES_TO"}
_SOLUTION_RELATIONS = {"SOLVED_BY", "solved_by", "MITIGATED_BY", "修复", "REQUIRES_ACTION"}
_CHECK_RELATIONS = {"CHECKED_BY", "checked_by", "建议检查", "建议核验", "CHECK_WITH"}
_COMPONENT_RELATIONS = {"HAS_SYMPTOM", "has_symptom", "发生于", "影响"}
_EQUIPMENT_RELATIONS = {"HAS_COMPONENT", "has_component", "OCCURS_ON", "occurs_on"}
_ROOT_CAUSE_RELATIONS = {"RELATED_TO", "related_to"}


def graph_edge_is_active(edge: GraphEdge) -> bool:
    if edge.properties.get("polarity") == "negative":
        return False
    if edge.properties.get("deprecated") is True:
        return False
    valid_to = edge.properties.get("valid_to")
    if isinstance(valid_to, str):
        try:
            expires_at = datetime.fromisoformat(valid_to.replace("Z", "+00:00"))
            return expires_at.tzinfo is not None and expires_at >= datetime.now(UTC)
        except ValueError:
            return False
    return True


def _claim_context(edge: GraphEdge) -> tuple[str, str, str, int] | None:
    if edge.properties.get("extraction") != "lexical@2":
        return None
    chunk_id = edge.properties.get("chunk_id")
    chunk_index = edge.properties.get("chunk_index")
    sentence_index = edge.properties.get("sentence_index")
    if type(sentence_index) is not int or sentence_index < 0:
        return None
    if isinstance(chunk_id, str) and chunk_id:
        chunk = "id:" + chunk_id
    elif type(chunk_index) is int and chunk_index >= 0:
        chunk = f"index:{chunk_index}"
    else:
        return None
    return edge.source_ref, edge.version_id, chunk, sentence_index


@dataclass(frozen=True, slots=True)
class FaultGraphPath:
    symptom: GraphNode
    equipment: GraphNode | None
    component: GraphNode | None
    causes: tuple[GraphNode, ...]
    solutions: tuple[GraphNode, ...]
    score: float
    hops: int
    checks: tuple[GraphNode, ...] = ()
    evidence_edges: tuple[GraphEdge, ...] = ()


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
        max_hops: int = 3,
    ) -> list[FaultGraphPath]:
        if limit < 1:
            raise ValueError("limit must be positive")
        if not 1 <= max_hops <= 8:
            raise ValueError("max_hops must be between 1 and 8")
        nodes, edges = self._graph_store.snapshot(
            active_version_ids=active_version_ids,
            limit=2000,
        )
        nodes_by_id = {
            node.node_id: node
            for node in nodes
            if active_version_ids is None or node.version_id in active_version_ids
        }
        usable_edges = []
        for edge in edges:
            source = nodes_by_id.get(edge.source_id)
            target = nodes_by_id.get(edge.target_id)
            if source is None or target is None or not graph_edge_is_active(edge):
                continue
            if active_version_ids is not None and edge.version_id not in active_version_ids:
                continue
            if edge.properties.get("extraction") == "lexical@2":
                if _claim_context(edge) is None:
                    continue
                if any(
                    node.source_ref != edge.source_ref or node.version_id != edge.version_id
                    for node in (source, target)
                ):
                    continue
            usable_edges.append(edge)

        legacy_edges = [
            edge for edge in usable_edges if edge.properties.get("extraction") != "lexical@2"
        ]
        claim_edges: dict[tuple[str, str, str, int], list[GraphEdge]] = defaultdict(list)
        for edge in usable_edges:
            context = _claim_context(edge)
            if context is not None:
                claim_edges[context].append(edge)
        candidates = [node for node in nodes_by_id.values() if _matches(node, _keywords(query))]
        paths: list[FaultGraphPath] = []
        for symptom in candidates:
            # Machine claims may share semantic nodes, but never form a sentence-spanning path.
            contexts = {
                context
                for context, group in claim_edges.items()
                if any(
                    edge.source_id == symptom.node_id and edge.relation in _CAUSE_RELATIONS
                    for edge in group
                )
            }
            groups = [claim_edges[context] for context in sorted(contexts)]
            has_legacy_claim = any(
                edge.source_id == symptom.node_id and edge.relation in _CAUSE_RELATIONS
                for edge in legacy_edges
            )
            if has_legacy_claim or not groups:
                groups.append(legacy_edges)
            for group in groups:
                outgoing: dict[str, list[GraphEdge]] = defaultdict(list)
                incoming: dict[str, list[GraphEdge]] = defaultdict(list)
                for edge in group:
                    outgoing[edge.source_id].append(edge)
                    incoming[edge.target_id].append(edge)
                paths.append(
                    self._build_path(
                        symptom, group, outgoing, incoming, nodes_by_id, max_hops=max_hops
                    )
                )

        paths.sort(key=lambda path: (-path.score, path.symptom.node_id))
        return paths[:limit]

    def _build_path(
        self,
        symptom: GraphNode,
        edges: list[GraphEdge],
        outgoing: dict[str, list[GraphEdge]],
        incoming: dict[str, list[GraphEdge]],
        nodes_by_id: dict[str, GraphNode],
        *,
        max_hops: int,
    ) -> FaultGraphPath:
        causes = self._causal_nodes(
            symptom,
            outgoing,
            incoming,
            nodes_by_id,
            max_hops=max_hops,
        )
        solutions = self._linked_nodes(symptom, outgoing, nodes_by_id, _SOLUTION_RELATIONS)
        checks = self._linked_nodes(symptom, outgoing, nodes_by_id, _CHECK_RELATIONS)
        for cause in causes:
            checks.extend(
                self._linked_nodes(
                    cause,
                    outgoing,
                    nodes_by_id,
                    _CHECK_RELATIONS,
                )
            )
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
        selected_ids = {
            symptom.node_id,
            *(node.node_id for node in causes + solutions + checks + components + equipment),
        }
        evidence_edges = tuple(
            edge
            for edge in edges
            if edge.source_id in selected_ids
            and edge.target_id in selected_ids
            and graph_edge_is_active(edge)
        )
        return FaultGraphPath(
            symptom=symptom,
            equipment=equipment[0] if equipment else None,
            component=components[0] if components else None,
            causes=tuple(causes),
            solutions=tuple(solutions),
            score=score,
            hops=hops,
            checks=tuple(checks),
            evidence_edges=evidence_edges,
        )

    @classmethod
    def _causal_nodes(
        cls,
        symptom: GraphNode,
        outgoing: dict[str, list[GraphEdge]],
        incoming: dict[str, list[GraphEdge]],
        nodes_by_id: dict[str, GraphNode],
        *,
        max_hops: int,
    ) -> list[GraphNode]:
        reached: dict[str, tuple[int, float]] = {}
        queue = deque([(symptom.node_id, 0, 1.0)])
        seen = {symptom.node_id}
        while queue:
            current_id, depth, confidence = queue.popleft()
            if depth >= max_hops:
                continue
            cause_edges = [
                edge for edge in outgoing.get(current_id, []) if edge.relation in _CAUSE_RELATIONS
            ]
            cause_edges.extend(
                edge
                for edge in incoming.get(current_id, [])
                if edge.relation in _FORWARD_CAUSE_RELATIONS
            )
            for edge in cause_edges:
                if not graph_edge_is_active(edge):
                    continue
                other_id = edge.target_id if edge.source_id == current_id else edge.source_id
                if other_id in seen or other_id not in nodes_by_id:
                    continue
                seen.add(other_id)
                path_confidence = confidence * edge.confidence
                reached[other_id] = (depth + 1, path_confidence)
                queue.append((other_id, depth + 1, path_confidence))
        ordered = sorted(
            (
                (nodes_by_id[node_id], depth, path_confidence)
                for node_id, (depth, path_confidence) in reached.items()
            ),
            key=lambda item: (item[1], -item[2], item[0].node_id),
        )
        return [node for node, _depth, _confidence in ordered]

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
            if target is not None and edge.relation in relations and graph_edge_is_active(edge):
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
        for node_id in related_ids:
            for edge in [*outgoing.get(node_id, []), *incoming.get(node_id, [])]:
                if edge.relation not in _CAUSE_RELATIONS | _FORWARD_CAUSE_RELATIONS:
                    continue
                if edge.source_id in related_ids and edge.target_id in related_ids:
                    confidences.append(edge.confidence)
            if confidences:
                break
        if not confidences:
            return 0.0
        return sum(confidences) / len(confidences)
