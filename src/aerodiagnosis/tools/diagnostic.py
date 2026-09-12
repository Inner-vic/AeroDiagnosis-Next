"""Manual, graph, case and parameter evidence tools with one typed surface."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from aerodiagnosis.domain import (
    DiagnosisCommand,
    EvidenceItem,
    FusedEvidenceHit,
    HybridRetrievalResult,
    RetrievalContribution,
    RetrievalRouteSummary,
    RetrievalStrategy,
    SourceKind,
    make_evidence_id,
)
from aerodiagnosis.ports import CaseStore, GraphNode, GraphStore, VectorMatch, VectorStore

MANUAL_TOOL = "search_manual_chunks"
GRAPH_TOOL = "traverse_fault_graph"
CASE_TOOL = "find_similar_cases"
PARAMETER_TOOL = "analyze_gas_path_parameters"
HYBRID_TOOL = "hybrid_retrieve_evidence"
HYBRID_ALGORITHM = "weighted_rrf@2"

_ROUTE_WEIGHTS = {"manual": 1.0, "graph": 0.88, "case": 0.94}
_RRF_K = 60


def _hash(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _keywords(value: str) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).lower()
    words = re.findall(r"[a-z0-9_]{2,}|[\u3400-\u9fff]{2,}", normalized)
    return tuple(dict.fromkeys((normalized.strip(), *words)))


class DiagnosticToolset:
    """Routes typed commands to deterministic evidence providers."""

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        graph_store: GraphStore,
        case_store: CaseStore,
    ) -> None:
        self._vector = vector_store
        self._graph = graph_store
        self._cases = case_store

    @property
    def backend_identity(self) -> tuple[str, ...]:
        return (
            self._vector.backend_name,
            self._graph.backend_name,
            self._cases.backend_name,
            HYBRID_ALGORITHM,
            "range_check@1",
        )

    def execute(
        self,
        tool_name: str,
        *,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        if tool_name == MANUAL_TOOL:
            return self._manual(command, query, active_version_ids)
        if tool_name == GRAPH_TOOL:
            return self._graph_evidence(command, query, active_version_ids)
        if tool_name == CASE_TOOL:
            return self._case_evidence(command, query)
        if tool_name == PARAMETER_TOOL:
            return self._parameter_evidence(command)
        if tool_name == HYBRID_TOOL:
            return tuple(
                hit.evidence
                for hit in self.hybrid_search(
                    command=command,
                    query=query,
                    active_version_ids=active_version_ids,
                ).hits
            )
        raise KeyError(f"unknown diagnostic tool: {tool_name}")

    def hybrid_search(
        self,
        *,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
        strategy: RetrievalStrategy = RetrievalStrategy.WEIGHTED_RRF,
        rrf_k: int = _RRF_K,
        route_weights: Mapping[str, float] | None = None,
        enforce_route_coverage: bool = True,
    ) -> HybridRetrievalResult:
        """Rank evidence with one production policy or a controlled experiment policy."""

        cleaned = query.strip()
        if not cleaned:
            raise ValueError("hybrid retrieval query must not be empty")
        if not 1 <= rrf_k <= 1000:
            raise ValueError("rrf_k must be between 1 and 1000")
        weights = dict(_ROUTE_WEIGHTS if route_weights is None else route_weights)
        if set(weights) != set(_ROUTE_WEIGHTS) or any(
            not 0 < value <= 1 for value in weights.values()
        ):
            raise ValueError("route_weights must define manual, graph and case in (0, 1]")
        candidate_limit = min(20, max(command.top_k, command.top_k * 3))
        expanded = command.model_copy(update={"top_k": candidate_limit})
        all_route_items = {
            "manual": self._manual(expanded, cleaned, active_version_ids),
            "graph": self._graph_evidence(expanded, cleaned, active_version_ids),
            "case": self._case_evidence(expanded, cleaned),
        }
        selected_route = {
            RetrievalStrategy.MANUAL_ONLY: "manual",
            RetrievalStrategy.GRAPH_ONLY: "graph",
            RetrievalStrategy.CASE_ONLY: "case",
        }.get(strategy)
        route_items = (
            {selected_route: all_route_items[selected_route]}
            if selected_route is not None
            else all_route_items
        )
        algorithm = f"{strategy.value}@{2 if strategy is RetrievalStrategy.WEIGHTED_RRF else 1}"
        candidates: dict[str, tuple[EvidenceItem, list[RetrievalContribution]]] = {}
        for route, items in route_items.items():
            for route_rank, evidence in enumerate(items, start=1):
                normalized = max(0.0, min(1.0, evidence.score))
                contribution = RetrievalContribution(
                    route=route,
                    route_rank=route_rank,
                    raw_score=evidence.score,
                    normalized_score=normalized,
                    reciprocal_rank_score=(rrf_k + 1) / (rrf_k + route_rank),
                    weight=1.0 if strategy is RetrievalStrategy.RRF else weights[route],
                )
                existing = candidates.get(evidence.evidence_id)
                if existing is None:
                    candidates[evidence.evidence_id] = (evidence, [contribution])
                else:
                    existing[1].append(contribution)

        scored: list[tuple[EvidenceItem, float, tuple[RetrievalContribution, ...]]] = []
        for evidence, contribution_list in candidates.values():
            contributions = tuple(contribution_list)
            if selected_route is not None:
                fused_score = contributions[0].normalized_score
            elif strategy is RetrievalStrategy.NORMALIZED_SCORE:
                fused_score = max(
                    item.weight * item.normalized_score for item in contributions
                )
            else:
                fused_score = min(
                    1.0,
                    sum(item.weight * item.reciprocal_rank_score for item in contributions),
                )
            scored.append((evidence, fused_score, contributions))
        scored.sort(key=lambda item: (-item[1], item[0].evidence_id))

        selected: list[tuple[EvidenceItem, float, tuple[RetrievalContribution, ...], str]] = []
        selected_ids: set[str] = set()
        coverage_enabled = (
            enforce_route_coverage and selected_route is None and command.top_k >= len(route_items)
        )
        if coverage_enabled:
            coverage = []
            for route in route_items:
                best = next(
                    (
                        item
                        for item in scored
                        if any(part.route == route for part in item[2])
                    ),
                    None,
                )
                if best is not None:
                    coverage.append(best)
            coverage.sort(key=lambda item: (-item[1], item[0].evidence_id))
            for evidence, score, contributions in coverage:
                if evidence.evidence_id in selected_ids or len(selected) >= command.top_k:
                    continue
                selected.append((evidence, score, contributions, "route_coverage"))
                selected_ids.add(evidence.evidence_id)
        for evidence, score, contributions in scored:
            if evidence.evidence_id in selected_ids or len(selected) >= command.top_k:
                continue
            selected.append((evidence, score, contributions, "fused_score"))
            selected_ids.add(evidence.evidence_id)

        hits = []
        for final_rank, (evidence, score, contributions, reason) in enumerate(selected, start=1):
            retrieval = {
                "algorithm": algorithm,
                "rrf_k": rrf_k,
                "selection_reason": reason,
                "contributions": [item.model_dump(mode="json") for item in contributions],
            }
            ranked_evidence = evidence.model_copy(
                update={"score": score, "locator": {**evidence.locator, "retrieval": retrieval}}
            )
            hits.append(
                FusedEvidenceHit(
                    rank=final_rank,
                    evidence=ranked_evidence,
                    fused_score=score,
                    selection_reason=reason,
                    contributions=contributions,
                )
            )
        included_by_route = {
            route: sum(
                any(part.route == route for part in hit.contributions) for hit in hits
            )
            for route in all_route_items
        }
        routes = tuple(
            RetrievalRouteSummary(
                route=route,
                candidate_count=len(items),
                included_count=included_by_route[route],
                top_raw_score=max((item.score for item in items), default=None),
            )
            for route, items in all_route_items.items()
        )
        return HybridRetrievalResult(
            query=cleaned,
            algorithm=algorithm,
            top_k=command.top_k,
            rrf_k=rrf_k,
            route_weights=weights,
            route_coverage_enabled=coverage_enabled,
            candidate_count=len(candidates),
            hits=tuple(hits),
            routes=routes,
        )

    def _manual(
        self,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        matches = self._vector.search(
            query,
            top_k=command.top_k,
            active_version_ids=active_version_ids,
        )
        evidence = []
        for match in matches:
            item = self._manual_item(match, command.min_relevance)
            if item is not None:
                evidence.append(item)
        return tuple(evidence)

    @staticmethod
    def _manual_item(match: VectorMatch, min_relevance: float) -> EvidenceItem | None:
        if match.score < min_relevance:
            return None
        metadata: Mapping[str, Any] = match.chunk.metadata
        locator_value = metadata.get(
            "locator",
            {"kind": "chunk", "coordinates": {"chunk_id": match.chunk.chunk_id}},
        )
        if not isinstance(locator_value, Mapping):
            return None
        locator = dict(locator_value)
        evidence_id = metadata.get("evidence_id")
        if not isinstance(evidence_id, str):
            evidence_id = make_evidence_id(
                SourceKind.DOCUMENT_CHUNK,
                match.chunk.chunk_id,
                locator,
            )
        content_hash = metadata.get("content_hash")
        if not isinstance(content_hash, str):
            content_hash = _hash(match.chunk.content)
        try:
            return EvidenceItem(
                evidence_id=evidence_id,
                source_kind=SourceKind.DOCUMENT_CHUNK,
                source_ref=match.chunk.chunk_id,
                document_id=match.chunk.document_id,
                version_id=match.chunk.version_id,
                content_hash=content_hash,
                excerpt=match.chunk.content[:600].strip(),
                locator=locator,
                score=match.score,
            )
        except ValueError:
            return None

    def _graph_evidence(
        self,
        command: DiagnosisCommand,
        query: str,
        active_version_ids: frozenset[str],
    ) -> tuple[EvidenceItem, ...]:
        nodes: dict[str, GraphNode] = {}
        for keyword in _keywords(query):
            for node in self._graph.search_nodes(keyword, limit=command.top_k):
                if node.version_id in active_version_ids:
                    nodes.setdefault(node.node_id, node)
            if len(nodes) >= command.top_k:
                break
        evidence = []
        for node in list(nodes.values())[: command.top_k]:
            paths = self._graph.neighbors(
                node.node_id,
                max_depth=2,
                active_version_ids=active_version_ids,
            )
            if paths:
                path = paths[0]
                excerpt = (
                    f"{path.source.name} --{path.edge.relation}--> {path.target.name}: "
                    f"{path.target.description}"
                )
                source_ref = path.edge.edge_id
                locator = {
                    "kind": "graph_path",
                    "coordinates": {
                        "source": path.source.node_id,
                        "edge": path.edge.edge_id,
                        "target": path.target.node_id,
                        "depth": path.depth,
                    },
                }
            else:
                excerpt = f"{node.name}: {node.description}"
                source_ref = node.node_id
                locator = {"kind": "graph_node", "coordinates": {"node_id": node.node_id}}
            evidence_id = make_evidence_id(
                SourceKind.KNOWLEDGE_GRAPH_PATH,
                source_ref,
                locator,
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=evidence_id,
                    source_kind=SourceKind.KNOWLEDGE_GRAPH_PATH,
                    source_ref=source_ref,
                    document_id=node.source_ref,
                    version_id=node.version_id,
                    content_hash=_hash(excerpt),
                    excerpt=excerpt[:600],
                    locator=locator,
                    score=1.0,
                )
            )
        return tuple(evidence)

    def _case_evidence(
        self,
        command: DiagnosisCommand,
        query: str,
    ) -> tuple[EvidenceItem, ...]:
        evidence = []
        for match in self._cases.search(query, limit=command.top_k):
            locator = {
                "kind": "case_version",
                "coordinates": {"case_id": match.case.case_id, "version": match.case.version},
            }
            source_ref = f"{match.case.case_id}:{match.case.version}"
            evidence.append(
                EvidenceItem(
                    evidence_id=make_evidence_id(SourceKind.CASE, source_ref, locator),
                    source_kind=SourceKind.CASE,
                    source_ref=source_ref,
                    document_id=match.case.case_id,
                    version_id=str(match.case.version),
                    content_hash=_hash(match.case.summary),
                    excerpt=match.case.summary[:600],
                    locator=locator,
                    score=match.score,
                )
            )
        return tuple(evidence)

    @staticmethod
    def _parameter_evidence(command: DiagnosisCommand) -> tuple[EvidenceItem, ...]:
        evidence = []
        for observation in command.parameters:
            if observation.value < observation.expected_min:
                state = "below"
            elif observation.value > observation.expected_max:
                state = "above"
            else:
                state = "within"
            payload = observation.model_dump(mode="json")
            canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
            source_ref = _hash(canonical)
            locator = {
                "kind": "parameter_observation",
                "coordinates": {"name": observation.name},
            }
            excerpt = (
                f"{observation.name}={observation.value:g}{observation.unit} is {state} "
                f"the supplied range [{observation.expected_min:g}, "
                f"{observation.expected_max:g}]{observation.unit}."
            )
            evidence.append(
                EvidenceItem(
                    evidence_id=make_evidence_id(
                        SourceKind.PARAMETER_ANALYSIS,
                        source_ref,
                        locator,
                    ),
                    source_kind=SourceKind.PARAMETER_ANALYSIS,
                    source_ref=source_ref,
                    document_id="user-supplied-parameters",
                    version_id="range_check@1",
                    content_hash=_hash(excerpt),
                    excerpt=excerpt,
                    locator=locator,
                    score=1.0,
                )
            )
        return tuple(evidence)
