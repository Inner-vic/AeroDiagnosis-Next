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
    SourceKind,
    make_evidence_id,
)
from aerodiagnosis.ports import CaseStore, GraphNode, GraphStore, VectorMatch, VectorStore

MANUAL_TOOL = "search_manual_chunks"
GRAPH_TOOL = "traverse_fault_graph"
CASE_TOOL = "find_similar_cases"
PARAMETER_TOOL = "analyze_gas_path_parameters"


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
        raise KeyError(f"unknown diagnostic tool: {tool_name}")

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
