"""Knowledge graph port without database query-language leakage."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol


@dataclass(frozen=True, slots=True)
class GraphNode:
    node_id: str
    name: str
    kind: str
    description: str
    source_ref: str
    version_id: str
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    relation: str
    source_ref: str
    version_id: str
    confidence: float = 1.0
    properties: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Neighbor:
    source: GraphNode
    edge: GraphEdge
    target: GraphNode
    depth: int


class GraphStore(Protocol):
    @property
    def backend_name(self) -> str: ...

    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> int: ...

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> int: ...

    def neighbors(
        self,
        node_id: str,
        *,
        max_depth: int = 2,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[Neighbor]: ...

    def search_nodes(self, keyword: str, *, limit: int = 20) -> list[GraphNode]: ...

    def snapshot(
        self,
        *,
        active_version_ids: frozenset[str] | None = None,
        limit: int = 300,
    ) -> tuple[list[GraphNode], list[GraphEdge]]: ...

    def delete_source(self, source_ref: str) -> tuple[int, int]: ...

    def counts(self) -> tuple[int, int]: ...
