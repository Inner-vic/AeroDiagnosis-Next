"""Bounded source-backed entity resolution and directed paths, without destructive merges."""

from __future__ import annotations

import hashlib
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from aerodiagnosis.application.fault_graph import graph_edge_is_active
from aerodiagnosis.ports import GraphEdge, GraphNode, GraphStore, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.chinese import normalize_query, relevance


def _quote_valid(chunk: VectorChunk, props: Any) -> bool:
    coordinates = chunk.metadata.get("locator", {}).get("coordinates", {})
    start, end, quote = (
        props.get(k) for k in ("canonical_start", "canonical_end", "evidence_text")
    )
    offset = coordinates.get("start")
    return bool(
        type(start) is int
        and type(end) is int
        and type(offset) is int
        and isinstance(quote, str)
        and quote
        and offset <= start < end <= offset + len(chunk.content)
        and chunk.content[start - offset : end - offset] == quote
        and coordinates.get("canonical_sha256")
        and coordinates["canonical_sha256"] == props.get("canonical_sha256")
    )


def entity_identity(node: GraphNode, chunks: dict[str, VectorChunk]) -> str:
    """Unknown roles and unsupported aliases stay distinct; labels alone never merge objects."""
    proof = node.properties.get("entity_link", {})
    chunk = chunks.get(str(proof.get("chunk_id", "")))
    role, surface = proof.get("role"), proof.get("surface")
    if (
        proof.get("review_status") != "source_verified_nonexpert"
        or not isinstance(role, str)
        or role in {"", "unknown"}
        or not isinstance(surface, str)
        or not surface.strip()
        or chunk is None
        or chunk.version_id != node.version_id
        or chunk.document_id != proof.get("document_id")
        or node.source_ref != "document:" + chunk.document_id
        or proof.get("kind") != node.kind
        or not _quote_valid(chunk, proof)
        or surface.casefold() not in proof["evidence_text"].casefold()
    ):
        return "mention:" + node.node_id
    # Only exact source-surface recurrence under the SAME explicit role is resolved.
    # Different aliases need an explicit source alias assertion, not translation similarity.
    key = [node.version_id, node.source_ref, node.kind, role, normalize_query(surface).strip()]
    return "entity:" + hashlib.sha256(repr(key).encode()).hexdigest()


class GraphPathCandidates:
    """Candidate discovery with directed 1--3 assertion paths; not a diagnosis engine.

    Every returned object is an original source chunk. Edge uncertainty, direction and
    polarity remain in provenance. No gold type, hop count, answer or support IDs enter search.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        load_chunks: Callable[[Sequence[str]], dict[str, VectorChunk]],
        *,
        max_hops: int = 3,
        max_visits: int = 4000,
        max_fanout: int = 24,
    ) -> None:
        if not 1 <= max_hops <= 3 or max_visits < 1 or max_fanout < 1:
            raise ValueError("Invalid bounded graph path policy")
        self._graph, self._load = graph_store, load_chunks
        self.max_hops, self.max_visits, self.max_fanout = max_hops, max_visits, max_fanout
        self.last_trace: dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return self._graph.backend_name + ":source_paths@1"

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        if not query.strip() or not 1 <= top_k <= 500:
            raise ValueError("Invalid graph path query/limit")
        self.last_trace = {"visits": 0, "paths": 0, "budget_exhausted": False}
        if not active_version_ids:
            return []
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for version in sorted(active_version_ids):
            vn, ve = self._graph.snapshot(active_version_ids=frozenset({version}), limit=2000)
            if len(vn) >= 2000 or len(ve) >= 4000:
                raise ValueError("Truncated graph snapshot")
            nodes.extend(n for n in vn if n.version_id == version)
            edges.extend(e for e in ve if e.version_id == version)
        keys = {str(e.properties.get("chunk_id", "")) for e in edges}
        keys.update(
            str(proof.get("chunk_id", ""))
            for edge in edges
            for proof in edge.properties.get("supporting_evidence", [])
        )
        keys.update(str(n.properties.get("entity_link", {}).get("chunk_id", "")) for n in nodes)
        chunks = self._load(sorted(keys - {""}))
        identities = {n.node_id: entity_identity(n, chunks) for n in nodes}
        node_map = {n.node_id: n for n in nodes}
        labels: dict[str, set[str]] = defaultdict(set)
        for node in nodes:
            labels[identities[node.node_id]].add(node.name)
        adjacency: dict[str, list[GraphEdge]] = defaultdict(list)
        for edge in edges:
            p = edge.properties
            chunk = chunks.get(str(p.get("chunk_id", "")))
            if (
                not graph_edge_is_active(edge)
                or p.get("validation_status") != "necessary_source_checks_passed"
                or p.get("polarity", "positive") != "positive"
                or edge.source_id not in node_map
                or edge.target_id not in node_map
                or chunk is None
                or chunk.version_id != edge.version_id
                or edge.source_ref != "document:" + chunk.document_id
                or any(
                    node_map[i].version_id != edge.version_id
                    for i in (edge.source_id, edge.target_id)
                )
                or not _quote_valid(chunk, p)
                or any(
                    (extra := chunks.get(str(proof.get("chunk_id", "")))) is None
                    or extra.version_id != edge.version_id
                    or extra.document_id != chunk.document_id
                    or not _quote_valid(extra, proof)
                    for proof in p.get("supporting_evidence", [])
                )
            ):
                continue
            adjacency[identities[edge.source_id]].append(edge)
        seeds = sorted(
            ((key, relevance(query, " ".join(value))) for key, value in labels.items()),
            key=lambda x: (-x[1], x[0]),
        )
        scored: dict[str, tuple[VectorChunk, float, list[dict[str, Any]]]] = {}
        visits = 0
        paths = 0
        exhausted = False
        for seed, signal in seeds[:32]:
            if signal <= 0:
                continue
            stack: list[tuple[str, tuple[GraphEdge, ...], frozenset[str]]] = [
                (seed, (), frozenset({seed}))
            ]
            while stack:
                current, prior, seen = stack.pop()
                outgoing = sorted(
                    adjacency.get(current, []),
                    key=lambda e: (-relevance(query, node_map[e.target_id].name), e.edge_id),
                )
                for edge in outgoing[: self.max_fanout]:
                    if visits >= self.max_visits:
                        exhausted = True
                        break
                    visits += 1
                    target = identities[edge.target_id]
                    if target in seen:
                        continue
                    path = (*prior, edge)
                    paths += 1
                    names = [node_map[path[0].source_id].name] + [
                        node_map[e.target_id].name for e in path
                    ]
                    text_signal = max(
                        relevance(
                            query,
                            str(
                                chunks[str(e.properties["chunk_id"])].metadata.get(
                                    "display_content_zh",
                                    chunks[str(e.properties["chunk_id"])].content,
                                )
                            ),
                        )
                        for e in path
                    )
                    score = 0.75 * relevance(query, " ".join(names)) + 0.25 * text_signal
                    score /= 1 + 0.04 * (len(path) - 1)
                    detail: dict[str, Any] = {
                        "edge_ids": [e.edge_id for e in path],
                        "entities": names,
                        "hops": len(path),
                        "relations": [e.relation for e in path],
                        "certainty": [e.properties.get("certainty", "unknown") for e in path],
                        "qualifiers": [e.properties.get("qualifiers", {}) for e in path],
                        "source_chunks": list(
                            dict.fromkeys(
                                str(proof["chunk_id"])
                                for e in path
                                for proof in [
                                    e.properties,
                                    *e.properties.get("supporting_evidence", []),
                                ]
                            )
                        ),
                        "direction": "forward",
                        "not_a_composed_causal_conclusion": True,
                    }
                    for position, chunk_id in enumerate(detail["source_chunks"]):
                        chunk = chunks[chunk_id]
                        candidate_score = score - position * 1e-6
                        if chunk_id not in scored:
                            scored[chunk_id] = (chunk, candidate_score, [detail])
                        else:
                            old = scored[chunk_id]
                            old[2].append(detail)
                            scored[chunk_id] = (chunk, max(old[1], candidate_score), old[2])
                    if len(path) < self.max_hops:
                        stack.append((target, path, seen | {target}))
                if exhausted:
                    break
            if exhausted:
                break
        self.last_trace = {
            "visits": visits,
            "paths": paths,
            "budget_exhausted": exhausted,
            "resolved_entities": len(labels),
            "mentions": len(nodes),
        }
        return [
            VectorMatch(
                replace(
                    chunk,
                    metadata={
                        **chunk.metadata,
                        "graph_retrieval": {
                            "policy": "source_paths@1",
                            "trace": dict(self.last_trace),
                            "paths": sorted(details, key=lambda d: (-d["hops"], d["edge_ids"]))[
                                :10
                            ],
                            "assertion_is_not_a_diagnostic_conclusion": True,
                        },
                    },
                ),
                score,
            )
            for chunk, score, details in sorted(
                scored.values(), key=lambda x: (-x[1], x[0].chunk_id)
            )[:top_k]
        ]
