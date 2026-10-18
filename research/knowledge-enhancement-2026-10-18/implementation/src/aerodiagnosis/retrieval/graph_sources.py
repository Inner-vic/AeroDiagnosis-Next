"""Read-only graph candidates grounded in immutable source children, never synthetic facts."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from aerodiagnosis.application.fault_graph import graph_edge_is_active
from aerodiagnosis.ports import GraphEdge, GraphNode, GraphStore, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.chinese import relevance
from aerodiagnosis.retrieval.concepts import concept_record, guided_edges


class GraphSourceCandidates:
    """Reuse source-scoped concept matching, retaining direction and qualification as provenance."""

    def __init__(
        self,
        graph_store: GraphStore,
        load_chunks: Callable[[Sequence[str]], dict[str, VectorChunk]],
    ) -> None:
        self._graph = graph_store
        self._load = load_chunks

    @property
    def backend_name(self) -> str:
        return self._graph.backend_name + ":graph_source_candidates@1"

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        if not query.strip() or not 1 <= top_k <= 500:
            raise ValueError("Graph candidate query/limit outside bounds")
        # Unscoped graph reads can mix superseded event assertions.
        if not active_version_ids:
            return []
        nodes: list[GraphNode] = []
        edges: list[GraphEdge] = []
        for version in sorted(active_version_ids):
            vn, ve = self._graph.snapshot(active_version_ids=frozenset({version}), limit=2000)
            if len(vn) >= 2000 or len(ve) >= 4000:
                raise ValueError("Graph candidate snapshot may be truncated")
            nodes.extend(n for n in vn if n.version_id == version)
            edges.extend(e for e in ve if e.version_id == version)
        by_node = {n.node_id: n for n in nodes}
        edges = [
            e
            for e in edges
            if graph_edge_is_active(e)
            and e.properties.get("validation_status") == "necessary_source_checks_passed"
            and e.source_id in by_node
            and e.target_id in by_node
            and by_node[e.source_id].version_id == e.version_id
            and by_node[e.target_id].version_id == e.version_id
        ]
        candidates = guided_edges(query, nodes, edges, [concept_record(n) for n in nodes])
        keys = sorted({str(e.properties.get("chunk_id", "")) for e, _, _ in candidates})
        chunks = self._load(keys)
        scored: dict[str, tuple[VectorChunk, float, list[dict[str, Any]]]] = {}
        for edge, signal, concepts in candidates:
            props = edge.properties
            chunk = chunks.get(str(props.get("chunk_id", "")))
            if (
                chunk is None
                or chunk.version_id != edge.version_id
                or chunk.version_id not in active_version_ids
                or edge.source_ref != "document:" + chunk.document_id
            ):
                continue
            coordinates = chunk.metadata.get("locator", {}).get("coordinates", {})
            start, end = props.get("canonical_start"), props.get("canonical_end")
            offset = coordinates.get("start")
            quote = props.get("evidence_text")
            if (
                type(start) is not int
                or type(end) is not int
                or type(offset) is not int
                or not isinstance(quote, str)
                or not quote
                or not offset <= start < end <= offset + len(chunk.content)
                or chunk.content[start - offset : end - offset] != quote
                or (
                    coordinates.get("canonical_sha256")
                    and coordinates["canonical_sha256"] != props.get("canonical_sha256")
                )
            ):
                continue
            lexical = relevance(query, str(chunk.metadata.get("display_content_zh", chunk.content)))
            if lexical <= 0:
                continue
            score = min(1.0, 0.85 * lexical + 0.15 * signal)
            details = {
                "edge_id": edge.edge_id,
                "relation": edge.relation,
                "source_id": edge.source_id,
                "target_id": edge.target_id,
                "certainty": props.get("certainty", "unknown"),
                "qualifiers": props.get("qualifiers", {}),
                "canonical_start": start,
                "canonical_end": end,
                "concept_ids": list(concepts),
            }
            if chunk.chunk_id not in scored:
                scored[chunk.chunk_id] = (chunk, score, [details])
            else:
                old = scored[chunk.chunk_id]
                old[2].append(details)
                scored[chunk.chunk_id] = (chunk, max(score, old[1]), old[2])
        return [
            VectorMatch(
                replace(
                    chunk,
                    metadata={
                        **chunk.metadata,
                        "graph_retrieval": {
                            "policy": "graph_source_candidates@1",
                            "backend": self._graph.backend_name,
                            "assertion_is_not_a_diagnostic_conclusion": True,
                            "edges": sorted(details, key=lambda d: d["edge_id"]),
                        },
                    },
                ),
                score,
            )
            for chunk, score, details in sorted(
                scored.values(), key=lambda r: (-r[1], r[0].chunk_id)
            )[:top_k]
        ]
