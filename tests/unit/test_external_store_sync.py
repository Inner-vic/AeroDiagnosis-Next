from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from aerodiagnosis.adapters.persistence.outbox import ExternalStoreOutbox
from aerodiagnosis.adapters.persistence.replicated import (
    ExternalStoreSync,
    ReplicatedGraphStore,
    ReplicatedVectorStore,
)
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.ports import GraphEdge, GraphNode, Neighbor, VectorChunk, VectorMatch


class MemoryVectorReplica:
    def __init__(self, *, fail: bool = False) -> None:
        self.chunks: dict[str, VectorChunk] = {}
        self.fail = fail

    @property
    def backend_name(self) -> str:
        return "memory-vector"

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        if self.fail:
            raise RuntimeError("replica unavailable")
        self.chunks.update((chunk.chunk_id, chunk) for chunk in chunks)
        return len(chunks)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[VectorMatch]:
        del query
        chunks = [
            chunk
            for chunk in self.chunks.values()
            if active_version_ids is None or chunk.version_id in active_version_ids
        ]
        return [VectorMatch(chunk=chunk, score=1.0) for chunk in chunks[:top_k]]

    def delete_document(self, document_id: str) -> int:
        ids = [key for key, value in self.chunks.items() if value.document_id == document_id]
        for chunk_id in ids:
            del self.chunks[chunk_id]
        return len(ids)

    def count(self) -> int:
        return len(self.chunks)


class MemoryGraphReplica:
    def __init__(self) -> None:
        self.nodes: dict[str, GraphNode] = {}
        self.edges: dict[str, GraphEdge] = {}

    @property
    def backend_name(self) -> str:
        return "memory-graph"

    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> int:
        self.nodes.update((node.node_id, node) for node in nodes)
        return len(nodes)

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> int:
        self.edges.update((edge.edge_id, edge) for edge in edges)
        return len(edges)

    def neighbors(
        self,
        node_id: str,
        *,
        max_depth: int = 2,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[Neighbor]:
        del node_id, max_depth, active_version_ids
        return []

    def search_nodes(self, keyword: str, *, limit: int = 20) -> list[GraphNode]:
        return [node for node in self.nodes.values() if keyword in node.name][:limit]

    def snapshot(
        self,
        *,
        active_version_ids: frozenset[str] | None = None,
        limit: int = 300,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        nodes = [
            node
            for node in self.nodes.values()
            if active_version_ids is None or node.version_id in active_version_ids
        ][:limit]
        node_ids = {node.node_id for node in nodes}
        edges = [
            edge
            for edge in self.edges.values()
            if edge.source_id in node_ids and edge.target_id in node_ids
        ][: limit * 2]
        return nodes, edges

    def delete_source(self, source_ref: str) -> tuple[int, int]:
        node_ids = [key for key, node in self.nodes.items() if node.source_ref == source_ref]
        edge_ids = [
            key
            for key, edge in self.edges.items()
            if edge.source_ref == source_ref
            or edge.source_id in node_ids
            or edge.target_id in node_ids
        ]
        for edge_id in edge_ids:
            del self.edges[edge_id]
        for node_id in node_ids:
            del self.nodes[node_id]
        return len(node_ids), len(edge_ids)

    def counts(self) -> tuple[int, int]:
        return len(self.nodes), len(self.edges)


def _chunk() -> VectorChunk:
    return VectorChunk("chunk-1", "document-1", "version-1", "compressor fouling")


def test_vector_changes_are_durable_and_replicated(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    outbox = ExternalStoreOutbox(path)
    primary = SQLiteVectorStore(path, outbox=outbox)
    replica = MemoryVectorReplica()
    sync = ExternalStoreSync(
        database_path=path, outbox=outbox, vector_replica=replica
    )
    store = ReplicatedVectorStore(primary, replica, sync, batch_size=10)

    assert store.upsert((_chunk(),)) == 1
    assert replica.count() == 1
    assert outbox.pending_count("vector") == 0
    assert store.search("compressor")[0].chunk.chunk_id == "chunk-1"
    assert store.delete_document("document-1") == 1
    assert replica.count() == 0


def test_repeated_value_after_intermediate_update_is_not_deduplicated(
    tmp_path: Path,
) -> None:
    path = tmp_path / "runtime.db"
    outbox = ExternalStoreOutbox(path)
    replica = MemoryVectorReplica()
    sync = ExternalStoreSync(
        database_path=path, outbox=outbox, vector_replica=replica
    )
    store = ReplicatedVectorStore(
        SQLiteVectorStore(path, outbox=outbox), replica, sync, batch_size=10
    )

    first = _chunk()
    changed = VectorChunk(
        first.chunk_id,
        first.document_id,
        first.version_id,
        "changed content",
    )
    store.upsert((first,))
    store.upsert((changed,))
    store.upsert((first,))

    assert replica.chunks[first.chunk_id].content == first.content


def test_vector_replica_failure_falls_back_to_local_source(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    outbox = ExternalStoreOutbox(path)
    primary = SQLiteVectorStore(path, outbox=outbox)
    replica = MemoryVectorReplica(fail=True)
    sync = ExternalStoreSync(
        database_path=path, outbox=outbox, vector_replica=replica, max_attempts=3
    )
    store = ReplicatedVectorStore(primary, replica, sync, batch_size=10)

    assert store.upsert((_chunk(),)) == 1
    assert outbox.pending_count("vector") == 1
    assert store.search("compressor")[0].chunk.chunk_id == "chunk-1"


def test_graph_changes_replicate_in_node_then_edge_order(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    outbox = ExternalStoreOutbox(path)
    primary = SQLiteGraphStore(path, outbox=outbox)
    replica = MemoryGraphReplica()
    sync = ExternalStoreSync(database_path=path, outbox=outbox, graph_replica=replica)
    store = ReplicatedGraphStore(primary, replica, sync, batch_size=10)
    nodes = (
        GraphNode("fan", "风扇", "component", "", "manual", "version-1"),
        GraphNode("fault", "腐蚀", "fault", "", "manual", "version-1"),
    )
    edge = GraphEdge("edge-1", "fan", "fault", "has_fault", "manual", "version-1")

    assert store.upsert_nodes(nodes) == 2
    assert store.upsert_edges((edge,)) == 1
    assert replica.counts() == (2, 1)
    snapshot = store.snapshot(active_version_ids=frozenset({"version-1"}))
    assert {node.node_id for node in snapshot[0]} == {"fan", "fault"}
    assert snapshot[1][0].edge_id == "edge-1"
    assert store.search_nodes("风扇")[0].node_id == "fan"
    assert store.neighbors("fan") == []
    assert store.delete_source("manual") == (2, 1)
    assert replica.counts() == (0, 0)


def test_existing_local_records_are_backfilled_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    SQLiteVectorStore(path).upsert((_chunk(),))
    graph = SQLiteGraphStore(path)
    graph.upsert_nodes(
        (
            GraphNode("fan", "风扇", "component", "", "manual", "version-1"),
            GraphNode("fault", "腐蚀", "fault", "", "manual", "version-1"),
        )
    )
    graph.upsert_edges(
        (GraphEdge("edge-1", "fan", "fault", "has_fault", "manual", "version-1"),)
    )
    outbox = ExternalStoreOutbox(path)
    vector_replica = MemoryVectorReplica()
    graph_replica = MemoryGraphReplica()
    sync = ExternalStoreSync(
        database_path=path,
        outbox=outbox,
        vector_replica=vector_replica,
        graph_replica=graph_replica,
    )

    assert sync.enqueue_backfill() == 4
    assert sync.flush(limit=10).applied == 3
    assert vector_replica.count() == 1
    assert graph_replica.counts() == (2, 1)
    assert sync.enqueue_backfill() == 4
    assert sync.flush(limit=10).claimed == 0
