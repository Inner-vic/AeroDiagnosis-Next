"""Local-first stores with durable incremental replication to external services."""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from aerodiagnosis.ports import (
    GraphEdge,
    GraphNode,
    GraphStore,
    Neighbor,
    VectorChunk,
    VectorMatch,
    VectorStore,
)

from .outbox import ExternalStoreEvent, ExternalStoreOutbox, OutboxStream
from .sqlite import SQLiteDatabase


@dataclass(frozen=True, slots=True)
class SyncReport:
    claimed: int
    applied: int
    failed: int
    pending: int


class ExternalStoreSync:
    """Apply ordered outbox changes to configured external replicas."""

    def __init__(
        self,
        *,
        database_path: Path,
        outbox: ExternalStoreOutbox,
        vector_replica: VectorStore | None = None,
        graph_replica: GraphStore | None = None,
        max_attempts: int = 12,
    ) -> None:
        self._database = SQLiteDatabase(database_path)
        self._outbox = outbox
        self._vector = vector_replica
        self._graph = graph_replica
        self._max_attempts = max_attempts

    @property
    def streams(self) -> tuple[OutboxStream, ...]:
        streams: list[OutboxStream] = []
        if self._vector is not None:
            streams.append("vector")
        if self._graph is not None:
            streams.append("graph")
        return tuple(streams)

    def enqueue_backfill(self, streams: Iterable[OutboxStream] | None = None) -> int:
        selected = set(streams or self.streams)
        total = 0
        with self._database.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            if "vector" in selected:
                rows = connection.execute(
                    "SELECT * FROM vector_chunks ORDER BY chunk_id"
                ).fetchall()
                for offset in range(0, len(rows), 100):
                    batch = rows[offset : offset + 100]
                    if not batch:
                        continue
                    payload = {
                        "chunks": [
                            {
                                "chunk_id": row["chunk_id"],
                                "document_id": row["document_id"],
                                "version_id": row["version_id"],
                                "content": row["content"],
                                "metadata": json.loads(row["metadata_json"]),
                            }
                            for row in batch
                        ]
                    }
                    ExternalStoreOutbox.enqueue_in_transaction(
                        connection,
                        stream="vector",
                        operation="upsert_chunks",
                        aggregate_id=f"backfill:{offset}:{batch[-1]['chunk_id']}",
                        payload=payload,
                        deduplicate=True,
                    )
                    total += len(batch)
            if "graph" in selected:
                node_rows = connection.execute(
                    "SELECT * FROM graph_nodes ORDER BY node_id"
                ).fetchall()
                for offset in range(0, len(node_rows), 100):
                    batch = node_rows[offset : offset + 100]
                    if not batch:
                        continue
                    ExternalStoreOutbox.enqueue_in_transaction(
                        connection,
                        stream="graph",
                        operation="upsert_nodes",
                        aggregate_id=f"backfill:nodes:{offset}:{batch[-1]['node_id']}",
                        payload={
                            "nodes": [
                                {
                                    "node_id": row["node_id"],
                                    "name": row["name"],
                                    "kind": row["kind"],
                                    "description": row["description"],
                                    "source_ref": row["source_ref"],
                                    "version_id": row["version_id"],
                                    "properties": json.loads(row["properties_json"]),
                                }
                                for row in batch
                            ]
                        },
                        deduplicate=True,
                    )
                    total += len(batch)
                edge_rows = connection.execute(
                    "SELECT * FROM graph_edges ORDER BY edge_id"
                ).fetchall()
                for offset in range(0, len(edge_rows), 100):
                    batch = edge_rows[offset : offset + 100]
                    if not batch:
                        continue
                    ExternalStoreOutbox.enqueue_in_transaction(
                        connection,
                        stream="graph",
                        operation="upsert_edges",
                        aggregate_id=f"backfill:edges:{offset}:{batch[-1]['edge_id']}",
                        payload={
                            "edges": [
                                {
                                    "edge_id": row["edge_id"],
                                    "source_id": row["source_id"],
                                    "target_id": row["target_id"],
                                    "relation": row["relation"],
                                    "source_ref": row["source_ref"],
                                    "version_id": row["version_id"],
                                    "confidence": row["confidence"],
                                    "properties": json.loads(row["properties_json"]),
                                }
                                for row in batch
                            ]
                        },
                        deduplicate=True,
                    )
                    total += len(batch)
            connection.commit()
        return total

    @staticmethod
    def _vector_chunks(payload: Mapping[str, Any]) -> tuple[VectorChunk, ...]:
        return tuple(
            VectorChunk(
                chunk_id=str(item["chunk_id"]),
                document_id=str(item["document_id"]),
                version_id=str(item["version_id"]),
                content=str(item["content"]),
                metadata=dict(item.get("metadata", {})),
            )
            for item in payload["chunks"]
        )

    @staticmethod
    def _graph_nodes(payload: Mapping[str, Any]) -> tuple[GraphNode, ...]:
        return tuple(
            GraphNode(
                node_id=str(item["node_id"]),
                name=str(item["name"]),
                kind=str(item["kind"]),
                description=str(item.get("description", "")),
                source_ref=str(item["source_ref"]),
                version_id=str(item["version_id"]),
                properties=dict(item.get("properties", {})),
            )
            for item in payload["nodes"]
        )

    @staticmethod
    def _graph_edges(payload: Mapping[str, Any]) -> tuple[GraphEdge, ...]:
        return tuple(
            GraphEdge(
                edge_id=str(item["edge_id"]),
                source_id=str(item["source_id"]),
                target_id=str(item["target_id"]),
                relation=str(item["relation"]),
                source_ref=str(item["source_ref"]),
                version_id=str(item["version_id"]),
                confidence=float(item.get("confidence", 1.0)),
                properties=dict(item.get("properties", {})),
            )
            for item in payload["edges"]
        )

    def _apply(self, event: ExternalStoreEvent) -> None:
        if event.stream == "vector" and self._vector is not None:
            if event.operation == "upsert_chunks":
                self._vector.upsert(self._vector_chunks(event.payload))
                return
            if event.operation == "delete_document":
                self._vector.delete_document(str(event.payload["document_id"]))
                return
        if event.stream == "graph" and self._graph is not None:
            if event.operation == "upsert_nodes":
                self._graph.upsert_nodes(self._graph_nodes(event.payload))
                return
            if event.operation == "upsert_edges":
                self._graph.upsert_edges(self._graph_edges(event.payload))
                return
            if event.operation == "delete_source":
                self._graph.delete_source(str(event.payload["source_ref"]))
                return
        raise ValueError(f"unsupported external-store event: {event.stream}/{event.operation}")

    def flush(self, *, limit: int = 50) -> SyncReport:
        events = self._outbox.claim(streams=self.streams, limit=limit)
        applied = 0
        failed = 0
        for event in events:
            try:
                self._apply(event)
            except Exception as exc:
                failed += 1
                self._outbox.mark_failed(event, exc, max_attempts=self._max_attempts)
            else:
                applied += 1
                self._outbox.mark_applied(event.event_id)
        return SyncReport(
            claimed=len(events),
            applied=applied,
            failed=failed,
            pending=sum(self._outbox.pending_count(stream) for stream in self.streams),
        )


class ReplicatedVectorStore:
    """Write to SQLite transactionally and use Chroma when its replica is current."""

    def __init__(
        self,
        primary: VectorStore,
        replica: VectorStore,
        sync: ExternalStoreSync,
        *,
        batch_size: int,
    ) -> None:
        self._primary = primary
        self._replica = replica
        self._sync = sync
        self._batch_size = batch_size

    @property
    def backend_name(self) -> str:
        return "chroma_http_cdc"

    @property
    def embedding_identity(self) -> str:
        return self._primary.embedding_identity

    def _flush(self) -> bool:
        report = self._sync.flush(limit=self._batch_size)
        return report.failed == 0 and report.pending == 0

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        count = self._primary.upsert(chunks)
        self._flush()
        return count

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[VectorMatch]:
        if self._flush():
            try:
                return self._replica.search(
                    query, top_k=top_k, active_version_ids=active_version_ids
                )
            except Exception:
                pass
        return self._primary.search(query, top_k=top_k, active_version_ids=active_version_ids)

    def delete_document(self, document_id: str) -> int:
        count = self._primary.delete_document(document_id)
        self._flush()
        return count

    def count(self) -> int:
        return self._primary.count()


class ReplicatedGraphStore:
    """Write to SQLite transactionally and read Neo4j after its replica catches up."""

    def __init__(
        self,
        primary: GraphStore,
        replica: GraphStore,
        sync: ExternalStoreSync,
        *,
        batch_size: int,
    ) -> None:
        self._primary = primary
        self._replica = replica
        self._sync = sync
        self._batch_size = batch_size

    @property
    def backend_name(self) -> str:
        return "neo4j_cdc"

    def _flush(self) -> bool:
        report = self._sync.flush(limit=self._batch_size)
        return report.failed == 0 and report.pending == 0

    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> int:
        count = self._primary.upsert_nodes(nodes)
        self._flush()
        return count

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> int:
        count = self._primary.upsert_edges(edges)
        self._flush()
        return count

    def neighbors(
        self,
        node_id: str,
        *,
        max_depth: int = 2,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[Neighbor]:
        if self._flush():
            try:
                return self._replica.neighbors(
                    node_id,
                    max_depth=max_depth,
                    active_version_ids=active_version_ids,
                )
            except Exception:
                pass
        return self._primary.neighbors(
            node_id, max_depth=max_depth, active_version_ids=active_version_ids
        )

    def search_nodes(self, keyword: str, *, limit: int = 20) -> list[GraphNode]:
        if self._flush():
            try:
                return self._replica.search_nodes(keyword, limit=limit)
            except Exception:
                pass
        return self._primary.search_nodes(keyword, limit=limit)

    def snapshot(
        self,
        *,
        active_version_ids: frozenset[str] | None = None,
        limit: int = 300,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if self._flush():
            try:
                return self._replica.snapshot(
                    active_version_ids=active_version_ids, limit=limit
                )
            except Exception:
                pass
        return self._primary.snapshot(active_version_ids=active_version_ids, limit=limit)

    def delete_source(self, source_ref: str) -> tuple[int, int]:
        counts = self._primary.delete_source(source_ref)
        self._flush()
        return counts

    def counts(self) -> tuple[int, int]:
        return self._primary.counts()
