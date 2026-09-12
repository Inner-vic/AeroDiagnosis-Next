"""SQLite knowledge graph adapter used when Neo4j is absent."""

from __future__ import annotations

import json
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.ports import GraphEdge, GraphNode, Neighbor

from .sqlite import SQLiteDatabase


class SQLiteGraphStore:
    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    @property
    def backend_name(self) -> str:
        return "sqlite_graph"

    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> int:
        now = datetime.now(UTC).isoformat()
        records = []
        for node in nodes:
            if not all(
                value.strip()
                for value in (
                    node.node_id,
                    node.name,
                    node.kind,
                    node.source_ref,
                    node.version_id,
                )
            ):
                raise ValueError("graph node identity fields must not be empty")
            records.append(
                (
                    node.node_id,
                    node.name,
                    node.kind,
                    node.description,
                    node.source_ref,
                    node.version_id,
                    json.dumps(dict(node.properties), ensure_ascii=False, sort_keys=True),
                    now,
                )
            )
        if not records:
            return 0
        with self._database.connect() as connection:
            connection.executemany(
                """
                INSERT INTO graph_nodes (
                    node_id, name, kind, description, source_ref, version_id,
                    properties_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(node_id) DO UPDATE SET
                    name=excluded.name,
                    kind=excluded.kind,
                    description=excluded.description,
                    source_ref=excluded.source_ref,
                    version_id=excluded.version_id,
                    properties_json=excluded.properties_json,
                    updated_at=excluded.updated_at
                """,
                records,
            )
            connection.commit()
        return len(records)

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> int:
        now = datetime.now(UTC).isoformat()
        records = []
        for edge in edges:
            if not all(
                value.strip()
                for value in (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.source_ref,
                    edge.version_id,
                )
            ):
                raise ValueError("graph edge identity fields must not be empty")
            if not 0.0 <= edge.confidence <= 1.0:
                raise ValueError("edge confidence must be between zero and one")
            records.append(
                (
                    edge.edge_id,
                    edge.source_id,
                    edge.target_id,
                    edge.relation,
                    edge.source_ref,
                    edge.version_id,
                    edge.confidence,
                    json.dumps(dict(edge.properties), ensure_ascii=False, sort_keys=True),
                    now,
                )
            )
        if not records:
            return 0
        with self._database.connect() as connection:
            connection.executemany(
                """
                INSERT INTO graph_edges (
                    edge_id, source_id, target_id, relation, source_ref, version_id,
                    confidence, properties_json, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(edge_id) DO UPDATE SET
                    source_id=excluded.source_id,
                    target_id=excluded.target_id,
                    relation=excluded.relation,
                    source_ref=excluded.source_ref,
                    version_id=excluded.version_id,
                    confidence=excluded.confidence,
                    properties_json=excluded.properties_json,
                    updated_at=excluded.updated_at
                """,
                records,
            )
            connection.commit()
        return len(records)

    @staticmethod
    def _node(row: object, prefix: str) -> GraphNode:
        return GraphNode(
            node_id=row[f"{prefix}_node_id"],  # type: ignore[index]
            name=row[f"{prefix}_name"],  # type: ignore[index]
            kind=row[f"{prefix}_kind"],  # type: ignore[index]
            description=row[f"{prefix}_description"],  # type: ignore[index]
            source_ref=row[f"{prefix}_source_ref"],  # type: ignore[index]
            version_id=row[f"{prefix}_version_id"],  # type: ignore[index]
            properties=json.loads(row[f"{prefix}_properties_json"]),  # type: ignore[index]
        )

    @staticmethod
    def _edge(row: object) -> GraphEdge:
        return GraphEdge(
            edge_id=row["edge_id"],  # type: ignore[index]
            source_id=row["edge_source_id"],  # type: ignore[index]
            target_id=row["edge_target_id"],  # type: ignore[index]
            relation=row["relation"],  # type: ignore[index]
            source_ref=row["edge_source_ref"],  # type: ignore[index]
            version_id=row["edge_version_id"],  # type: ignore[index]
            confidence=float(row["confidence"]),  # type: ignore[index]
            properties=json.loads(row["edge_properties_json"]),  # type: ignore[index]
        )

    def neighbors(
        self,
        node_id: str,
        *,
        max_depth: int = 2,
        active_version_ids: frozenset[str] | None = None,
    ) -> list[Neighbor]:
        if max_depth < 1 or max_depth > 5:
            raise ValueError("max_depth must be between 1 and 5")
        if active_version_ids is not None and not active_version_ids:
            return []
        seen = {node_id}
        frontier = {node_id}
        output: list[Neighbor] = []
        with self._database.connect() as connection:
            for depth in range(1, max_depth + 1):
                next_frontier: set[str] = set()
                for current_id in sorted(frontier):
                    rows = connection.execute(_NEIGHBOR_QUERY, (current_id,)).fetchall()
                    for row in rows:
                        if active_version_ids is not None and (
                            row["source_version_id"] not in active_version_ids
                            or row["target_version_id"] not in active_version_ids
                            or row["edge_version_id"] not in active_version_ids
                        ):
                            continue
                        target_id = str(row["target_node_id"])
                        if target_id in seen:
                            continue
                        output.append(
                            Neighbor(
                                source=self._node(row, "source"),
                                edge=self._edge(row),
                                target=self._node(row, "target"),
                                depth=depth,
                            )
                        )
                        seen.add(target_id)
                        next_frontier.add(target_id)
                frontier = next_frontier
                if not frontier:
                    break
        return output

    def search_nodes(self, keyword: str, *, limit: int = 20) -> list[GraphNode]:
        if limit < 1:
            raise ValueError("limit must be positive")
        escaped = keyword.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
        pattern = f"%{escaped}%"
        with self._database.connect() as connection:
            rows = connection.execute(
                """
                SELECT
                    node_id AS result_node_id, name AS result_name, kind AS result_kind,
                    description AS result_description, source_ref AS result_source_ref,
                    version_id AS result_version_id, properties_json AS result_properties_json
                FROM graph_nodes
                WHERE name LIKE ? ESCAPE '\\' OR description LIKE ? ESCAPE '\\'
                ORDER BY name, node_id LIMIT ?
                """,
                (pattern, pattern, limit),
            ).fetchall()
        return [self._node(row, "result") for row in rows]

    def snapshot(
        self,
        *,
        active_version_ids: frozenset[str] | None = None,
        limit: int = 300,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if limit < 1 or limit > 2000:
            raise ValueError("graph snapshot limit must be between 1 and 2000")
        if active_version_ids is not None and not active_version_ids:
            return [], []
        parameters: tuple[object, ...] = (limit,)
        where = ""
        if active_version_ids is not None:
            ordered_versions = tuple(sorted(active_version_ids))
            placeholders = ",".join("?" for _ in ordered_versions)
            where = f"WHERE version_id IN ({placeholders})"
            parameters = (*ordered_versions, limit)
        with self._database.connect() as connection:
            node_rows = connection.execute(
                f"""
                SELECT
                    node_id AS result_node_id, name AS result_name, kind AS result_kind,
                    description AS result_description, source_ref AS result_source_ref,
                    version_id AS result_version_id, properties_json AS result_properties_json
                FROM graph_nodes {where} ORDER BY kind, name, node_id LIMIT ?
                """,
                parameters,
            ).fetchall()
            node_ids = tuple(str(row["result_node_id"]) for row in node_rows)
            if not node_ids:
                return [], []
            node_placeholders = ",".join("?" for _ in node_ids)
            edge_rows = connection.execute(
                f"""
                SELECT edge_id, source_id AS edge_source_id, target_id AS edge_target_id,
                       relation, source_ref AS edge_source_ref,
                       version_id AS edge_version_id, confidence,
                       properties_json AS edge_properties_json
                FROM graph_edges
                WHERE source_id IN ({node_placeholders}) AND target_id IN ({node_placeholders})
                ORDER BY edge_id LIMIT ?
                """,
                (*node_ids, *node_ids, limit * 2),
            ).fetchall()
        return [self._node(row, "result") for row in node_rows], [
            self._edge(row) for row in edge_rows
        ]

    def delete_source(self, source_ref: str) -> tuple[int, int]:
        with self._database.connect() as connection:
            edge_cursor = connection.execute(
                """
                DELETE FROM graph_edges
                WHERE source_ref = ? OR source_id IN (
                    SELECT node_id FROM graph_nodes WHERE source_ref = ?
                ) OR target_id IN (
                    SELECT node_id FROM graph_nodes WHERE source_ref = ?
                )
                """,
                (source_ref, source_ref, source_ref),
            )
            node_cursor = connection.execute(
                "DELETE FROM graph_nodes WHERE source_ref = ?", (source_ref,)
            )
            connection.commit()
            return node_cursor.rowcount, edge_cursor.rowcount

    def counts(self) -> tuple[int, int]:
        with self._database.connect() as connection:
            nodes = int(connection.execute("SELECT count(*) FROM graph_nodes").fetchone()[0])
            edges = int(connection.execute("SELECT count(*) FROM graph_edges").fetchone()[0])
        return nodes, edges


_NEIGHBOR_QUERY = """
SELECT
    source.node_id AS source_node_id, source.name AS source_name,
    source.kind AS source_kind, source.description AS source_description,
    source.source_ref AS source_source_ref, source.version_id AS source_version_id,
    source.properties_json AS source_properties_json,
    target.node_id AS target_node_id, target.name AS target_name,
    target.kind AS target_kind, target.description AS target_description,
    target.source_ref AS target_source_ref, target.version_id AS target_version_id,
    target.properties_json AS target_properties_json,
    edge.edge_id, edge.source_id AS edge_source_id, edge.target_id AS edge_target_id,
    edge.relation, edge.source_ref AS edge_source_ref,
    edge.version_id AS edge_version_id, edge.confidence,
    edge.properties_json AS edge_properties_json
FROM graph_edges AS edge
JOIN graph_nodes AS source ON source.node_id = ?
JOIN graph_nodes AS target ON target.node_id = CASE
    WHEN edge.source_id = source.node_id THEN edge.target_id ELSE edge.source_id END
WHERE edge.source_id = source.node_id OR edge.target_id = source.node_id
ORDER BY edge.edge_id, target.node_id
"""
