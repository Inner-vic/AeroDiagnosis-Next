"""Neo4j implementation of the knowledge-graph port."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from typing import Any

from aerodiagnosis.ports import GraphEdge, GraphNode, Neighbor


class Neo4jGraphStore:
    """Persist typed v3 graph records without exposing Cypher to callers."""

    def __init__(self, *, uri: str, user: str, password: str, database: str) -> None:
        self._uri = uri
        self._user = user
        self._password = password
        self._database = database
        self._driver_handle: Any = None

    @property
    def _driver(self) -> Any:
        if self._driver_handle is None:
            from neo4j import GraphDatabase

            self._driver_handle = GraphDatabase.driver(
                self._uri, auth=(self._user, self._password)
            )
            self._driver_handle.verify_connectivity()
            with self._driver_handle.session(database=self._database) as session:
                session.run(
                    "CREATE CONSTRAINT aero_node_id IF NOT EXISTS "
                    "FOR (node:AeroNode) REQUIRE node.node_id IS UNIQUE"
                ).consume()
                session.run(
                    "CREATE INDEX aero_node_name IF NOT EXISTS "
                    "FOR (node:AeroNode) ON (node.name)"
                ).consume()
                session.run(
                    "CREATE INDEX aero_node_version IF NOT EXISTS "
                    "FOR (node:AeroNode) ON (node.version_id)"
                ).consume()
        return self._driver_handle

    @property
    def backend_name(self) -> str:
        return "neo4j"

    @staticmethod
    def _node(values: Mapping[str, Any]) -> GraphNode:
        return GraphNode(
            node_id=str(values["node_id"]),
            name=str(values["name"]),
            kind=str(values["kind"]),
            description=str(values.get("description", "")),
            source_ref=str(values["source_ref"]),
            version_id=str(values["version_id"]),
            properties=json.loads(str(values.get("properties_json", "{}"))),
        )

    @staticmethod
    def _edge(values: Mapping[str, Any]) -> GraphEdge:
        return GraphEdge(
            edge_id=str(values["edge_id"]),
            source_id=str(values["source_id"]),
            target_id=str(values["target_id"]),
            relation=str(values["relation"]),
            source_ref=str(values["source_ref"]),
            version_id=str(values["version_id"]),
            confidence=float(values["confidence"]),
            properties=json.loads(str(values.get("properties_json", "{}"))),
        )

    def upsert_nodes(self, nodes: Sequence[GraphNode]) -> int:
        if not nodes:
            return 0
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
                {
                    "node_id": node.node_id,
                    "name": node.name,
                    "kind": node.kind,
                    "description": node.description,
                    "source_ref": node.source_ref,
                    "version_id": node.version_id,
                    "properties_json": json.dumps(
                        dict(node.properties), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                UNWIND $nodes AS row
                MERGE (node:AeroNode {node_id: row.node_id})
                SET node.name = row.name,
                    node.kind = row.kind,
                    node.description = row.description,
                    node.source_ref = row.source_ref,
                    node.version_id = row.version_id,
                    node.properties_json = row.properties_json
                RETURN count(node) AS total
                """,
                nodes=records,
            ).single()
        return int(result["total"] if result else 0)

    def upsert_edges(self, edges: Sequence[GraphEdge]) -> int:
        if not edges:
            return 0
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
                {
                    "edge_id": edge.edge_id,
                    "source_id": edge.source_id,
                    "target_id": edge.target_id,
                    "relation": edge.relation,
                    "source_ref": edge.source_ref,
                    "version_id": edge.version_id,
                    "confidence": edge.confidence,
                    "properties_json": json.dumps(
                        dict(edge.properties), ensure_ascii=False, sort_keys=True
                    ),
                }
            )
        with self._driver.session(database=self._database) as session:
            result = session.run(
                """
                UNWIND $edges AS row
                MATCH (source:AeroNode {node_id: row.source_id})
                MATCH (target:AeroNode {node_id: row.target_id})
                MERGE (source)-[edge:AERO_RELATION {edge_id: row.edge_id}]->(target)
                SET edge.source_id = row.source_id,
                    edge.target_id = row.target_id,
                    edge.relation = row.relation,
                    edge.source_ref = row.source_ref,
                    edge.version_id = row.version_id,
                    edge.confidence = row.confidence,
                    edge.properties_json = row.properties_json
                RETURN count(edge) AS total
                """,
                edges=records,
            ).single()
        total = int(result["total"] if result else 0)
        if total != len(edges):
            raise ValueError("one or more graph edge endpoints do not exist")
        return total

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
        active = sorted(active_version_ids) if active_version_ids is not None else None
        seen = {node_id}
        frontier = {node_id}
        output: list[Neighbor] = []
        with self._driver.session(database=self._database) as session:
            for depth in range(1, max_depth + 1):
                next_frontier: set[str] = set()
                for current_id in sorted(frontier):
                    records = session.run(
                        """
                        MATCH (source:AeroNode {node_id: $node_id})
                              -[edge:AERO_RELATION]-(target:AeroNode)
                        WHERE $active IS NULL OR (
                            source.version_id IN $active AND target.version_id IN $active
                            AND edge.version_id IN $active
                        )
                        RETURN properties(source) AS source,
                               properties(edge) AS edge,
                               properties(target) AS target
                        ORDER BY edge.edge_id, target.node_id
                        """,
                        node_id=current_id,
                        active=active,
                    )
                    for record in records:
                        target_values = dict(record["target"])
                        target_id = str(target_values["node_id"])
                        if target_id in seen:
                            continue
                        output.append(
                            Neighbor(
                                source=self._node(dict(record["source"])),
                                edge=self._edge(dict(record["edge"])),
                                target=self._node(target_values),
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
        with self._driver.session(database=self._database) as session:
            records = session.run(
                """
                MATCH (node:AeroNode)
                WHERE toLower(node.name) CONTAINS toLower($keyword)
                   OR toLower(node.description) CONTAINS toLower($keyword)
                RETURN properties(node) AS node
                ORDER BY node.name, node.node_id LIMIT $limit
                """,
                keyword=keyword,
                limit=limit,
            )
            return [self._node(dict(record["node"])) for record in records]

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
        active = sorted(active_version_ids) if active_version_ids is not None else None
        with self._driver.session(database=self._database) as session:
            node_records = list(
                session.run(
                    """
                    MATCH (node:AeroNode)
                    WHERE $active IS NULL OR node.version_id IN $active
                    RETURN properties(node) AS node
                    ORDER BY node.kind, node.name, node.node_id LIMIT $limit
                    """,
                    active=active,
                    limit=limit,
                )
            )
            nodes = [self._node(dict(record["node"])) for record in node_records]
            node_ids = [node.node_id for node in nodes]
            if not node_ids:
                return [], []
            edge_records = session.run(
                """
                MATCH (source:AeroNode)-[edge:AERO_RELATION]->(target:AeroNode)
                WHERE source.node_id IN $node_ids AND target.node_id IN $node_ids
                RETURN properties(edge) AS edge
                ORDER BY edge.edge_id LIMIT $limit
                """,
                node_ids=node_ids,
                limit=limit * 2,
            )
            edges = [self._edge(dict(record["edge"])) for record in edge_records]
        return nodes, edges

    def delete_source(self, source_ref: str) -> tuple[int, int]:
        with self._driver.session(database=self._database) as session:
            counts = session.run(
                """
                MATCH (source:AeroNode)-[edge:AERO_RELATION]->(target:AeroNode)
                WHERE edge.source_ref = $source_ref OR source.source_ref = $source_ref
                   OR target.source_ref = $source_ref
                RETURN count(DISTINCT edge) AS edges
                """,
                source_ref=source_ref,
            ).single()
            node_count = session.run(
                "MATCH (node:AeroNode {source_ref: $source_ref}) RETURN count(node) AS nodes",
                source_ref=source_ref,
            ).single()
            session.run(
                "MATCH ()-[edge:AERO_RELATION]->() WHERE edge.source_ref = $source_ref "
                "DELETE edge",
                source_ref=source_ref,
            ).consume()
            session.run(
                "MATCH (node:AeroNode {source_ref: $source_ref}) DETACH DELETE node",
                source_ref=source_ref,
            ).consume()
        return (
            int(node_count["nodes"] if node_count else 0),
            int(counts["edges"] if counts else 0),
        )

    def counts(self) -> tuple[int, int]:
        with self._driver.session(database=self._database) as session:
            nodes = session.run("MATCH (node:AeroNode) RETURN count(node) AS total").single()
            edges = session.run(
                "MATCH ()-[edge:AERO_RELATION]->() RETURN count(edge) AS total"
            ).single()
        return int(nodes["total"] if nodes else 0), int(edges["total"] if edges else 0)

    def close(self) -> None:
        if self._driver_handle is not None:
            self._driver_handle.close()
            self._driver_handle = None
