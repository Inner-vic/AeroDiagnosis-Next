from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.ports import GraphEdge, GraphNode


def test_sqlite_snapshot_filters_edge_versions_before_limit(tmp_path: Path) -> None:
    graph = SQLiteGraphStore(tmp_path / "graph.db")
    graph.upsert_nodes(
        [
            GraphNode("a", "A", "Symptom", "", "doc", "active"),
            GraphNode("b", "B", "Symptom", "", "doc", "active"),
            GraphNode("z", "Z", "Symptom", "", "old-doc", "old"),
        ]
    )
    graph.upsert_edges(
        [
            *(
                GraphEdge(f"a-foreign-{index}", "a", "b", "CAUSED_BY", "other", "old", 0.5)
                for index in range(6)
            ),
            GraphEdge("z-active", "a", "b", "CAUSED_BY", "doc", "active", 0.5),
            GraphEdge("z-old-target", "a", "z", "CAUSED_BY", "doc", "active", 0.5),
        ]
    )
    nodes, edges = graph.snapshot(active_version_ids=frozenset({"active"}), limit=2)
    assert {node.node_id for node in nodes} == {"a", "b"}
    # Six earlier foreign edges must not exhaust the four-edge LIMIT before filtering.
    assert [edge.edge_id for edge in edges] == ["z-active"]
    all_nodes, all_edges = graph.snapshot()
    assert len(all_nodes) == 3 and len(all_edges) == 8
    assert graph.snapshot(active_version_ids=frozenset()) == ([], [])


class RecordingDriver:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def session(self, **kwargs: Any) -> RecordingDriver:
        return self

    def __enter__(self) -> RecordingDriver:
        return self

    def __exit__(self, *args: Any) -> None:
        pass

    def run(self, query: str, **kwargs: Any) -> list[dict[str, Any]]:
        self.calls.append((" ".join(query.split()), kwargs))
        if "RETURN properties(node) AS node" in query:
            return [
                {
                    "node": {
                        "node_id": "a",
                        "name": "A",
                        "kind": "Symptom",
                        "source_ref": "doc",
                        "version_id": "v1",
                    }
                }
            ]
        return []


@pytest.mark.parametrize(
    "versions,expected",
    [
        (frozenset({"v2", "v1"}), ["v1", "v2"]),
        (None, None),
    ],
)
def test_neo4j_snapshot_constrains_edge_version_before_limit(
    versions: frozenset[str] | None, expected: list[str] | None
) -> None:
    driver = RecordingDriver()
    graph = Neo4jGraphStore(uri="bolt://unused", user="unused", password="unused", database="test")
    graph._driver_handle = driver
    graph.snapshot(active_version_ids=versions, limit=2)
    node_query, node_args = driver.calls[0]
    edge_query, edge_args = driver.calls[1]
    assert "node.version_id IN $active" in node_query
    assert "AND ($active IS NULL OR edge.version_id IN $active)" in edge_query
    assert edge_query.index("edge.version_id IN $active") < edge_query.index("LIMIT $limit")
    assert node_args["active"] == expected
    assert edge_args["active"] == expected
    assert edge_args["node_ids"] == ["a"]
    assert edge_args["limit"] == 4


def test_neo4j_empty_version_snapshot_never_opens_driver() -> None:
    graph = Neo4jGraphStore(uri="bolt://unused", user="unused", password="unused", database="test")
    assert graph.snapshot(active_version_ids=frozenset()) == ([], [])
    assert graph._driver_handle is None
