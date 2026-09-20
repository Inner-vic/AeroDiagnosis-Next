from __future__ import annotations

import sys
from collections.abc import Iterator, Mapping
from types import SimpleNamespace
from typing import Any

import pytest

from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore
from aerodiagnosis.ports import GraphEdge, GraphNode


class FakeResult:
    def __init__(self, records: list[Mapping[str, Any]] | None = None) -> None:
        self.records = records or []

    def __iter__(self) -> Iterator[Mapping[str, Any]]:
        return iter(self.records)

    def single(self) -> Mapping[str, Any] | None:
        return self.records[0] if self.records else None

    def consume(self) -> None:
        return None


class FakeSession:
    def __init__(self, driver: FakeDriver) -> None:
        self.driver = driver

    def __enter__(self) -> FakeSession:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def run(self, query: str, **params: Any) -> FakeResult:
        compact = " ".join(query.split())
        if compact.startswith(("CREATE CONSTRAINT", "CREATE INDEX")):
            return FakeResult()
        if "UNWIND $nodes" in compact:
            for node in params["nodes"]:
                self.driver.nodes[node["node_id"]] = dict(node)
            return FakeResult([{"total": len(params["nodes"])}])
        if "UNWIND $edges" in compact:
            inserted = 0
            for edge in params["edges"]:
                if (
                    edge["source_id"] in self.driver.nodes
                    and edge["target_id"] in self.driver.nodes
                ):
                    self.driver.edges[edge["edge_id"]] = dict(edge)
                    inserted += 1
            return FakeResult([{"total": inserted}])
        if "properties(source) AS source" in compact:
            current = params["node_id"]
            active = params["active"]
            records = []
            for edge in self.driver.edges.values():
                if current not in (edge["source_id"], edge["target_id"]):
                    continue
                target_id = (
                    edge["target_id"]
                    if edge["source_id"] == current
                    else edge["source_id"]
                )
                source = self.driver.nodes[current]
                target = self.driver.nodes[target_id]
                if active is not None and not all(
                    value["version_id"] in active for value in (source, target, edge)
                ):
                    continue
                records.append({"source": source, "edge": edge, "target": target})
            return FakeResult(records)
        if "toLower(node.name)" in compact:
            keyword = params["keyword"].lower()
            nodes = [
                value
                for value in self.driver.nodes.values()
                if keyword in value["name"].lower()
                or keyword in value["description"].lower()
            ]
            return FakeResult([{"node": node} for node in nodes[: params["limit"]]])
        if "properties(node) AS node" in compact:
            active = params["active"]
            nodes = [
                value
                for value in self.driver.nodes.values()
                if active is None or value["version_id"] in active
            ]
            return FakeResult([{"node": node} for node in nodes[: params["limit"]]])
        if "properties(edge) AS edge" in compact:
            node_ids = set(params["node_ids"])
            edges = [
                edge
                for edge in self.driver.edges.values()
                if edge["source_id"] in node_ids and edge["target_id"] in node_ids
            ]
            return FakeResult([{"edge": edge} for edge in edges[: params["limit"]]])
        if "count(DISTINCT edge) AS edges" in compact:
            source_ref = params["source_ref"]
            total = sum(
                edge["source_ref"] == source_ref
                or self.driver.nodes[edge["source_id"]]["source_ref"] == source_ref
                or self.driver.nodes[edge["target_id"]]["source_ref"] == source_ref
                for edge in self.driver.edges.values()
            )
            return FakeResult([{"edges": total}])
        if "count(node) AS nodes" in compact:
            total = sum(
                node["source_ref"] == params["source_ref"]
                for node in self.driver.nodes.values()
            )
            return FakeResult([{"nodes": total}])
        if "DELETE edge" in compact:
            source_ref = params["source_ref"]
            self.driver.edges = {
                key: edge
                for key, edge in self.driver.edges.items()
                if edge["source_ref"] != source_ref
            }
            return FakeResult()
        if "DETACH DELETE node" in compact:
            source_ref = params["source_ref"]
            removed = {
                key
                for key, node in self.driver.nodes.items()
                if node["source_ref"] == source_ref
            }
            self.driver.nodes = {
                key: node
                for key, node in self.driver.nodes.items()
                if key not in removed
            }
            self.driver.edges = {
                key: edge
                for key, edge in self.driver.edges.items()
                if edge["source_id"] not in removed and edge["target_id"] not in removed
            }
            return FakeResult()
        if "count(node) AS total" in compact:
            return FakeResult([{"total": len(self.driver.nodes)}])
        if "count(edge) AS total" in compact:
            return FakeResult([{"total": len(self.driver.edges)}])
        raise AssertionError(f"unexpected Cypher: {compact}")


class FakeDriver:
    def __init__(self) -> None:
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: dict[str, dict[str, Any]] = {}
        self.connected = False
        self.closed = False

    def verify_connectivity(self) -> None:
        self.connected = True

    def session(self, *, database: str) -> FakeSession:
        assert database == "neo4j"
        return FakeSession(self)

    def close(self) -> None:
        self.closed = True


def _store(monkeypatch: pytest.MonkeyPatch) -> tuple[Neo4jGraphStore, FakeDriver]:
    driver = FakeDriver()
    graph_database = SimpleNamespace(
        driver=lambda uri, auth: (
            driver
            if (uri, auth) == ("bolt://neo4j:7687", ("neo4j", "pw"))
            else None
        )
    )
    monkeypatch.setitem(sys.modules, "neo4j", SimpleNamespace(GraphDatabase=graph_database))
    return (
        Neo4jGraphStore(
            uri="bolt://neo4j:7687", user="neo4j", password="pw", database="neo4j"
        ),
        driver,
    )


def test_neo4j_adapter_round_trips_graph(monkeypatch: pytest.MonkeyPatch) -> None:
    store, driver = _store(monkeypatch)
    nodes = (
        GraphNode("fan", "Fan", "component", "front stage", "manual", "v1"),
        GraphNode("fault", "Corrosion", "fault", "blade corrosion", "manual", "v1"),
    )
    edge = GraphEdge(
        "edge-1", "fan", "fault", "HAS_FAULT", "manual", "v1", 0.8
    )

    assert store.backend_name == "neo4j"
    assert store.upsert_nodes(nodes) == 2
    assert driver.connected is True
    assert store.upsert_edges((edge,)) == 1
    assert store.counts() == (2, 1)
    assert store.search_nodes("corrosion")[0].node_id == "fault"
    assert store.neighbors("fan", active_version_ids=frozenset({"v1"}))[0].depth == 1
    snapshot = store.snapshot(active_version_ids=frozenset({"v1"}))
    assert len(snapshot[0]) == 2 and snapshot[1][0].confidence == pytest.approx(0.8)
    assert store.snapshot(active_version_ids=frozenset()) == ([], [])
    assert store.delete_source("manual") == (2, 1)
    assert store.counts() == (0, 0)
    store.close()
    assert driver.closed is True


def test_neo4j_adapter_validates_inputs(monkeypatch: pytest.MonkeyPatch) -> None:
    store, _driver = _store(monkeypatch)
    assert store.upsert_nodes(()) == 0
    assert store.upsert_edges(()) == 0
    assert store.neighbors("fan", active_version_ids=frozenset()) == []
    with pytest.raises(ValueError, match="identity fields"):
        store.upsert_nodes((GraphNode("", "Fan", "component", "", "source", "v1"),))
    with pytest.raises(ValueError, match="confidence"):
        store.upsert_edges(
            (GraphEdge("edge", "a", "b", "rel", "source", "v1", 1.1),)
        )
    with pytest.raises(ValueError, match="endpoints"):
        store.upsert_edges(
            (GraphEdge("edge", "a", "b", "rel", "source", "v1"),)
        )
    with pytest.raises(ValueError, match="max_depth"):
        store.neighbors("fan", max_depth=6)
    with pytest.raises(ValueError, match="positive"):
        store.search_nodes("fan", limit=0)
    with pytest.raises(ValueError, match="between 1 and 2000"):
        store.snapshot(limit=0)
