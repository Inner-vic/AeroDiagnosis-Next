from __future__ import annotations

from pathlib import Path

import pytest

from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.ports import GraphEdge, GraphNode


def _node(node_id: str, name: str, version: str = "version-1") -> GraphNode:
    return GraphNode(
        node_id=node_id,
        name=name,
        kind="component",
        description=f"{name} description",
        source_ref="manual-1",
        version_id=version,
        properties={"chapter": 1},
    )


def test_graph_store_persists_and_traverses_bidirectionally(tmp_path: Path) -> None:
    path = tmp_path / "runtime.db"
    store = SQLiteGraphStore(path)
    nodes = (_node("fan", "风扇"), _node("compressor", "压气机"), _node("egt", "EGT"))
    edges = (
        GraphEdge("edge-1", "fan", "compressor", "feeds", "manual-1", "version-1"),
        GraphEdge("edge-2", "compressor", "egt", "affects", "manual-1", "version-1"),
    )

    assert store.upsert_nodes(nodes) == 3
    assert store.upsert_edges(edges) == 2
    reopened = SQLiteGraphStore(path)

    assert reopened.backend_name == "sqlite_graph"
    assert reopened.counts() == (3, 2)
    assert [item.target.node_id for item in reopened.neighbors("fan", max_depth=2)] == [
        "compressor",
        "egt",
    ]
    assert reopened.neighbors("egt", max_depth=1)[0].target.node_id == "compressor"
    assert reopened.search_nodes("压气")[0].node_id == "compressor"
    snapshot_nodes, snapshot_edges = reopened.snapshot(active_version_ids=frozenset({"version-1"}))
    assert {node.node_id for node in snapshot_nodes} == {"fan", "compressor", "egt"}
    assert {edge.edge_id for edge in snapshot_edges} == {"edge-1", "edge-2"}


def test_graph_store_filters_versions_and_deletes_source(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "runtime.db")
    store.upsert_nodes((_node("a", "A"), _node("b", "B", "version-2")))
    store.upsert_edges((GraphEdge("edge", "a", "b", "related", "manual-1", "version-2"),))

    assert store.neighbors("a", active_version_ids=frozenset({"version-1"})) == []
    assert store.snapshot(active_version_ids=frozenset()) == ([], [])
    assert store.delete_source("manual-1") == (2, 1)
    assert store.counts() == (0, 0)


def test_graph_store_rejects_invalid_depth_and_confidence(tmp_path: Path) -> None:
    store = SQLiteGraphStore(tmp_path / "runtime.db")

    with pytest.raises(ValueError, match="between 1 and 5"):
        store.neighbors("node", max_depth=0)
    with pytest.raises(ValueError, match="between zero and one"):
        store.upsert_edges((GraphEdge("edge", "a", "b", "related", "source", "version", 1.5),))
    with pytest.raises(ValueError, match="between 1 and 2000"):
        store.snapshot(limit=0)
