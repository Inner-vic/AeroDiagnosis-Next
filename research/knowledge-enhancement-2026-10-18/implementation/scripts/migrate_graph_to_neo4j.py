"""Migrate preserved SQLite graph records to the configured real Neo4j database."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.config import RuntimeSettings


def migrate(profile_root: Path) -> dict[str, object]:
    settings = RuntimeSettings.from_env(profile_root)
    if settings.neo4j_password is None:
        raise ValueError("Configure Neo4j credentials locally before migration")
    source = SQLiteGraphStore(settings.database_path)
    target = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    nodes, edges = source.snapshot(limit=2000)
    if (len(nodes), len(edges)) != source.counts():
        raise ValueError("Graph exceeds bounded migration limit; use durable external sync")
    target.upsert_nodes(nodes)
    target.upsert_edges(edges)
    actual_nodes, actual_edges = target.snapshot(limit=2000)
    node_map = {n.node_id: n for n in actual_nodes}
    edge_map = {e.edge_id: e for e in actual_edges}
    if any(node_map.get(n.node_id) != n for n in nodes):
        raise ValueError("Neo4j node round-trip mismatch")
    if any(edge_map.get(e.edge_id) != e for e in edges):
        raise ValueError("Neo4j edge round-trip mismatch")
    receipt: dict[str, object] = {
        "status": "verified",
        "backend": "neo4j",
        "nodes": len(nodes),
        "edges": len(edges),
        "identities_and_properties_preserved": True,
    }
    (settings.runtime_dir / "neo4j-migration-receipt.json").write_text(
        json.dumps(receipt, indent=2), encoding="utf-8"
    )
    return receipt


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-root", type=Path, required=True)
    print(json.dumps(migrate(parser.parse_args().profile_root), ensure_ascii=False))


if __name__ == "__main__":
    main()
