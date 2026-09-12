"""Legacy-compatible knowledge graph service with SQLite as the local default."""

from __future__ import annotations

import os
import re
import sqlite3
import time
from typing import Any

from agents.knowledge_extract_agent import Entity, Relation
from config import settings


class KnowledgeGraphService:
    """Expose the legacy graph API through a selected persistence adapter."""

    def __init__(self) -> None:
        self._driver: Any = None
        self._sqlite: sqlite3.Connection | None = None
        self._backend = settings.graph_store_backend

    @property
    def backend_name(self) -> str:
        return self._backend

    async def init(self) -> None:
        if self._backend == "sqlite":
            directory = os.path.dirname(os.path.abspath(settings.sqlite_graph_path))
            os.makedirs(directory, exist_ok=True)
            self._sqlite = sqlite3.connect(settings.sqlite_graph_path)
            self._sqlite.row_factory = sqlite3.Row
            self._sqlite.execute("PRAGMA foreign_keys = ON")
            self._sqlite.executescript(
                """
                CREATE TABLE IF NOT EXISTS entities (
                    name TEXT PRIMARY KEY,
                    type TEXT NOT NULL,
                    description TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    source TEXT NOT NULL,
                    created_at INTEGER NOT NULL,
                    updated_at INTEGER NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_entities_type ON entities(type);
                CREATE INDEX IF NOT EXISTS idx_entities_source ON entities(source);
                CREATE TABLE IF NOT EXISTS relations (
                    head TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
                    relation TEXT NOT NULL,
                    tail TEXT NOT NULL REFERENCES entities(name) ON DELETE CASCADE,
                    confidence REAL NOT NULL,
                    source TEXT NOT NULL,
                    updated_at INTEGER NOT NULL,
                    PRIMARY KEY(head, relation, tail, source)
                );
                """
            )
            self._sqlite.commit()
            return
        if self._backend != "neo4j":
            raise ValueError(f"Unsupported graph store backend: {self._backend}")
        if not settings.neo4j_password:
            raise ValueError("NEO4J_PASSWORD is required when graph_store_backend=neo4j")
        from neo4j import AsyncGraphDatabase

        self._driver = AsyncGraphDatabase.driver(
            settings.neo4j_uri,
            auth=(settings.neo4j_user, settings.neo4j_password),
        )
        await self._ensure_indexes()

    async def close(self) -> None:
        if self._sqlite is not None:
            self._sqlite.close()
            self._sqlite = None
        if self._driver:
            await self._driver.close()

    async def _ensure_indexes(self) -> None:
        index_queries = [
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.name)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.type)",
            "CREATE INDEX IF NOT EXISTS FOR (n:Entity) ON (n.source)",
        ]
        async with self._driver.session() as session:
            for query in index_queries:
                await session.run(query)

    def _connection(self) -> sqlite3.Connection:
        if self._sqlite is None:
            raise RuntimeError("SQLite graph backend has not been initialized")
        return self._sqlite

    async def upsert_entity(self, entity: Entity, version: int = 1, source: str = "") -> None:
        if self._backend == "sqlite":
            now = int(time.time())
            connection = self._connection()
            connection.execute(
                """
                INSERT INTO entities (
                    name, type, description, version, source, created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(name) DO UPDATE SET
                    type=excluded.type,
                    description=CASE WHEN excluded.description <> ''
                        THEN excluded.description ELSE entities.description END,
                    version=excluded.version,
                    source=excluded.source,
                    updated_at=excluded.updated_at
                """,
                (entity.name, entity.type, entity.description, version, source, now, now),
            )
            connection.commit()
            return
        cypher = """
        MERGE (e:Entity {name: $name})
        ON CREATE SET e.type = $type, e.description = $description,
            e.version = $version, e.source = $source,
            e.created_at = $now, e.updated_at = $now
        ON MATCH SET e.description = CASE WHEN $description <> ''
            THEN $description ELSE e.description END,
            e.version = $version, e.source = $source, e.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(
                cypher,
                {
                    "name": entity.name,
                    "type": entity.type,
                    "description": entity.description,
                    "version": version,
                    "source": source,
                    "now": int(time.time()),
                },
            )

    @staticmethod
    def _relation_type(value: str) -> str:
        cleaned = re.sub(r"[^A-Z0-9_]", "_", value.upper().replace(" ", "_"))
        return cleaned if cleaned and not cleaned[0].isdigit() else "RELATED_TO"

    async def add_relation(self, relation: Relation, source: str = "") -> None:
        relation_type = self._relation_type(relation.relation)
        if self._backend == "sqlite":
            connection = self._connection()
            connection.execute(
                """
                INSERT INTO relations (
                    head, relation, tail, confidence, source, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(head, relation, tail, source) DO UPDATE SET
                    confidence=excluded.confidence,
                    updated_at=excluded.updated_at
                """,
                (
                    relation.head,
                    relation_type,
                    relation.tail,
                    relation.confidence,
                    source,
                    int(time.time()),
                ),
            )
            connection.commit()
            return
        cypher = f"""
        MATCH (h:Entity {{name: $head}})
        MATCH (t:Entity {{name: $tail}})
        MERGE (h)-[r:{relation_type}]->(t)
        SET r.confidence = $confidence, r.source = $source, r.updated_at = $now
        """
        async with self._driver.session() as session:
            await session.run(
                cypher,
                {
                    "head": relation.head,
                    "tail": relation.tail,
                    "confidence": relation.confidence,
                    "source": source,
                    "now": int(time.time()),
                },
            )

    async def execute_cypher(self, cypher: str, params: dict | None = None) -> list[dict]:
        if self._backend == "sqlite":
            raise NotImplementedError(
                "Cypher administration is unavailable with the SQLite graph backend; "
                "use the typed graph methods or configure Neo4j"
            )
        async with self._driver.session() as session:
            result = await session.run(cypher, params or {})
            return await result.data()

    async def get_entity(self, name: str) -> dict | None:
        if self._backend == "sqlite":
            row = self._connection().execute(
                "SELECT * FROM entities WHERE name = ?", (name,)
            ).fetchone()
            return {"e": dict(row)} if row else None
        records = await self.execute_cypher(
            "MATCH (e:Entity {name: $name}) RETURN e", {"name": name}
        )
        return records[0] if records else None

    async def get_neighbors(self, entity_name: str, hops: int = 2) -> list[dict]:
        if hops < 1 or hops > 5:
            raise ValueError("hops must be between 1 and 5")
        if self._backend != "sqlite":
            cypher = f"""
            MATCH path = (start:Entity {{name: $name}})-[*1..{hops}]-(neighbor)
            RETURN start.name AS source,
                [r IN relationships(path) | type(r)] AS relations,
                neighbor.name AS target, neighbor.type AS target_type,
                neighbor.description AS target_desc
            LIMIT 50
            """
            return await self.execute_cypher(cypher, {"name": entity_name})

        connection = self._connection()
        seen = {entity_name}
        frontier = [(entity_name, [])]
        results: list[dict] = []
        for _depth in range(hops):
            next_frontier: list[tuple[str, list[str]]] = []
            for current, path_relations in frontier:
                rows = connection.execute(
                    """
                    SELECT relation,
                        CASE WHEN head = ? THEN tail ELSE head END AS target
                    FROM relations WHERE head = ? OR tail = ?
                    ORDER BY relation, target
                    """,
                    (current, current, current),
                ).fetchall()
                for row in rows:
                    target = str(row["target"])
                    if target in seen:
                        continue
                    entity = connection.execute(
                        "SELECT type, description FROM entities WHERE name = ?", (target,)
                    ).fetchone()
                    if entity is None:
                        continue
                    relations = [*path_relations, str(row["relation"])]
                    results.append(
                        {
                            "source": entity_name,
                            "relations": relations,
                            "target": target,
                            "target_type": entity["type"],
                            "target_desc": entity["description"],
                        }
                    )
                    seen.add(target)
                    next_frontier.append((target, relations))
            frontier = next_frontier
            if not frontier or len(results) >= 50:
                break
        return results[:50]

    async def search_entities(self, keyword: str, limit: int = 20) -> list[dict]:
        if self._backend == "sqlite":
            rows = self._connection().execute(
                """
                SELECT name, type, description FROM entities
                WHERE name LIKE ? OR description LIKE ?
                ORDER BY name LIMIT ?
                """,
                (f"%{keyword}%", f"%{keyword}%", limit),
            ).fetchall()
            return [dict(row) for row in rows]
        cypher = """
        MATCH (e:Entity)
        WHERE e.name CONTAINS $keyword OR e.description CONTAINS $keyword
        RETURN e.name AS name, e.type AS type, e.description AS description
        LIMIT $limit
        """
        return await self.execute_cypher(cypher, {"keyword": keyword, "limit": limit})

    async def delete_by_source(self, source: str) -> int:
        if self._backend == "sqlite":
            connection = self._connection()
            connection.execute(
                """
                DELETE FROM relations WHERE source = ?
                    OR head IN (SELECT name FROM entities WHERE source = ?)
                    OR tail IN (SELECT name FROM entities WHERE source = ?)
                """,
                (source, source, source),
            )
            cursor = connection.execute("DELETE FROM entities WHERE source = ?", (source,))
            connection.commit()
            return cursor.rowcount
        records = await self.execute_cypher(
            """
            MATCH (e:Entity {source: $source})
            DETACH DELETE e
            RETURN count(e) AS deleted
            """,
            {"source": source},
        )
        return records[0].get("deleted", 0) if records else 0

    async def get_stats(self) -> dict:
        if self._backend == "sqlite":
            connection = self._connection()
            entity_count = int(connection.execute("SELECT count(*) FROM entities").fetchone()[0])
            relation_count = int(
                connection.execute("SELECT count(*) FROM relations").fetchone()[0]
            )
            return {
                "backend": self._backend,
                "total_entities": entity_count,
                "total_relations": relation_count,
            }
        entity_count = await self.execute_cypher("MATCH (e:Entity) RETURN count(e) AS cnt")
        relation_count = await self.execute_cypher("MATCH ()-[r]->() RETURN count(r) AS cnt")
        return {
            "backend": self._backend,
            "total_entities": entity_count[0]["cnt"] if entity_count else 0,
            "total_relations": relation_count[0]["cnt"] if relation_count else 0,
        }
