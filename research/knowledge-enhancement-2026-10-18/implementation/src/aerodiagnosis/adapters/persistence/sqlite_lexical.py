"""Persistent indexed BM25 candidates; exact identifiers plus Chinese unigram/bigrams."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

from aerodiagnosis.adapters.persistence.sqlite import SQLiteDatabase
from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder
from aerodiagnosis.ports import VectorChunk, VectorMatch


class SQLiteLexicalStore:
    def __init__(self, path: Path) -> None:
        self._database = SQLiteDatabase(path)
        self._database.migrate()

    @property
    def backend_name(self) -> str:
        return "sqlite_fts5_bm25@1"

    @property
    def embedding_identity(self) -> str:
        return "lexical_nfkc_en_zh_ngrams@1"

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        with self._database.connect() as connection:
            for chunk in chunks:
                translated = chunk.metadata.get("display_content_zh")
                search_text = chunk.content
                if isinstance(translated, str) and translated != chunk.content:
                    search_text += "\n" + translated
                if not all(
                    (chunk.chunk_id, chunk.document_id, chunk.version_id, chunk.content.strip())
                ):
                    raise ValueError("lexical chunk identity/content must not be empty")
                connection.execute(
                    "INSERT INTO knowledge_lexical_chunks VALUES (?, ?, ?, ?, ?) "
                    "ON CONFLICT(chunk_id) DO UPDATE SET document_id=excluded.document_id, "
                    "version_id=excluded.version_id, content=excluded.content, "
                    "metadata_json=excluded.metadata_json",
                    (
                        chunk.chunk_id,
                        chunk.document_id,
                        chunk.version_id,
                        chunk.content,
                        json.dumps(dict(chunk.metadata), ensure_ascii=False, sort_keys=True),
                    ),
                )
                connection.execute(
                    "DELETE FROM knowledge_lexical_fts WHERE chunk_id=?", (chunk.chunk_id,)
                )
                connection.execute(
                    "INSERT INTO knowledge_lexical_fts(chunk_id,terms) VALUES (?,?)",
                    (chunk.chunk_id, " ".join(HashingEmbedder._tokens(search_text))),
                )
            connection.commit()
        return len(chunks)

    def backfill_from_vectors(self, *, batch_size: int = 200) -> int:
        """Build missing local lexical rows without re-embedding or changing active versions."""
        total = 0
        while True:
            with self._database.connect() as connection:
                rows = connection.execute(
                    "SELECT v.* FROM vector_chunks v WHERE NOT EXISTS "
                    "(SELECT 1 FROM knowledge_lexical_chunks k WHERE k.chunk_id=v.chunk_id) "
                    "ORDER BY v.chunk_id LIMIT ?",
                    (batch_size,),
                ).fetchall()
            if not rows:
                return total
            total += self.upsert(
                [
                    VectorChunk(
                        r["chunk_id"],
                        r["document_id"],
                        r["version_id"],
                        r["content"],
                        json.loads(r["metadata_json"]),
                    )
                    for r in rows
                ]
            )

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        if top_k < 1:
            raise ValueError("top_k must be positive")
        terms = list(dict.fromkeys(HashingEmbedder._tokens(query)))[:64]
        if not terms or active_version_ids == frozenset():
            return []
        match_query = " OR ".join('"' + token.replace('"', '""') + '"' for token in terms)
        with self._database.connect() as connection:
            table = "knowledge_lexical_fts"
            if active_version_ids is not None:
                # A WHERE filter alone leaves inactive rows in BM25's IDF/length statistics.
                # TEMP tables belong to this short-lived connection: parallel scopes cannot
                # contaminate each other, and the persistent local index stays authoritative.
                table = "scoped_lexical_fts"
                connection.execute(
                    "CREATE VIRTUAL TABLE temp.scoped_lexical_fts USING fts5("
                    "chunk_id UNINDEXED, terms, tokenize='unicode61 remove_diacritics 0')"
                )
                versions = sorted(active_version_ids)
                placeholders = ",".join("?" for _ in versions)
                connection.execute(
                    "INSERT INTO temp.scoped_lexical_fts(chunk_id,terms) "
                    "SELECT f.chunk_id,f.terms FROM knowledge_lexical_fts f "
                    "JOIN knowledge_lexical_chunks c ON c.chunk_id=f.chunk_id "
                    "WHERE c.version_id IN (" + placeholders + ")",
                    versions,
                )
            # Only internally selected table identifiers are interpolated; values are bound.
            statement = (
                f"SELECT c.*, bm25({table}) AS rank_score FROM {table} "
                f"JOIN knowledge_lexical_chunks c ON c.chunk_id={table}.chunk_id "
                f"WHERE {table} MATCH ? ORDER BY rank_score ASC, c.chunk_id ASC LIMIT ?"
            )
            rows = connection.execute(statement, (match_query, top_k)).fetchall()
        return [
            VectorMatch(
                VectorChunk(
                    r["chunk_id"],
                    r["document_id"],
                    r["version_id"],
                    r["content"],
                    json.loads(r["metadata_json"]),
                ),
                max(0.0, -float(r["rank_score"])),
            )
            for r in rows
        ]

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, VectorChunk]:
        """Batch local source copies for validating remote candidate payloads, not ranking."""
        identities = sorted(set(chunk_ids))
        if not identities:
            return {}
        result = {}
        with self._database.connect() as connection:
            # Stay below common SQLite variable limits, even for external audit batches.
            for start in range(0, len(identities), 500):
                batch = identities[start : start + 500]
                placeholders = ",".join("?" for _ in batch)
                rows = connection.execute(
                    "SELECT * FROM knowledge_lexical_chunks WHERE chunk_id IN ("
                    + placeholders
                    + ")",
                    batch,
                ).fetchall()
                for row in rows:
                    chunk = VectorChunk(
                        row["chunk_id"],
                        row["document_id"],
                        row["version_id"],
                        row["content"],
                        json.loads(row["metadata_json"]),
                    )
                    result[chunk.chunk_id] = chunk
        return result

    def delete_document(self, document_id: str) -> int:
        with self._database.connect() as connection:
            connection.execute(
                "DELETE FROM knowledge_lexical_fts WHERE chunk_id IN "
                "(SELECT chunk_id FROM knowledge_lexical_chunks WHERE document_id=?)",
                (document_id,),
            )
            cursor = connection.execute(
                "DELETE FROM knowledge_lexical_chunks WHERE document_id=?", (document_id,)
            )
            connection.commit()
            return cursor.rowcount

    def count(self) -> int:
        with self._database.connect() as connection:
            return int(
                connection.execute("SELECT COUNT(*) FROM knowledge_lexical_chunks").fetchone()[0]
            )
