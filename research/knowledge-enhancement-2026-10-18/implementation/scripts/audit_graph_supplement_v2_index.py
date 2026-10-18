"""Source-only independent vector, old-row and graph anchor audit; no questions."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from prepare_graph_supplement_v2 import OUT, ROOT, digest, read, save, sha

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.ports import GraphNode, VectorChunk
from aerodiagnosis.retrieval.graph_paths import _quote_valid, entity_identity


def main():
    index = OUT / "index.sqlite"
    before = digest(index)
    corpus = read(ROOT / "corpus-final.json")
    assert before == corpus["index_sha256"]
    base = Path(".runtime/strict-graph-holdout-v1/index.sqlite")
    assert digest(base) == corpus["base_index_sha256"]
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    new = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in new["chunks"]}
    old_counts = {}
    maximum = 0.0
    max_tokens = 0
    with (
        sqlite3.connect(base.resolve().as_uri() + "?mode=ro", uri=True) as old,
        sqlite3.connect(index.resolve().as_uri() + "?mode=ro", uri=True) as db,
    ):
        for table in ["vector_chunks", "knowledge_lexical_chunks"]:
            rows = old.execute("select * from " + table).fetchall()
            for row in rows:
                found = db.execute(
                    "select * from " + table + " where chunk_id=?", (row[0],)
                ).fetchone()
                assert found == row, (table, row[0])
            old_counts[table] = len(rows)
        db.row_factory = sqlite3.Row
        for i, c in enumerate(chunks.values()):
            row = db.execute(
                "select * from vector_chunks where chunk_id=?", (c.chunk_id,)
            ).fetchone()
            assert (row["document_id"], row["version_id"], row["content"]) == (
                c.document_id,
                c.version_id,
                c.content,
            )
            meta = json.loads(row["metadata_json"])
            assert meta["display_translation_source_sha256"] == sha(c.content)
            assert meta["translation_review_status"] == "source_semantic_checked_nonexpert"
            display = meta["display_content_zh"]
            assert meta["vector_embedding_input"] == {
                "policy": "chinese_display@1",
                "provider": provider.name,
                "sha256": sha(display),
            }
            stored = json.loads(row["embedding_json"])
            expected = provider.embed(display)
            assert len(stored) == len(expected) == 384
            error = max(abs(a - b) for a, b in zip(stored, expected, strict=True))
            maximum = max(maximum, error)
            assert error < 1e-7
            max_tokens = max(max_tokens, len(provider.counter.encode(display).ids))
            lexical = db.execute(
                "select * from knowledge_lexical_chunks where chunk_id=?", (c.chunk_id,)
            ).fetchone()
            assert lexical["content"] == c.content and lexical["version_id"] == c.version_id
            assert json.loads(lexical["metadata_json"]) == c.metadata
            if (i + 1) % 200 == 0:
                print("New vectors rechecked", i + 1, "/", len(chunks), flush=True)
    graph = read(ROOT / "source-graph-final.json")
    proofs = 0
    for n in graph["nodes"]:
        node = GraphNode(**n)
        assert entity_identity(node, chunks).startswith("entity:")
        p = node.properties["entity_link"]
        assert _quote_valid(chunks[p["chunk_id"]], p)
        proofs += 1
    for e in graph["edges"]:
        for p in [e["properties"], *e["properties"].get("supporting_evidence", [])]:
            assert _quote_valid(chunks[p["chunk_id"]], p)
            proofs += 1
    assert digest(index) == before and digest(base) == corpus["base_index_sha256"]
    save(
        OUT / "index-independent-audit.json",
        dict(
            vectors_recomputed=len(chunks),
            dimensions=384,
            max_abs_error=maximum,
            old_rows_all_columns_unchanged=old_counts,
            graph_proofs=proofs,
            max_chunk_tokens=max_tokens,
            all_new_translation_provenance_equal=True,
            index_sha256=before,
            index_unchanged=True,
            api_calls=0,
            questions_seen=False,
        ),
    )
    print("Index audit passed", len(chunks), maximum, flush=True)


if __name__ == "__main__":
    main()
