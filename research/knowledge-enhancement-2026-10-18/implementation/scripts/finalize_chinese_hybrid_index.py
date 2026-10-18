"""Finalize reviewed translations and dual indexes without reading held-out questions."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import sys
from contextlib import closing
from pathlib import Path


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def save(p, value):
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, required=True)
    cli.add_argument("--index", type=Path, required=True)
    cli.add_argument("--experiment", type=Path, required=True)
    args = cli.parse_args()
    root, ds, index, exp = (
        Path.cwd(),
        args.dataset.resolve(),
        args.index.resolve(),
        args.experiment.resolve(),
    )
    protocol = read(exp / "protocol.json")
    for name, digest in protocol["file_sha256"].items():
        if sha(exp / "frozen-code" / name) != digest:
            raise ValueError("Frozen implementation changed")
    sys.path.insert(0, str(exp / "frozen-code/src"))
    from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
    from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
    from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
    from aerodiagnosis.ports import VectorChunk

    if (ds / "freeze.json").exists() or (exp / "predictions").exists():
        raise ValueError("Cannot change frozen data or completed results")
    aligned_path = index / "chinese.sqlite"
    if aligned_path.exists():
        raise ValueError("Finalized aligned index is immutable")
    review = read(ds / "translation-review.json")
    items = {item["chunk_id"]: item for item in review["items"]}
    catalog = read(index / "catalog.json")
    expected = {k for doc in catalog["documents"] for k in doc["chunk_ids"]}
    chunks = [VectorChunk(**read(p)) for p in sorted((index / "translated").glob("*.json"))]
    if (
        {c.chunk_id for c in chunks} != expected
        or set(items) != expected
        or not review["coverage_complete"]
        or review["expert_reviewed"]
        or review["reviewed_chunk_count"] != len(chunks)
    ):
        raise ValueError("Complete disclosed nonexpert source review required")
    sources = {}
    for row in read(ds / "source_manifest.json")["sources"]:
        source = ds / row["saved_source_path"]
        if sha(source) != row["saved_source_sha256"]:
            raise ValueError("Source changed")
        sources[row["document_id"]] = source.read_text(encoding="utf-8")
    for c in chunks:
        item = items[c.chunk_id]
        if hashlib.sha256(c.content.encode()).hexdigest() != item["source_sha256"]:
            raise ValueError("Reviewed source differs")
        if c.metadata["display_content_zh"] != (
            item.get("replacement_translation") or item["original_translation"]
        ):
            raise ValueError("Source-verified repair not applied")
        co = c.metadata["locator"]["coordinates"]
        if sources[c.metadata["original_source_id"]][co["start"] : co["end"]] != c.content:
            raise ValueError("Chunk is outside original saved evidence")
    database = index / "knowledge.sqlite"
    with closing(sqlite3.connect(database)) as db:
        before = list(
            db.execute("SELECT chunk_id,embedding_json FROM vector_chunks ORDER BY chunk_id")
        )
        for c in chunks:
            row = db.execute(
                "SELECT content,document_id,version_id FROM vector_chunks WHERE chunk_id=?",
                (c.chunk_id,),
            ).fetchone()
            if row != (c.content, c.document_id, c.version_id):
                raise ValueError("Index differs from reviewed cache")
            db.execute(
                "UPDATE vector_chunks SET metadata_json=? WHERE chunk_id=?",
                (json.dumps(c.metadata, ensure_ascii=False, sort_keys=True), c.chunk_id),
            )
        db.commit()
        if (
            list(db.execute("SELECT chunk_id,embedding_json FROM vector_chunks ORDER BY chunk_id"))
            != before
        ):
            raise ValueError("Original source embeddings changed")
    SQLiteLexicalStore(database).upsert(chunks)
    with closing(sqlite3.connect(database)) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
        with closing(sqlite3.connect(aligned_path)) as dest:
            db.backup(dest)
        active = sorted(catalog["active_versions"])
        active_chunks = [
            VectorChunk(k, d, v, text, json.loads(meta))
            for k, d, v, text, meta in db.execute(
                (
                    "SELECT chunk_id,document_id,version_id,content,metadata_json "
                    "FROM vector_chunks WHERE version_id IN ("
                )
                + ",".join("?" for _ in active)
                + ")",
                active,
            )
        ]
    provider = LocalE5EmbeddingProvider(root / ".runtime/models/multilingual-e5-small")
    if provider.name != catalog["embedding_identity"]:
        raise ValueError("Model differs from original vector corpus")
    aligned = SQLiteChineseVectorStore(aligned_path, embedding_provider=provider)
    if aligned.upsert(active_chunks) != len(active_chunks):
        raise ValueError("Chinese vector rebuild incomplete")
    with closing(sqlite3.connect(aligned_path)) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    qa = {
        "source_only": True,
        "questions_read": False,
        "expert_reviewed": False,
        "reviewed_new_chunks": len(chunks),
        "active_index_chunks": len(active_chunks),
        "original_embeddings_unchanged": True,
        "aligned_embedding_identity": aligned.embedding_identity,
        "review_sha256": sha(ds / "translation-review.json"),
        "repairs_sha256": sha(ds / "translation-repairs.json"),
        "index_sha256": sha(database),
        "aligned_index_sha256": sha(aligned_path),
        "reviewed_cache_sha256": {
            c.chunk_id: sha(index / "translated" / (c.chunk_id + ".json")) for c in chunks
        },
        "finalizer_sha256": sha(Path(__file__)),
    }
    save(index / "quality-finalization.json", qa)
    catalog.update(
        active_index_chunks=len(active_chunks),
        independent_nonexpert_reviewed_chunks=len(chunks),
        expert_reviewed=False,
        reviewed_index_sha256=sha(database),
        aligned_index_sha256=sha(aligned_path),
        quality_finalization_sha256=sha(index / "quality-finalization.json"),
    )
    save(index / "catalog.json", catalog)
    print(json.dumps({k: v for k, v in qa.items() if k != "reviewed_cache_sha256"}), flush=True)


if __name__ == "__main__":
    main()
