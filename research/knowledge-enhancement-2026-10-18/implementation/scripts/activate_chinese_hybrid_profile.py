"""Clone a live profile and rebuild active Chinese vectors; no API or graph writes."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
from contextlib import closing
from pathlib import Path

from aerodiagnosis.adapters.persistence import create_embedding_provider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import VectorChunk


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--from-profile", type=Path, required=True)
    cli.add_argument("--new-runtime", type=Path, required=True)
    args = cli.parse_args()
    before = RuntimeSettings.from_env(args.from_profile)
    root = Path.cwd().resolve()
    target_root = args.new_runtime.resolve()
    if not target_root.is_relative_to(root / ".runtime") or target_root.exists():
        raise ValueError("Fresh project-owned runtime target required")
    target_root.mkdir(parents=True)
    application = target_root / "application"
    application.mkdir()
    profile = target_root / "profile"
    profile.mkdir()
    target = application / "knowledge.sqlite"
    source_sha = sha(before.database_path)
    with (
        closing(sqlite3.connect(before.database_path.as_uri() + "?mode=ro", uri=True)) as src,
        closing(sqlite3.connect(target)) as dst,
    ):
        src.backup(dst)
    for name in ("sources", "extraction-audits"):
        if (before.runtime_dir / name).exists():
            shutil.copytree(before.runtime_dir / name, application / name)
    updates = {
        "AERODIAGNOSIS_RUNTIME_DIR": str(application),
        "AERODIAGNOSIS_DATABASE_PATH": str(target),
        "AERODIAGNOSIS_EMBEDDING_INPUT_POLICY": "chinese_display",
        "AERODIAGNOSIS_MANUAL_VECTOR_WEIGHT": "0.25",
        "AERODIAGNOSIS_MANUAL_SPARSE_WEIGHT": "1",
        "AERODIAGNOSIS_MANUAL_GRAPH_WEIGHT": "0",
        "AERODIAGNOSIS_MANUAL_RETRIEVAL": "hybrid_rrf",
        "AERODIAGNOSIS_MANUAL_RERANKER": "identity",
        "AERODIAGNOSIS_MANUAL_EXPAND_CONTEXT": "false",
        "AERODIAGNOSIS_TEXT_HYBRID_ONLY": "true",
    }
    lines = (args.from_profile / ".env.local").read_text(encoding="utf-8-sig").splitlines()
    lines = [line for line in lines if line.split("=", 1)[0] not in updates]
    lines.extend(key + "=" + value for key, value in updates.items())
    temporary = target_root / "profile-pending"
    temporary.mkdir()
    (temporary / ".env.local").write_text("\n".join(lines) + "\n", encoding="utf-8")
    configured = RuntimeSettings.from_env(temporary)
    assert configured.database_path.resolve() == target
    provider = create_embedding_provider(configured)
    vector = SQLiteChineseVectorStore(target, embedding_provider=provider)
    lexical = SQLiteLexicalStore(target)
    with closing(sqlite3.connect(target)) as db:
        active = frozenset(
            r[0]
            for r in db.execute(
                "SELECT active_version_id FROM documents WHERE active_version_id IS NOT NULL"
            )
        )
        raw = list(
            db.execute(
                "SELECT chunk_id,document_id,version_id,content,metadata_json FROM vector_chunks"
            )
        )
    chunks = [VectorChunk(*r[:4], json.loads(r[4])) for r in raw if r[2] in active]
    if not chunks:
        raise ValueError("Non-empty active source corpus required")
    vector.upsert(chunks)
    lexical.upsert(chunks)
    # Validates every active input marker, source hash and encoder identity.
    vector.search("压气机", top_k=1, active_version_ids=active)
    with closing(sqlite3.connect(target)) as db:
        db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    if sha(before.database_path) != source_sha:
        raise ValueError("Source database changed during migration; do not activate this clone")
    (profile / ".env.local").write_text("\n".join(lines) + "\n", encoding="utf-8")
    receipt = {
        "original_profile": str(args.from_profile.resolve()),
        "new_profile": str(profile),
        "source_database_sha256": source_sha,
        "new_database_sha256": sha(target),
        "active_versions": len(active),
        "active_chunks_rebuilt": len(chunks),
        "physical_chunks": len(raw),
        "inactive_chunks_preserved_excluded_from_search": len(raw) - len(chunks),
        "embedding_identity": vector.embedding_identity,
        "vector_weight": 0.25,
        "sparse_weight": 1,
        "graph_weight": 0,
        "original_profile_unchanged": True,
        "graph_writes": 0,
        "api_calls": 0,
        "evaluation_corpus_imported": False,
    }
    (target_root / "activation-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
