"""Clone an existing profile and reindex every stored chunk with pinned local E5."""

from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
from pathlib import Path

from aerodiagnosis.adapters.persistence import create_embedding_provider
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import VectorChunk


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--from-profile", type=Path, required=True)
    cli.add_argument("--to-profile", type=Path, required=True)
    cli.add_argument("--runtime", type=Path, required=True)
    args = cli.parse_args()
    before = RuntimeSettings.from_env(args.from_profile)
    runtime, profile = args.runtime.resolve(), args.to_profile.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    profile.mkdir(parents=True, exist_ok=True)
    target = runtime / "knowledge.sqlite"
    if target.exists():
        raise ValueError("Fresh clone target required; old profile is retained")
    with sqlite3.connect(before.database_path) as src, sqlite3.connect(target) as dst:
        src.backup(dst)
    for name in ["sources", "extraction-audits"]:
        if (before.runtime_dir / name).exists():
            shutil.copytree(before.runtime_dir / name, runtime / name)
    updates = {
        "AERODIAGNOSIS_RUNTIME_DIR": str(runtime),
        "AERODIAGNOSIS_DATABASE_PATH": str(target),
        "AERODIAGNOSIS_EMBEDDING_BACKEND": "local_e5",
        "AERODIAGNOSIS_EMBEDDING_DIMENSIONS": "384",
        "AERODIAGNOSIS_EMBEDDING_MODEL_PATH": str(
            Path(".runtime/models/multilingual-e5-small").resolve()
        ),
        "AERODIAGNOSIS_MANUAL_RERANKER": "local_e5",
    }
    lines = (args.from_profile / ".env.local").read_text(encoding="utf-8").splitlines()
    lines = [line for line in lines if line.split("=", 1)[0] not in updates]
    lines.extend(key + "=" + value for key, value in updates.items())
    (profile / ".env.local").write_text("\n".join(lines) + "\n", encoding="utf-8")
    settings = RuntimeSettings.from_env(profile)
    if settings.database_path.resolve() != target:
        raise ValueError("Profile did not resolve cloned database")
    provider = create_embedding_provider(settings)
    vector = SQLiteVectorStore(target, embedding_provider=provider)
    with sqlite3.connect(target) as db:
        rows = db.execute(
            "select chunk_id,document_id,version_id,content,metadata_json from vector_chunks"
        ).fetchall()
    chunks = [VectorChunk(*r[:4], json.loads(r[4])) for r in rows]
    vector.upsert(chunks)
    with sqlite3.connect(target) as db:
        dimensions = {
            len(json.loads(r[0])) for r in db.execute("select embedding_json from vector_chunks")
        }
    if dimensions != {384}:
        raise ValueError("Mixed vector dimensions after migration")
    receipt = {
        "original_profile": str(args.from_profile.resolve()),
        "new_profile": str(profile),
        "chunks_reindexed": len(chunks),
        "provider": provider.name,
        "dimensions": 384,
        "original_profile_retained": True,
        "new_holdout_data_promoted": False,
        "api_calls": 0,
    }
    (runtime / "semantic-reindex-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(receipt, ensure_ascii=False))


if __name__ == "__main__":
    main()
