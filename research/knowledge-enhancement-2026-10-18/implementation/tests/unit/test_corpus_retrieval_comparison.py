from __future__ import annotations

import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from aerodiagnosis.ports import VectorChunk, VectorMatch

ROOT = Path(__file__).parents[2]


def load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_duplicate_chunks_cannot_inflate_document_recall() -> None:
    runner = load("compare_corpus_retrieval")
    ranked = [VectorMatch(VectorChunk(str(i), "a", "v", "text"), 1) for i in range(5)]
    metrics = runner.score(ranked, {"a": 2, "b": 1})
    assert metrics["labeled_doc_recall_in_top5_chunks"] == 0.5
    assert metrics["unique_documents_in_top5_chunks"] == 1
    assert metrics["labeled_doc_recall_at3"] == 0.5


def test_snapshot_drift_requires_explicit_refreeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = load("materialize_corpus_snapshot")
    raw = tmp_path / "raw"
    raw.mkdir()
    original = b"<article><body><p>Original text.</p></body></article>"
    changed = b"<article><body><p>Updated text.</p></body></article>"
    (raw / "good.xml").write_bytes(original)
    (raw / "changed.xml").write_bytes(changed)
    docs = [
        {
            "document_id": name,
            "processor": "jats_xml",
            "artifacts": [
                {
                    "name": name + ".xml",
                    "sha256": hashlib.sha256(original).hexdigest(),
                    "bytes": len(original),
                    "url": "https://example.test/" + name,
                    "media_type": "application/xml",
                }
            ],
        }
        for name in ("good", "changed", "missing")
    ]
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": docs}), encoding="utf-8")
    monkeypatch.setattr(runner, "MANIFEST", manifest)
    strict = runner.materialize(tmp_path)
    assert [d["document_id"] for d in strict["documents"]] == ["good"]
    frozen_bytes = (tmp_path / "snapshot.json").read_bytes()
    with pytest.raises(FileExistsError, match="already frozen"):
        runner.materialize(tmp_path, allow_refreshed=True)
    assert (tmp_path / "snapshot.json").read_bytes() == frozen_bytes
    refreshed_root = tmp_path / "refreshed"
    shutil.copytree(raw, refreshed_root / "raw")
    refreshed = runner.materialize(refreshed_root, allow_refreshed=True)
    assert refreshed["documents"][1]["cohort"] == "refreshed"
    assert refreshed["unavailable_document_ids"] == ["missing"]


def test_retrieval_excludes_missing_labels_and_checks_frozen_hashes(tmp_path: Path) -> None:
    runner = load("compare_corpus_retrieval")
    text = tmp_path / "text.txt"
    text.write_text("Compressor fouling can cause EGT rise. " * 12, encoding="utf-8")
    snap = tmp_path / "snapshot.json"
    snap.write_text(
        json.dumps(
            {
                "dataset_id": "test",
                "documents": [
                    {
                        "document_id": "a",
                        "processed_path": "text.txt",
                        "sha256": runner.digest(text),
                        "artifacts": [],
                        "cohort": "historical_exact",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    queries = tmp_path / "queries.jsonl"
    queries.write_text(
        "\n".join(
            json.dumps(row)
            for row in [
                {
                    "query_id": "available",
                    "query": "EGT rise",
                    "qrels": {"a": 2},
                    "review_status": "machine_candidate",
                },
                {
                    "query_id": "missing",
                    "query": "NASA",
                    "qrels": {"b": 2},
                    "review_status": "machine_candidate",
                },
            ]
        ),
        encoding="utf-8",
    )
    report = runner.run(snap, queries, tmp_path / "report.json")
    assert report["query_count"] == 1
    assert report["excluded_query_count"] == 1
    assert len(report["query_results"]) == 6
    assert {r["query_id"] for r in report["query_results"]} == {"available"}
    assert report == runner.run(snap, queries, tmp_path / "repeat.json")
    text.write_text("tampered", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        runner.run(snap, queries, tmp_path / "bad.json")
