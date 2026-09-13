from __future__ import annotations

import json
import re
from pathlib import Path


def _asset_root() -> Path:
    return Path(__file__).parents[2] / "evaluation" / "public_aero_corpus_v1"


def _jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_public_corpus_manifest_has_traceable_official_sources() -> None:
    manifest = json.loads((_asset_root() / "source_manifest.json").read_text(encoding="utf-8"))

    assert manifest["dataset_id"] == "public-aero-corpus-v1"
    assert len(manifest["documents"]) == 5
    assert len({item["document_id"] for item in manifest["documents"]}) == 5
    assert manifest["raw_storage"].startswith(".runtime/")
    artifacts = [artifact for item in manifest["documents"] for artifact in item["artifacts"]]
    assert len(artifacts) == 9
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in artifacts)
    assert all(
        item["url"].startswith(("https://ntrs.nasa.gov/", "https://www.faa.gov/"))
        for item in artifacts
    )
    assert manifest["excluded_sources"][0]["title"] == "CMAPSS Jet Engine Simulated Data"


def test_public_corpus_annotation_starters_are_explicitly_not_gold() -> None:
    parsing = _jsonl(_asset_root() / "parsing_samples.jsonl")
    knowledge = _jsonl(_asset_root() / "knowledge_candidates.jsonl")
    retrieval = _jsonl(_asset_root() / "retrieval_queries.jsonl")

    assert len(parsing) == 8
    assert len(knowledge) == 12
    assert len(retrieval) == 15
    assert all(item["review_status"] != "gold" for item in (*parsing, *knowledge, *retrieval))
    assert {item["document_id"] for item in parsing} >= {
        "NASA-20140003941",
        "FAA-H-8083-32B-CH10",
    }
    assert all(item["qrels"] for item in retrieval)
