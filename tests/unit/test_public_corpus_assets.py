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
    assert manifest["version"] == "1.1.0"
    assert len(manifest["documents"]) == 10
    assert len({item["document_id"] for item in manifest["documents"]}) == 10
    assert manifest["raw_storage"].startswith(".runtime/")
    artifacts = [artifact for item in manifest["documents"] for artifact in item["artifacts"]]
    assert len(artifacts) == 17
    assert all(re.fullmatch(r"[0-9a-f]{64}", item["sha256"]) for item in artifacts)
    assert all(
        item["url"].startswith(
            (
                "https://ntrs.nasa.gov/",
                "https://www.faa.gov/",
                "https://external.apic4e.faa.gov/",
                "https://data.ntsb.gov/",
                "https://www.federalregister.gov/",
                "https://www.ebi.ac.uk/",
            )
        )
        for item in artifacts
    )
    assert {
        "ntrs_pdf_with_text_reference",
        "pdf_pages",
        "parallel_markup",
        "jats_xml",
        "faa_sdr_csv",
    } == {item["processor"] for item in manifest["documents"]}
    assert len(manifest["excluded_sources"]) == 3
    assert any("retracted" in item["reason"] for item in manifest["excluded_sources"])
    assert any("BY-NC-ND" in item["reason"] for item in manifest["excluded_sources"])


def test_public_corpus_annotation_starters_are_explicitly_not_gold() -> None:
    parsing = _jsonl(_asset_root() / "parsing_samples.jsonl")
    knowledge = _jsonl(_asset_root() / "knowledge_candidates.jsonl")
    retrieval = _jsonl(_asset_root() / "retrieval_queries.jsonl")

    assert len(parsing) == 16
    assert len(knowledge) == 24
    assert len(retrieval) == 30
    assert all(item["review_status"] != "gold" for item in (*parsing, *knowledge, *retrieval))
    assert {item["document_id"] for item in parsing} >= {
        "NASA-20140003941",
        "FAA-H-8083-32B-CH10",
        "NTSB-DCA18MA142",
        "FR-2022-11926",
        "PMC9407329",
        "PMC9573422",
        "FAA-SDR-2024-ENGINE-SAMPLE",
    }
    assert all(item["qrels"] for item in retrieval)
