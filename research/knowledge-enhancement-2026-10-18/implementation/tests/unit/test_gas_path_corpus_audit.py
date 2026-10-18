from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts/audit_gas_path_corpus.py"
spec = importlib.util.spec_from_file_location("gas_path_scope_test", MODULE)
assert spec and spec.loader
audit = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = audit
spec.loader.exec_module(audit)


def source(text: str = "Gas turbine compressor map. Unrelated text.") -> dict[str, Any]:
    return {
        "document_id": "D",
        "logical_document_id": "L",
        "family_id": "F",
        "title": "test",
        "publisher_id": "FAA",
        "language": "en",
        "source_type": "training",
        "source_url": "https://example.test",
        "raw_path": "raw",
        "processed_path": "text",
        "blocks_path": "blocks",
        "raw_sha256": "raw",
        "canonical_sha256": "canonical",
        "blocks_sha256": "blocks",
        "rights": {"status": "unknown"},
        "artifact_description": {"processor": "pdf"},
        "text": text,
        "document_description": {"engine_domain": "gas_turbine"},
        "blocks": [
            {
                "text": text,
                "canonical_start": 0,
                "canonical_end": len(text),
                "source_locator": {"page": 1},
                "text_sha256": audit.digest(text.encode()),
            }
        ],
    }


def annotation(s: dict[str, Any], start: int = 0, end: int = 27) -> dict[str, Any]:
    return {
        "range_id": "gp-r001",
        "document_id": "D",
        "block_index": 0,
        "scope_label": "core",
        "scope_unit": "evidence_span_not_whole_block",
        "topic_tags": ["compressor"],
        "rationale": "Source-based scope reasoning.",
        "quality_status": "prose_source_verified",
        "evidence": {
            "document_id": "D",
            "block_index": 0,
            "canonical_start": start,
            "canonical_end": end,
            "canonical_sha256": "canonical",
            "raw_sha256": "raw",
            "source_locator": {"page": 1},
            "text_sha256": audit.digest(s["text"][start:end].encode()),
        },
    }


def test_gas_turbine_keywords_never_promote_unreviewed_block() -> None:
    sources, blocks = audit.build_inventory([source()], [])
    assert blocks[0]["candidate_topics"] == ["compressor"]
    assert blocks[0]["block_scope"] == "needs_review"
    assert not blocks[0]["audited_range_ids"]
    assert sources[0]["selection"] == "background_or_pending"
    assert sources[0]["unreviewed_characters"] == len(source()["text"])


def test_verified_subspan_does_not_label_whole_page_or_double_count_overlap() -> None:
    s = source()
    a = annotation(s, 0, 20)
    b = annotation(s, 10, 25)
    b["range_id"] = "gp-r002"
    sources, blocks = audit.build_inventory([s], [a, b])
    assert sources[0]["audited_span_characters"] == 25
    assert sources[0]["unreviewed_characters"] == len(s["text"]) - 25
    assert sources[0]["selection"] == "core_passages_available"
    assert blocks[0]["block_scope"] == "needs_review"
    assert blocks[0]["audited_range_ids"] == ["gp-r001", "gp-r002"]


@pytest.mark.parametrize(
    "mutation", ["hash", "bounds", "whole_document", "label", "identity", "negative_index"]
)
def test_invalid_scope_evidence_rejected(mutation: str) -> None:
    s = source()
    row = annotation(s)
    if mutation == "hash":
        row["evidence"]["text_sha256"] = "forged"
    elif mutation == "bounds":
        row["evidence"]["canonical_end"] = len(s["text"]) + 1
    elif mutation == "whole_document":
        row["scope_unit"] = "whole_document"
    elif mutation == "identity":
        row["document_id"] = "OTHER"
    elif mutation == "negative_index":
        row["block_index"] = row["evidence"]["block_index"] = -1
    else:
        row["scope_label"] = "gas_turbine"
    with pytest.raises(ValueError):
        audit.validate_ranges([row], [s])


def test_old_task_partial_overlap_is_not_promoted_as_complete_scope() -> None:
    s = source()
    a = annotation(s, 0, 20)
    task = {
        "sample_id": "q",
        "fact_id": "f",
        "split": "test",
        "evidence": [{"document_id": "D", "canonical_start": 15, "canonical_end": 30}],
    }
    row = audit.map_tasks([task], [a])[0]
    assert row["recommendation"] == "needs_reauthor_or_scope_review"
    assert row["evidence_scope_candidates"][0]["matches"][0]["full_span_contained"] is False
    task["evidence"][0]["canonical_end"] = 20
    row = audit.map_tasks([task], [a])[0]
    assert row["recommendation"] == "reuse_fact_candidate_requires_new_review"
    a["quality_status"] = "requires_visual_check"
    assert audit.map_tasks([task], [a])[0]["recommendation"] == "needs_reauthor_or_scope_review"


def test_raw_canonical_and_blocks_are_hash_locked(tmp_path: Path) -> None:
    text = "Scope fixture."
    raw = b"raw fixture"
    block = {
        "text": text,
        "canonical_start": 0,
        "canonical_end": len(text),
        "source_locator": {"page": 1},
        "text_sha256": audit.digest(text.encode()),
        "canonical_sha256": audit.digest(text.encode()),
        "raw_sha256": audit.digest(raw),
    }
    block_bytes = (json.dumps(block) + "\n").encode()
    (tmp_path / "raw").write_bytes(raw)
    (tmp_path / "text").write_bytes(text.encode())
    (tmp_path / "blocks").write_bytes(block_bytes)
    s = {
        "document_id": "D",
        "raw_path": "raw",
        "processed_path": "text",
        "blocks_path": "blocks",
        "raw_sha256": audit.digest(raw),
        "canonical_sha256": audit.digest(text.encode()),
        "blocks_sha256": audit.digest(block_bytes),
        "raw_bytes": len(raw),
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [s]}), encoding="utf-8")
    assert audit.load_sources(manifest, tmp_path)[0]["text"] == text
    (tmp_path / "raw").write_bytes(raw + b"changed")
    with pytest.raises(ValueError, match="raw_sha256 mismatch"):
        audit.load_sources(manifest, tmp_path)
