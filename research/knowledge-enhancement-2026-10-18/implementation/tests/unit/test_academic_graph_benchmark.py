from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts/benchmark_academic_graph.py"
spec = importlib.util.spec_from_file_location("academic_graph_test", MODULE)
assert spec and spec.loader
graph = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = graph
spec.loader.exec_module(graph)


def corpus(tmp_path: Path, text: str) -> tuple[Path, dict[str, Any]]:
    raw = b"raw fixture"
    blocks = [
        {
            "text": text,
            "canonical_start": 0,
            "canonical_end": len(text),
            "text_sha256": graph.digest(text.encode()),
            "canonical_sha256": graph.digest(text.encode()),
            "raw_sha256": graph.digest(raw),
            "source_locator": {"page": 7},
        }
    ]
    (tmp_path / "raw").write_bytes(raw)
    (tmp_path / "canonical").write_bytes(text.encode())
    block_bytes = (json.dumps(blocks[0], ensure_ascii=False) + "\n").encode()
    (tmp_path / "blocks").write_bytes(block_bytes)
    source = {
        "document_id": "D1",
        "family_id": "F1",
        "source_url": "https://example.test/source",
        "raw_path": "raw",
        "processed_path": "canonical",
        "blocks_path": "blocks",
        "raw_sha256": graph.digest(raw),
        "raw_bytes": len(raw),
        "canonical_sha256": graph.digest(text.encode()),
        "blocks_sha256": graph.digest(block_bytes),
        "document_description": {"engine_models": ["MODEL-DECLARED-ONLY"], "revision": "candidate"},
    }
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"documents": [source]}), encoding="utf-8")
    return manifest, source


@pytest.mark.parametrize("name", ["raw", "canonical", "blocks"])
def test_hash_locked_sources_reject_any_mutated_artifact(tmp_path: Path, name: str) -> None:
    manifest, _ = corpus(tmp_path, "Compressor fouling causes EGT rise.")
    path = tmp_path / name
    path.write_bytes(path.read_bytes() + b"changed")
    with pytest.raises(ValueError, match="mismatch"):
        graph.load_sources(manifest, tmp_path)


def test_block_span_is_checked_even_when_block_file_hash_is_updated(tmp_path: Path) -> None:
    manifest, source = corpus(tmp_path, "Compressor fouling causes EGT rise.")
    blocks = graph.read_jsonl(tmp_path / "blocks")
    blocks[0]["canonical_start"] = 1
    data = (json.dumps(blocks[0]) + "\n").encode()
    (tmp_path / "blocks").write_bytes(data)
    source["blocks_sha256"] = graph.digest(data)
    manifest.write_text(json.dumps({"documents": [source]}), encoding="utf-8")
    with pytest.raises(ValueError, match="canonical span"):
        graph.load_sources(manifest, tmp_path)


def test_unicode_offsets_and_qualifier_non_inference(tmp_path: Path) -> None:
    text = "Intro: \u03b1\u03b2\u03b3.\nCompressor fouling can cause EGT rise.\n"
    manifest, _ = corpus(tmp_path, text)
    sources = graph.load_sources(manifest, tmp_path)
    summaries, rows, packet = graph.build_observation(sources)
    assert all(s["valid_offsets"] == s["edges"] for s in summaries)
    assert {r["strategy"] for r in rows} == set(graph.STRATEGIES)
    causal = [r for r in rows if r["relation"] == "CAUSED_BY"]
    assert causal
    for row in causal:
        span = text[row["canonical_start"] : row["canonical_end"]]
        assert span == row["evidence_text"]
        assert row["evidence_sha256"] == graph.digest(span.encode())
        assert row["canonical_start"] == text.index("Compressor")
        assert row["source_blocks"][0]["source_locator"] == {"page": 7}
        assert row["source_locator_complete"]
        assert row["modality"] == "possible"
        assert row["polarity"] == "affirmed"
        assert not any(row["structured_qualifiers"].values())
    assert packet
    assert all("evidence_text" not in row and "properties" not in row for row in packet)
    assert all("label" not in row and "confidence" not in row for row in packet)


def test_uniform_review_population_deduplicates_strategy_and_overlap(tmp_path: Path) -> None:
    manifest, _ = corpus(tmp_path, "Compressor fouling causes EGT rise.\n" * 40)
    _, rows, _ = graph.build_observation(graph.load_sources(manifest, tmp_path))
    packet = graph.review_packet(rows, size=10000)
    assert len(packet) == len({graph.claim_key(row) for row in rows})
    assert len({graph.claim_key(row) for row in packet}) == len(packet)
    assert any(len(row["strategies"]) > 1 for row in packet)
    assert graph.review_packet(rows, size=12) == graph.review_packet(list(reversed(rows)), size=12)
    assert len(graph.review_packet(rows, size=12)) == 12


def test_source_locator_mapping_detects_nonwhitespace_gaps() -> None:
    source = {
        "text": "first LOST second",
        "blocks": [
            {
                "text": "first",
                "canonical_start": 0,
                "canonical_end": 5,
                "source_locator": {"page": 1},
                "text_sha256": "a",
            },
            {
                "text": "second",
                "canonical_start": 11,
                "canonical_end": 17,
                "source_locator": {"page": 2},
                "text_sha256": "b",
            },
        ],
    }
    refs, complete = graph.source_blocks(source, 0, 17)
    assert len(refs) == 2
    assert not complete
    source["text"] = "first      second"
    _, complete = graph.source_blocks(source, 0, 17)
    assert complete
