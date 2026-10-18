from __future__ import annotations

import copy
import importlib.util
from pathlib import Path
from typing import Any

import pytest


@pytest.fixture
def packet(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, Path, dict, dict]:
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location(
        "thesis_preflight_test", scripts / "thesis_preflight.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    text = "Deposits can impair acceleration."
    sources = {
        "d": {
            "text": text,
            "canonical_sha256": module.builder.digest(text.encode()),
            "raw_sha256": "a" * 64,
            "family_id": "family",
            "blocks": [
                {
                    "text": text,
                    "canonical_start": 0,
                    "canonical_end": len(text),
                    "source_locator": {"page": 1},
                }
            ],
        }
    }
    evidence = module.builder.materialize_span(
        {"document_id": "d", "block_index": 0, "anchor_text": text}, sources
    )
    task = {
        "sample_id": "q",
        "fact_id": "f",
        "query": "What can deposits impair?",
        "query_language": "en",
        "scope": {},
        "scope_label": "core",
        "answerability": "answerable",
        "minimal_facts": [{"text": "Acceleration"}],
        "evidence": [evidence],
        "family_ids": ["family"],
        "human_expert_reviewed": False,
        "split": "unassigned",
    }
    monkeypatch.setattr(module.builder, "load_sources", lambda _: sources)
    module.builder.write_jsonl(tmp_path / "tasks.jsonl", [task])
    module.builder.write_jsonl(tmp_path / "blind_queries.jsonl", [module.builder.blind_view(task)])
    module.builder.write_jsonl(
        tmp_path / "review_ledger.jsonl", [{"sample_id": "q", "disposition": "pending"}]
    )
    return module, tmp_path, sources, task


def test_preflight_checks_exact_snapshot_and_does_not_claim_frozen_gold(packet: tuple) -> None:
    module, root, sources, _ = packet
    report = module.validate_candidates(root, sources)
    assert report["status"] == "candidate_not_frozen"
    assert report["validated_evidence_references"] == 1
    assert report["statistical_independence_established"] is False


def test_blind_packet_rejects_reference_answer_leak(packet: tuple) -> None:
    module, root, sources, task = packet
    leaked = {**module.builder.blind_view(task), "reference_answer": "Acceleration"}
    module.builder.write_jsonl(root / "blind_queries.jsonl", [leaked])
    with pytest.raises(ValueError, match="answer-hidden"):
        module.validate_candidates(root, sources)


@pytest.mark.parametrize("change", ["offset", "family", "expert", "split", "duplicate_review"])
def test_preflight_rejects_invalid_candidate_provenance(packet: tuple, change: str) -> None:
    module, root, sources, original = packet
    task = copy.deepcopy(original)
    if change == "offset":
        task["evidence"][0]["canonical_start"] += 1
    if change == "family":
        task["family_ids"] = ["other-family"]
    if change == "expert":
        task["human_expert_reviewed"] = True
    if change == "split":
        task["split"] = "test"
    if change == "duplicate_review":
        module.builder.write_jsonl(
            root / "review_ledger.jsonl",
            [
                {"sample_id": "q", "disposition": "pending"},
                {"sample_id": "q", "disposition": "pending"},
            ],
        )
    module.builder.write_jsonl(root / "tasks.jsonl", [task])
    with pytest.raises(ValueError):
        module.validate_candidates(root, sources)
