from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from aerodiagnosis.knowledge_benchmark import KnowledgeBenchmark, ReviewRecord, SourceSpan, main


def _payload() -> dict[str, object]:
    source = {"document_id": "doc", "version_id": "v1", "source_hash": "a" * 64}
    return {
        "dataset_id": "engineering-contract-only",
        "version": "1",
        "judgment_protocol": "Synthetic contract fixture; not a domain gold standard",
        "snapshot": [source],
        "items": [
            {
                "annotation_id": "no-answer",
                "task": "retrieval",
                "query": "missing fault?",
                "answerability": "unanswerable",
                "inapplicable_ids": ["doc"],
            },
            {
                "annotation_id": "extract",
                "task": "extraction",
                "source_spans": [
                    {**source, "locator": {"page": 1}, "start": 0, "end": 4, "text": "test"}
                ],
                "relations": [
                    {
                        "subject": "A",
                        "relation": "may_cause",
                        "object": "B",
                        "certainty": "possible",
                        "source_span_index": 0,
                    }
                ],
            },
        ],
    }


def test_candidate_snapshot_is_reproducible_and_unanswerable_supported() -> None:
    benchmark = KnowledgeBenchmark.model_validate(_payload())
    repeated = KnowledgeBenchmark.model_validate_json(benchmark.model_dump_json())
    assert benchmark.content_hash == repeated.content_hash
    summary = benchmark.validation_summary()
    assert summary["review_counts"] == {"machine_candidate": 2}
    assert summary["unanswerable_queries"] == 1
    assert summary["incomplete_judgments"] == 2


@pytest.mark.parametrize("status", ["silver", "gold"])
def test_unreviewed_labels_cannot_be_upgraded(status: str) -> None:
    with pytest.raises(ValidationError, match="two annotators"):
        ReviewRecord.model_validate({"status": status})


def test_gold_needs_expert_freeze_and_timezone() -> None:
    silver = {
        "status": "gold",
        "annotator_ids": ["a", "b"],
        "outcome": "agreed",
        "reviewed_at": "2026-10-12T09:00:00+08:00",
    }
    with pytest.raises(ValidationError, match="expert reviewer"):
        ReviewRecord.model_validate(silver)
    review = ReviewRecord.model_validate(
        {**silver, "expert_reviewer_id": "expert", "frozen_at": "2026-10-12T10:00:00+08:00"}
    )
    assert review.status == "gold"
    with pytest.raises(ValidationError, match="timezone"):
        ReviewRecord.model_validate(
            {**silver, "status": "silver", "reviewed_at": "2026-10-12T09:00:00"}
        )


def test_wrong_snapshot_and_span_offsets_are_rejected() -> None:
    payload = _payload()
    payload["snapshot"] = [{"document_id": "doc", "version_id": "v2", "source_hash": "a" * 64}]
    with pytest.raises(ValidationError, match="frozen source snapshot"):
        KnowledgeBenchmark.model_validate(payload)
    with pytest.raises(ValidationError, match="exact Unicode"):
        SourceSpan(
            document_id="d",
            version_id="v",
            source_hash="a" * 64,
            locator={"page": 1},
            start=0,
            end=3,
            text="test",
        )


def test_invalid_relation_provenance_and_overlap_are_rejected() -> None:
    payload = _payload()
    items = payload["items"]
    assert isinstance(items, list)
    items[1]["relations"][0]["source_span_index"] = 2
    with pytest.raises(ValidationError, match="existing source span"):
        KnowledgeBenchmark.model_validate(payload)
    items[1]["relations"][0]["source_span_index"] = 0
    items[0].update(answerability="answerable", relevant_ids=["doc"])
    with pytest.raises(ValidationError, match="disjoint"):
        KnowledgeBenchmark.model_validate(payload)


def test_validation_cli_is_offline_and_does_not_mutate_labels(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "benchmark.json"
    text = json.dumps(_payload())
    path.write_text(text, encoding="utf-8")
    assert main([str(path)]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["review_counts"] == {"machine_candidate": 2}
    assert path.read_text(encoding="utf-8") == text
    assert main([str(tmp_path / "missing.json")]) == 1
