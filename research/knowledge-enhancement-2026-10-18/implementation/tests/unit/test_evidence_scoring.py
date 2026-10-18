from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

MODULE = Path(__file__).resolve().parents[2] / "scripts/score_evidence_benchmark.py"
spec = importlib.util.spec_from_file_location("evidence_scoring_test", MODULE)
assert spec and spec.loader
scorer = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = scorer
spec.loader.exec_module(scorer)


def span(doc: str = "D1", start: int = 10, end: int = 20) -> dict[str, Any]:
    return {
        "document_id": doc,
        "canonical_start": start,
        "canonical_end": end,
        "canonical_sha256": "source-hash",
    }


def test_overlapping_chunks_do_not_double_count_evidence() -> None:
    assert scorer.covered_fraction(span(), [span(start=8, end=15), span(start=13, end=18)]) == 0.8
    assert scorer.covered_fraction(span(), [span(start=8, end=15), span(start=15, end=22)]) == 1


def test_same_text_on_other_source_is_not_exact_evidence() -> None:
    assert scorer.covered_fraction(span(), [span("D2")]) == 0
    assert scorer.covered_fraction(span(), [{**span(), "canonical_sha256": "wrong"}]) == 0


def test_required_multi_evidence_is_not_satisfied_by_one_piece() -> None:
    query = {
        "answerability": "answerable",
        "evidence_requirement": "all",
        "evidence": [span(), span("D2")],
    }
    score = scorer.evidence_metrics(query, [span()], 5)
    assert score["source_hit_at_5"] == 1
    assert score["required_evidence_hit_at_5"] == 0
    assert score["evidence_coverage_at_5"] == 0.5
    query["evidence_requirement"] = "any"
    assert scorer.evidence_metrics(query, [span()], 5)["required_evidence_hit_at_5"] == 1


def test_partial_evidence_not_counted_as_complete() -> None:
    query = {"answerability": "answerable", "evidence_requirement": "all", "evidence": [span()]}
    score = scorer.evidence_metrics(query, [span(end=19)], 5)
    assert score["required_evidence_hit_at_5"] == 0
    assert score["evidence_coverage_at_5"] == 0.9


def test_insufficiency_is_not_in_recall_denominator() -> None:
    with pytest.raises(ValueError, match="denominator"):
        scorer.evidence_metrics({"answerability": "insufficient_evidence"}, [], 5)


def test_macro_f1_penalizes_majority_class_prediction() -> None:
    assert scorer.macro_f1(
        ["supported", "unsupported"], ["supported", "supported"], ("supported", "unsupported")
    ) == pytest.approx(1 / 3)


def test_context_rank_cutoff_matters() -> None:
    query = {"answerability": "answerable", "evidence_requirement": "all", "evidence": [span()]}
    chunks = [span("D2")] * 5 + [span()]
    assert scorer.evidence_metrics(query, chunks, 5)["required_evidence_hit_at_5"] == 0
    assert scorer.evidence_metrics(query, chunks, 10)["required_evidence_hit_at_10"] == 1


@pytest.fixture
def scoring_dataset(tmp_path: Path) -> tuple[Path, Path]:
    def write(name: str, values: Any, jsonl: bool = True) -> None:
        text = "".join(json.dumps(v) + "\n" for v in values) if jsonl else json.dumps(values)
        (tmp_path / name).write_text(text, encoding="utf-8")

    queries = [
        {
            "sample_id": sid,
            "fact_id": fact,
            "split": "test",
            "query_variant_type": "native",
            "answerability": "answerable",
            "evidence_requirement": "all",
            "evidence": [span()],
            "statistical_cluster": "F1",
            "task_type": "mechanism",
            "query_language": "en",
        }
        for sid, fact in [("q1", "fact1"), ("q2", "fact1"), ("q3", "fact2")]
    ]
    queries.append(
        {
            **queries[0],
            "sample_id": "q4",
            "fact_id": "fact3",
            "answerability": "insufficient_evidence",
            "evidence": [],
        }
    )
    write("queries.jsonl", queries)
    write("graph_claims.jsonl", [])
    write(
        "corpus_manifest.json",
        {
            "documents": [
                {
                    "document_id": "D1",
                    "canonical_sha256": "source-hash",
                    "canonical_character_count": 30,
                }
            ]
        },
        False,
    )
    core = ("queries.jsonl", "graph_claims.jsonl", "corpus_manifest.json")
    write(
        "release.json",
        {"files": {n: hashlib.sha256((tmp_path / n).read_bytes()).hexdigest() for n in core}},
        False,
    )
    write(
        "predictions.jsonl",
        [
            {
                "sample_id": q["sample_id"],
                "abstained": q["sample_id"] == "q4",
                "chunks": [span()] if q["sample_id"] == "q1" else [],
            }
            for q in queries
        ],
    )
    return tmp_path, tmp_path / "predictions.jsonl"


def test_fact_macro_average_and_refusal_denominator(scoring_dataset: tuple[Path, Path]) -> None:
    dataset, predictions = scoring_dataset
    result = scorer.evaluate(dataset, predictions, "test")
    assert result["answerable_query_count"] == 3
    assert result["answerable_fact_count"] == 2
    assert result["metrics"]["required_evidence_hit_at_5"] == 0.25
    assert result["abstention_precision"] == result["abstention_recall"] == 1
    assert (
        result["dataset_release_sha256"]
        == hashlib.sha256((dataset / "release.json").read_bytes()).hexdigest()
    )


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "wrong_hash", "bool_offset"])
def test_invalid_prediction_sets_fail_closed(
    scoring_dataset: tuple[Path, Path], mutation: str
) -> None:
    dataset, path = scoring_dataset
    predictions = scorer.rows(path)
    if mutation == "missing":
        predictions.pop()
    elif mutation == "duplicate":
        predictions.append(predictions[0])
    elif mutation == "wrong_hash":
        predictions[0]["chunks"][0]["canonical_sha256"] = "wrong"
    else:
        predictions[0]["chunks"][0]["canonical_start"] = True
    path.write_text("".join(json.dumps(p) + "\n" for p in predictions), encoding="utf-8")
    with pytest.raises(ValueError):
        scorer.evaluate(dataset, path, "test")


def test_edited_labels_cannot_be_silently_scored(scoring_dataset: tuple[Path, Path]) -> None:
    dataset, predictions = scoring_dataset
    with (dataset / "queries.jsonl").open("a", encoding="utf-8") as stream:
        stream.write("\n")
    with pytest.raises(ValueError, match="Dataset fingerprint mismatch"):
        scorer.evaluate(dataset, predictions, "test")


@pytest.mark.parametrize("variant", ["native", "negation_stress", "all"])
def test_graph_pairs_are_grouped_and_single_class_f1_is_not_reported(
    scoring_dataset: tuple[Path, Path], variant: str
) -> None:
    dataset, predictions = scoring_dataset
    claims = [
        {
            "sample_id": "g1",
            "split": "test",
            "label": "supported",
            "polarity": "positive",
            "graph_variant_type": "native",
            "claim_group_id": "g1",
        },
        {
            "sample_id": "g2",
            "split": "test",
            "label": "unsupported",
            "polarity": "negative",
            "graph_variant_type": "negation_stress",
            "claim_group_id": "g1",
        },
    ]
    (dataset / "graph_claims.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in claims), encoding="utf-8"
    )
    release_path = dataset / "release.json"
    release = json.loads(release_path.read_text(encoding="utf-8"))
    release["files"]["graph_claims.jsonl"] = hashlib.sha256(
        (dataset / "graph_claims.jsonl").read_bytes()
    ).hexdigest()
    release_path.write_text(json.dumps(release), encoding="utf-8")
    graph_predictions = dataset / "graph_predictions.jsonl"
    graph_predictions.write_text(
        "".join(
            json.dumps({"sample_id": c["sample_id"], "label": c["label"]}) + "\n"
            for c in claims
            if variant == "all" or c["graph_variant_type"] == variant
        ),
        encoding="utf-8",
    )
    result = scorer.evaluate(
        dataset, predictions, "test", graph_predictions=graph_predictions, graph_variant=variant
    )
    assert result["graph_claim_group_count"] == 1
    assert result["qualified_claim_accuracy"] == 1
    assert result["qualified_claim_macro_f1"] == (1 if variant == "all" else None)
