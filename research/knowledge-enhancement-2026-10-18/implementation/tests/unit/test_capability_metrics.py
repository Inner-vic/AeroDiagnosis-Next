from __future__ import annotations

import hashlib

import pytest
from scripts.capability_metrics import evidence_metrics


def fixture():
    sources = {"doc-a@v1": "alpha bridge", "doc-b@v1": "bridge omega"}
    units = [
        {
            "unit_id": "u1",
            "source_key": "doc-a@v1",
            "start": 0,
            "end": 12,
            "quote": "alpha bridge",
            "source_sha256": hashlib.sha256(b"alpha bridge").hexdigest(),
        },
        {
            "unit_id": "u2",
            "source_key": "doc-b@v1",
            "start": 0,
            "end": 12,
            "quote": "bridge omega",
            "source_sha256": hashlib.sha256(b"bridge omega").hexdigest(),
        },
    ]
    hits = [{"source_key": k, "start": 0, "end": len(v), "content": v} for k, v in sources.items()]
    return sources, units, hits


def test_multi_support_is_not_first_single_fact_mrr():
    sources, units, hits = fixture()
    result = evidence_metrics(
        sources=sources,
        support_units=units,
        acceptable_support_sets=[["u1", "u2"]],
        hits=hits,
        ks=(1, 2),
    )
    assert not result["retrieval_evidence_metrics"]["1"]["complete_necessary_evidence"]
    assert result["retrieval_evidence_metrics"]["1"]["best_sufficient_set_unit_coverage"] == 0.5
    assert result["retrieval_evidence_metrics"]["2"]["complete_necessary_evidence"]
    assert result["first_complete_support_rank"] == 2
    assert result["complete_support_reciprocal_rank"] == 0.5
    assert result["path_reasoning_correctness"] is None


def test_alternative_sufficient_set_is_accepted():
    sources, units, hits = fixture()
    result = evidence_metrics(
        sources=sources,
        support_units=units,
        acceptable_support_sets=[["u1", "u2"], ["u2"]],
        hits=hits[1:],
    )
    assert result["retrieval_evidence_metrics"]["5"]["complete_necessary_evidence"]
    assert result["semantic_precision"] is None


def test_duplicate_hits_do_not_complete_missing_link():
    sources, units, hits = fixture()
    result = evidence_metrics(
        sources=sources,
        support_units=units,
        acceptable_support_sets=[["u1", "u2"]],
        hits=[hits[0]] * 5,
    )
    assert not result["retrieval_evidence_metrics"]["5"]["complete_necessary_evidence"]
    assert result["complete_support_reciprocal_rank"] == 0


@pytest.mark.parametrize(
    "change", [{"source_key": "doc-a@v0"}, {"content": "fabricated"}, {"start": 1}]
)
def test_forged_or_other_version_hits_are_rejected(change):
    sources, units, hits = fixture()
    hits[0] = {**hits[0], **change}
    with pytest.raises(ValueError):
        evidence_metrics(
            sources=sources, support_units=units, acceptable_support_sets=[["u1", "u2"]], hits=hits
        )


def test_unanswerable_is_not_scored_as_empty_evidence_success():
    result = evidence_metrics(
        sources={}, support_units=[], acceptable_support_sets=[], hits=[], answerable=False
    )
    assert result["retrieval_evidence_metrics"] is None
    assert result["abstention_requires_answer_output"]


def test_blank_and_unknown_gold_are_rejected():
    sources, units, hits = fixture()
    with pytest.raises(ValueError):
        evidence_metrics(
            sources=sources, support_units=units, acceptable_support_sets=[[]], hits=hits
        )
    with pytest.raises(ValueError):
        evidence_metrics(
            sources=sources, support_units=units, acceptable_support_sets=[["missing"]], hits=hits
        )
