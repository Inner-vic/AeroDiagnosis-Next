"""Integrity checks for paired source-scoped holdout measurement."""

# ruff: noqa: RUF001 -- Tests preserve the actual Chinese query syntax.
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from evaluate_gas_path_holdout import citation_gate, resolve_scope
from summarize_holdout_comparison import paired_summary


def test_explicit_source_scope_never_uses_gold_or_unrelated_documents():
    documents = [
        {"document_id": "holdout-v1-source-a", "active_version_id": "v-a"},
        {"document_id": "holdout-v1-source-b", "active_version_id": "v-b"},
    ]
    body, versions = resolve_scope("依据资料「source-a」：可能原因有哪些？", documents)
    assert body == "可能原因有哪些？" and versions == frozenset(["v-a"])
    with pytest.raises(ValueError):
        resolve_scope("普通问题没有资料范围", documents)
    with pytest.raises(ValueError):
        resolve_scope("依据资料「unknown」：原因是什么？", documents)


def test_empty_citations_require_explicit_abstention():
    assert not citation_gate(
        {"citation_exact_checks": [], "answer": {"insufficient_evidence": False}}
    )
    assert citation_gate({"citation_exact_checks": [], "answer": {"insufficient_evidence": True}})
    assert not citation_gate(
        {"citation_exact_checks": [False], "answer": {"insufficient_evidence": True}}
    )


def result(identifier, cluster, text, graph):
    return {
        "id": identifier,
        "source_cluster": cluster,
        "judge": {"text": {"strict_correct": text}, "text_graph": {"strict_correct": graph}},
    }


def test_cluster_bootstrap_preserves_pairs_and_whole_source_groups():
    data = [
        result("a1", "a", False, True),
        result("a2", "a", False, True),
        result("b", "b", True, False),
    ]
    summary = paired_summary(data, "source_cluster")
    assert summary["graph_minus_text"] == pytest.approx(1 / 3)
    assert summary["equal_source_weight_mean_difference"] == 0
    assert summary["paired_cluster_bootstrap_95_percentile_interval"] == [-1, 1]
    assert summary["graph_only_correct"] == 2 and summary["text_only_correct"] == 1
    assert paired_summary(list(reversed(data)), "source_cluster") == summary


def test_statistics_reject_missing_pairs_and_duplicate_sample_ids():
    with pytest.raises(ValueError, match="Duplicate"):
        paired_summary([result("a", "a", True, True)] * 2, "source_cluster")
    with pytest.raises(ValueError, match="booleans"):
        paired_summary([result("a", "a", 1, True)], "source_cluster")


def test_followup_compact_judge_keeps_every_claim_and_full_evidence():
    from evaluate_followup_holdout import judge_view

    arm = {
        "answer": {
            "answer": "A; B",
            "insufficient_evidence": False,
            "claims": [
                {"text": "A", "support_span_ids": ["full-a"]},
                {"text": "B", "support_span_ids": ["full-b"]},
            ],
        },
        "span_catalog": [
            {"spans": [{"id": "s1", "text": "source A"}, {"id": "s2", "text": "source B"}]}
        ],
        "span_records": [{"span_id": "full-a"}, {"span_id": "full-b"}],
    }
    view = judge_view(arm)
    assert view["answer"]["answer"] == "A; B"
    assert view["answer"]["claims"] == [
        {"text": "A", "support_span_ids": ["s1"]},
        {"text": "B", "support_span_ids": ["s2"]},
    ]
    assert view["evidence"] == arm["span_catalog"]
    arm["span_records"].pop()
    with pytest.raises(ValueError):
        judge_view(arm)


def test_followup_compact_gold_keeps_alternative_acceptance_and_uncertainty():
    from evaluate_followup_holdout import judge_facts

    fact = {
        "fact_key": "f",
        "text": "any one option",
        "certainty": "possible",
        "scope": "this engine",
        "source_quote": "exact source",
        "allowed_alternative_texts": ["a", "b"],
        "source_text_sha256": "redundant",
    }
    actual = judge_facts({"required_facts": [fact]})[0]
    assert actual["allowed_alternative_texts"] == ["a", "b"]
    assert actual["certainty"] == "possible" and actual["scope"] == "this engine"
    assert actual["source_quote"] == "exact source"
    assert "source_text_sha256" not in actual
