from __future__ import annotations

from aerodiagnosis.retrieval.evaluation import evaluate_rankings


def test_evaluate_rankings_aggregates_metrics() -> None:
    report = evaluate_rankings(
        [
            (["a", "b"], {"a"}, {"a": 2, "b": 1}),
            (["x", "a"], {"a"}, {"a": 2}),
        ],
        k=2,
    )

    assert report.query_count == 2
    assert report.precision_at_k == 0.5
    assert report.recall_at_k == 1.0
    assert report.mrr == 0.75
    assert report.ndcg_at_k > 0.8
