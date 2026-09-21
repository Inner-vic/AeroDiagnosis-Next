from __future__ import annotations

from aerodiagnosis.retrieval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


def test_precision_and_recall_at_k() -> None:
    relevant = {"a", "b"}
    retrieved = ["a", "c", "b"]

    assert precision_at_k(retrieved, relevant, k=2) == 0.5
    assert recall_at_k(retrieved, relevant, k=2) == 0.5


def test_mean_reciprocal_rank_averages_queries() -> None:
    rankings = [
        (["x", "a"], {"a"}),
        (["a", "b"], {"a"}),
    ]

    assert mean_reciprocal_rank(rankings) == 0.75


def test_ndcg_at_k_scores_relevant_items() -> None:
    retrieved = ["a", "c"]
    relevance = {"a": 2, "c": 0}

    assert ndcg_at_k(retrieved, relevance, k=2) > 0.99
