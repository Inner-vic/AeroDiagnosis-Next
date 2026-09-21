from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from aerodiagnosis.retrieval.metrics import (
    mean_reciprocal_rank,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)


@dataclass(frozen=True, slots=True)
class RetrievalEvaluationReport:
    precision_at_k: float
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    query_count: int


def evaluate_rankings(
    rankings: Sequence[tuple[Sequence[str], set[str], Mapping[str, float] | None]],
    *,
    k: int = 10,
) -> RetrievalEvaluationReport:
    if not rankings:
        return RetrievalEvaluationReport(0.0, 0.0, 0.0, 0.0, 0)
    precisions: list[float] = []
    recalls: list[float] = []
    ndcgs: list[float] = []
    for retrieved, relevant, relevance in rankings:
        precisions.append(precision_at_k(retrieved, relevant, k=k))
        recalls.append(recall_at_k(retrieved, relevant, k=k))
        ndcgs.append(ndcg_at_k(retrieved, relevance or {}, k=k))
    mrr_input = [(retrieved, relevant) for retrieved, relevant, _ in rankings]
    return RetrievalEvaluationReport(
        precision_at_k=sum(precisions) / len(precisions),
        recall_at_k=sum(recalls) / len(recalls),
        mrr=mean_reciprocal_rank(mrr_input),
        ndcg_at_k=sum(ndcgs) / len(ndcgs),
        query_count=len(rankings),
    )
