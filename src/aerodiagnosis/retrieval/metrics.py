from __future__ import annotations

import math
from collections.abc import Mapping, Sequence


def precision_at_k(retrieved: Sequence[str], relevant: set[str], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    top = retrieved[:k]
    if not top:
        return 0.0
    return sum(item in relevant for item in top) / len(top)


def recall_at_k(retrieved: Sequence[str], relevant: set[str], *, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevant:
        return 0.0
    return sum(item in relevant for item in retrieved[:k]) / len(relevant)


def mean_reciprocal_rank(
    rankings: Sequence[tuple[Sequence[str], set[str]]],
) -> float:
    if not rankings:
        return 0.0
    reciprocal_ranks: list[float] = []
    for retrieved, relevant in rankings:
        for rank, item in enumerate(retrieved, start=1):
            if item in relevant:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)
    return sum(reciprocal_ranks) / len(rankings)


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: Mapping[str, float],
    *,
    k: int,
) -> float:
    if k <= 0:
        raise ValueError("k must be positive")
    dcg = 0.0
    for index, item in enumerate(retrieved[:k], start=1):
        gain = relevance.get(item, 0.0)
        if gain > 0:
            dcg += gain / math.log2(index + 1)
    ideal_gains = sorted((gain for gain in relevance.values() if gain > 0), reverse=True)
    idcg = sum(gain / math.log2(index + 1) for index, gain in enumerate(ideal_gains[:k], start=1))
    if idcg == 0:
        return 0.0
    return dcg / idcg
