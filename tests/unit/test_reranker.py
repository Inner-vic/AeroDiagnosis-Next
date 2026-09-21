from __future__ import annotations

from aerodiagnosis.retrieval.reranker import IdentityReranker, LexicalReranker, RankedText


def test_identity_reranker_preserves_order() -> None:
    candidates = [
        RankedText("a", "compressor", 0.2),
        RankedText("b", "turbine", 0.9),
    ]

    assert IdentityReranker().rerank("query", candidates) == candidates


def test_lexical_reranker_boosts_query_overlap() -> None:
    candidates = [
        RankedText("a", "compressor stall guidance", 0.3),
        RankedText("b", "fuel nozzle flow", 0.9),
    ]

    reranked = LexicalReranker().rerank("compressor stall", candidates)

    assert reranked[0].key == "a"
