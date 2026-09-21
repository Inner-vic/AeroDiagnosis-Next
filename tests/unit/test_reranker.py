from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aerodiagnosis.retrieval.reranker import (
    IdentityReranker,
    LexicalReranker,
    LLMReranker,
    RankedText,
)


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


class FakeLanguageModel:
    @property
    def identity(self) -> str:
        return "fake-model"

    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        assert task == "rerank_evidence"
        assert payload["query"] == "compressor stall"
        return {"ordered_keys": ["b", "a"]}


def test_llm_reranker_reorders_only_supplied_keys() -> None:
    candidates = [
        RankedText("a", "compressor stall guidance", 0.3),
        RankedText("b", "fuel nozzle flow", 0.9),
    ]

    reranked = LLMReranker(FakeLanguageModel()).rerank(
        "compressor stall",
        candidates,
    )

    assert [item.key for item in reranked] == ["b", "a"]


def test_llm_reranker_falls_back_when_model_returns_unknown_key() -> None:
    class InvalidModel:
        @property
        def identity(self) -> str:
            return "invalid-model"

        def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
            del task, payload
            return {"ordered_keys": ["missing", "a"]}

    candidates = [
        RankedText("a", "compressor stall guidance", 0.3),
        RankedText("b", "fuel nozzle flow", 0.9),
    ]

    reranked = LLMReranker(InvalidModel()).rerank(
        "compressor stall",
        candidates,
    )

    assert [item.key for item in reranked] == ["a", "b"]
