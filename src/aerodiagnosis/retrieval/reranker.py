from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol

from aerodiagnosis.ports import LanguageModel


@dataclass(frozen=True, slots=True)
class RankedText:
    key: str
    text: str
    score: float


class Reranker(Protocol):
    @property
    def identity(self) -> str: ...

    def rerank(self, query: str, candidates: list[RankedText]) -> list[RankedText]: ...


class IdentityReranker:
    @property
    def identity(self) -> str:
        return "identity@1"

    def rerank(self, query: str, candidates: list[RankedText]) -> list[RankedText]:
        del query
        return candidates


class LexicalReranker:
    def __init__(self, *, lexical_weight: float = 0.45) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between zero and one")
        self._lexical_weight = lexical_weight
        self._semantic_weight = 1.0 - lexical_weight

    @property
    def identity(self) -> str:
        return f"lexical@{self._lexical_weight}"

    def rerank(self, query: str, candidates: list[RankedText]) -> list[RankedText]:
        query_tokens = _tokens(query)
        if not query_tokens:
            return sorted(candidates, key=lambda item: -item.score)
        scored: list[tuple[RankedText, float]] = []
        for candidate in candidates:
            overlap = len(query_tokens & _tokens(candidate.text))
            lexical_score = overlap / len(query_tokens)
            combined = (
                self._semantic_weight * candidate.score
                + self._lexical_weight * lexical_score
            )
            scored.append((candidate, combined))
        scored.sort(key=lambda item: (-item[1], item[0].key))
        return [candidate for candidate, _score in scored]


class LLMReranker:
    """Use the configured language model as a constrained evidence reranker."""

    def __init__(self, language_model: LanguageModel, *, fallback: Reranker | None = None) -> None:
        self._language_model = language_model
        self._fallback = fallback or LexicalReranker()

    @property
    def identity(self) -> str:
        return f"llm_reranker:{self._language_model.identity}"

    def rerank(self, query: str, candidates: list[RankedText]) -> list[RankedText]:
        if not candidates:
            return []
        by_key = {candidate.key: candidate for candidate in candidates}
        payload = {
            "query": query,
            "candidates": [
                {"key": candidate.key, "text": candidate.text, "score": candidate.score}
                for candidate in candidates
            ],
        }
        try:
            raw = self._language_model.complete_json("rerank_evidence", payload)
            ordered_keys = raw.get("ordered_keys")
            if not isinstance(ordered_keys, list):
                return self._fallback.rerank(query, candidates)
            missing = [key for key in ordered_keys if key not in by_key]
            if missing:
                return self._fallback.rerank(query, candidates)
            ordered = [by_key[key] for key in ordered_keys if isinstance(key, str)]
            for candidate in candidates:
                if candidate.key not in ordered_keys:
                    ordered.append(candidate)
            return list(dict.fromkeys(ordered))
        except Exception:
            return self._fallback.rerank(query, candidates)


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).lower()
    tokens = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.update(sequence)
        tokens.update(
            sequence[index : index + 2] for index in range(len(sequence) - 1)
        )
    return tokens
