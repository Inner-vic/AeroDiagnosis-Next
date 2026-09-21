from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class RankedText:
    key: str
    text: str
    score: float


class Reranker(Protocol):
    def rerank(self, query: str, candidates: list[RankedText]) -> list[RankedText]: ...


class IdentityReranker:
    def rerank(self, query: str, candidates: list[RankedText]) -> list[RankedText]:
        del query
        return candidates


class LexicalReranker:
    def __init__(self, *, lexical_weight: float = 0.45) -> None:
        if not 0.0 <= lexical_weight <= 1.0:
            raise ValueError("lexical_weight must be between zero and one")
        self._lexical_weight = lexical_weight
        self._semantic_weight = 1.0 - lexical_weight

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


def _tokens(value: str) -> set[str]:
    normalized = unicodedata.normalize("NFKC", value).lower()
    tokens = set(re.findall(r"[a-z0-9_]{2,}", normalized))
    for sequence in re.findall(r"[\u3400-\u9fff]+", normalized):
        tokens.update(sequence)
        tokens.update(
            sequence[index : index + 2] for index in range(len(sequence) - 1)
        )
    return tokens
