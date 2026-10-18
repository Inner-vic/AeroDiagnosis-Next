"""Auditable weighted rank fusion and source-safe selection; no benchmark labels."""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.manual import _covered_source


@dataclass(frozen=True)
class FusedCandidate:
    chunk: VectorChunk
    score: float
    route_ranks: Mapping[str, int]


def fuse_ranked(
    routes: Mapping[str, Sequence[VectorMatch]], weights: Mapping[str, float], *, rrf_k: int = 60
) -> list[FusedCandidate]:
    if set(routes) != set(weights) or not 1 <= rrf_k <= 1000:
        raise ValueError("Routes and valid RRF weights required")
    if not weights or any(not math.isfinite(w) or w <= 0 for w in weights.values()):
        raise ValueError("Weights must be finite and positive")
    scores: dict[str, float] = defaultdict(float)
    ranks: dict[str, dict[str, int]] = defaultdict(dict)
    chunks: dict[str, VectorChunk] = {}
    denominator = sum(weights.values()) / (rrf_k + 1)
    for route, matches in routes.items():
        for rank, match in enumerate(matches, 1):
            key = match.chunk.chunk_id
            if route in ranks[key]:
                continue
            if key in chunks and (
                chunks[key].document_id,
                chunks[key].version_id,
                chunks[key].content,
            ) != (match.chunk.document_id, match.chunk.version_id, match.chunk.content):
                raise ValueError("Conflicting authoritative candidate identity")
            chunks[key] = match.chunk
            ranks[key][route] = rank
            scores[key] += weights[route] / (rrf_k + rank) / denominator
    return [
        FusedCandidate(chunks[key], scores[key], ranks[key])
        for key in sorted(scores, key=lambda key: (-scores[key], key))
    ]


def display_terms(chunk: VectorChunk) -> frozenset[str]:
    return frozenset(
        HashingEmbedder._tokens(str(chunk.metadata.get("display_content_zh", chunk.content)))
    )


def select_fused(
    candidates: Sequence[FusedCandidate],
    *,
    token_count: Callable[[VectorChunk], int],
    mode: str = "rrf",
    top_k: int = 10,
    pool_k: int = 50,
    max_per_document: int = 5,
    token_budget: int = 4096,
    terms: Mapping[str, frozenset[str]] | None = None,
) -> tuple[list[tuple[FusedCandidate, float]], dict[str, int]]:
    """MMR changes order, never deletes source tails by semantic similarity.

    Dedup collapses only identical/contained original evidence within a version.
    'no_source_dedup' is an explicit development negative control.
    """
    if mode not in {"rrf", "agreement", "mmr90", "no_source_dedup", "legacy_dedup"}:
        raise ValueError("Unknown source selection policy")
    if min(top_k, pool_k, token_budget) < 1 or max_per_document < 0:
        raise ValueError("Invalid selection budget")
    remaining = list(candidates[:pool_k])
    selected: list[tuple[FusedCandidate, float]] = []
    counts: Counter[str] = Counter()
    tokens = duplicates = document_cap = 0
    terms = terms or {c.chunk.chunk_id: display_terms(c.chunk) for c in remaining}
    while remaining and len(selected) < top_k:

        def score(candidate: FusedCandidate) -> float:
            if mode == "agreement":
                return candidate.score + 0.04 * max(0, len(candidate.route_ranks) - 1)
            if mode != "mmr90":
                return candidate.score
            source = terms[candidate.chunk.chunk_id]
            redundancy = max(
                (
                    len(source & terms[old.chunk.chunk_id])
                    / max(1, len(source | terms[old.chunk.chunk_id]))
                    for old, _ in selected
                ),
                default=0.0,
            )
            return 0.9 * candidate.score - 0.1 * redundancy

        # Include original score to retain deterministic RRF ties in transformed order.
        candidate = min(remaining, key=lambda c: (-score(c), -c.score, c.chunk.chunk_id))
        remaining.remove(candidate)
        if max_per_document and counts[candidate.chunk.document_id] >= max_per_document:
            document_cap += 1
            continue
        if mode != "no_source_dedup" and any(
            (mode == "legacy_dedup" or same_source_scope(candidate.chunk, old.chunk))
            and _covered_source(
                VectorMatch(candidate.chunk, candidate.score), VectorMatch(old.chunk, old.score)
            )
            for old, _ in selected
        ):
            duplicates += 1
            continue
        size = token_count(candidate.chunk)
        if size < 0:
            raise ValueError("Negative token cost")
        if tokens + size > token_budget:
            break
        selected.append((candidate, score(candidate)))
        tokens += size
        counts[candidate.chunk.document_id] += 1
    return selected, {
        "context_tokens": tokens,
        "source_duplicates_skipped": duplicates,
        "document_cap_skipped": document_cap,
    }


def same_source_scope(left: VectorChunk, right: VectorChunk) -> bool:
    keys = ("qualifiers", "applicability", "role", "engine_model", "polarity", "certainty")
    if any(left.metadata.get(key) != right.metadata.get(key) for key in keys):
        return False
    left_path = left.metadata.get("locator", {}).get("coordinates", {}).get("section_path")
    right_path = right.metadata.get("locator", {}).get("coordinates", {}).get("section_path")
    return bool(left_path == right_path)
