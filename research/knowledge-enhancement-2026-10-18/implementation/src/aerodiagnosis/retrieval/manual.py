"""Production/manual evidence candidates: indexed BM25 + vector, rerank, safe parent context."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from typing import Any, Literal, cast

from aerodiagnosis.domain import SourceKind, make_evidence_id
from aerodiagnosis.ports import VectorChunk, VectorMatch, VectorStore
from aerodiagnosis.ports.context import ParentContext, ParentContextStore
from aerodiagnosis.ports.vector import CandidateSearch
from aerodiagnosis.retrieval.reranker import IdentityReranker, RankedText, Reranker


@dataclass(frozen=True, slots=True)
class ManualRetrievalPolicy:
    strategy: Literal["vector_only", "sparse_only", "hybrid_rrf"] = "hybrid_rrf"
    candidate_k: int = 50
    rrf_k: int = 60
    vector_weight: float = 1.0
    sparse_weight: float = 1.0
    graph_weight: float = 0.0
    max_per_document: int = 3
    single_source_unlimited: bool = True
    expand_context: bool = True
    vector_min_score: float = 0.05
    max_context_chars: int = 8000
    rerank_char_budget: int = 24000

    def __post_init__(self) -> None:
        if self.strategy not in {"vector_only", "sparse_only", "hybrid_rrf"}:
            raise ValueError("unknown manual retrieval strategy")
        if not 1 <= self.candidate_k <= 500 or not 1 <= self.rrf_k <= 1000:
            raise ValueError("candidate_k/rrf_k outside supported range")
        if (
            not all(
                math.isfinite(w) and 0 <= w <= 5
                for w in (self.vector_weight, self.sparse_weight, self.graph_weight)
            )
            or self.vector_weight + self.sparse_weight <= 0
        ):
            raise ValueError("fusion weights must be finite, nonnegative and not both zero")
        if not 0 <= self.max_per_document <= 500 or not -1 <= self.vector_min_score <= 1:
            raise ValueError("diversity or vector score budget outside supported range")
        if not 100 <= self.max_context_chars <= 8000:
            raise ValueError("context budget must be within evidence bounds")
        if not 100 <= self.rerank_char_budget <= 1_000_000:
            raise ValueError("rerank character budget outside supported range")

    @property
    def identity(self) -> str:
        payload = json.dumps(asdict(self), sort_keys=True, separators=(",", ":"))
        return "manual_multistage@4:" + hashlib.sha256(payload.encode()).hexdigest()


class ManualEvidenceRetriever:
    handles_reranking = True

    def __init__(
        self,
        *,
        vector_store: VectorStore,
        lexical_store: VectorStore,
        context_store: ParentContextStore | None = None,
        policy: ManualRetrievalPolicy | None = None,
        reranker: Reranker | None = None,
        graph_candidates: CandidateSearch | None = None,
    ) -> None:
        self._vector = vector_store
        self._lexical = lexical_store
        self._contexts = context_store
        self.policy = policy or ManualRetrievalPolicy()
        self._reranker = reranker or IdentityReranker()
        self._graph_candidates = graph_candidates
        if self.policy.graph_weight > 0 and (
            graph_candidates is None or self.policy.strategy != "hybrid_rrf"
        ):
            raise ValueError(
                "Positive graph weight requires a graph candidate provider and hybrid_rrf"
            )

    @property
    def backend_name(self) -> str:
        return ":".join(
            (
                self._vector.backend_name,
                self._lexical.backend_name,
                self.policy.identity,
                self._reranker.identity,
            )
        )

    @property
    def embedding_identity(self) -> str:
        return self._vector.embedding_identity

    @property
    def reranker_identity(self) -> str:
        return self._reranker.identity

    def with_reranker(self, reranker: Reranker) -> ManualEvidenceRetriever:
        return ManualEvidenceRetriever(
            vector_store=self._vector,
            lexical_store=self._lexical,
            context_store=self._contexts,
            policy=self.policy,
            reranker=reranker,
            graph_candidates=self._graph_candidates,
        )

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        written = self._vector.upsert(chunks)
        if written != len(chunks):
            raise RuntimeError("vector candidate index write incomplete")
        if self._lexical.upsert(chunks) != len(chunks):
            raise RuntimeError("lexical candidate index write incomplete")
        return written

    def delete_document(self, document_id: str) -> int:
        count = self._vector.delete_document(document_id)
        self._lexical.delete_document(document_id)
        if self._contexts is not None:
            self._contexts.delete_document(document_id)
        return count

    def get_chunk(self, chunk_id: str) -> VectorChunk | None:
        getter = getattr(self._vector, "get_chunk", None)
        return getter(chunk_id) if getter is not None else None

    def count(self) -> int:
        return self._vector.count()

    def _expand(
        self, match: VectorMatch, cache: dict[tuple[str, str], ParentContext | None]
    ) -> VectorMatch:
        metadata = dict(match.chunk.metadata)
        detail = dict(metadata.get("retrieval", {}))
        detail["context_expansion"] = "disabled" if not self.policy.expand_context else "no_parent"
        result = replace(
            match, chunk=replace(match.chunk, metadata={**metadata, "retrieval": detail})
        )
        parent_id = metadata.get("parent_context_id")
        if (
            not self.policy.expand_context
            or self._contexts is None
            or not isinstance(parent_id, str)
        ):
            return result
        # A child translation cannot describe a larger parent source range.
        # Keep the child until separately aligned parent translations are available.
        if isinstance(metadata.get("display_content_zh"), str):
            detail["context_expansion"] = "translated_child_preserved"
            return result
        cache_key = (parent_id, match.chunk.version_id)
        if cache_key not in cache:
            cache[cache_key] = self._contexts.get(parent_id, version_id=match.chunk.version_id)
        parent = cache[cache_key]
        if parent is None or parent.document_id != match.chunk.document_id:
            detail["context_expansion"] = "missing_or_wrong_source"
            return result
        if len(parent.content) > self.policy.max_context_chars:
            detail["context_expansion"] = "parent_over_budget"
            return result
        child_locator = metadata.get("locator")
        if not isinstance(child_locator, Mapping):
            detail["context_expansion"] = "invalid_child_locator"
            return result
        child_coords = child_locator.get("coordinates", {})
        parent_coords = parent.locator.get("coordinates", {})
        if not isinstance(child_coords, Mapping) or not isinstance(parent_coords, Mapping):
            return result
        points = [
            child_coords.get("start"),
            child_coords.get("end"),
            parent_coords.get("start"),
            parent_coords.get("end"),
        ]
        if not all(type(point) is int for point in points):
            detail["context_expansion"] = "unsupported_locator"
            return result
        start, end, parent_start, parent_end = (cast(int, point) for point in points)
        source_matches = (
            child_locator.get("kind") == parent.locator.get("kind")
            and child_coords.get("page") == parent_coords.get("page")
            and parent_start <= start < end <= parent_end
            and parent.content[start - parent_start : end - parent_start] == match.chunk.content
        )
        if not source_matches:
            detail["context_expansion"] = "parent_anchor_mismatch"
            return result
        detail.update(
            {
                "context_expansion": "expanded",
                "child_chunk_id": match.chunk.chunk_id,
                "child_locator": dict(child_locator),
                "child_content_sha256": hashlib.sha256(match.chunk.content.encode()).hexdigest(),
            }
        )
        locator = dict(parent.locator)
        context_chunk = VectorChunk(
            parent.context_id,
            parent.document_id,
            parent.version_id,
            parent.content,
            {
                **metadata,
                "locator": locator,
                "content_hash": hashlib.sha256(parent.content.encode()).hexdigest(),
                "evidence_id": make_evidence_id(
                    SourceKind.DOCUMENT_CHUNK, parent.context_id, locator
                ),
                "retrieval": detail,
            },
        )
        return VectorMatch(context_chunk, match.score)

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
        active_version_ids: frozenset[str] | None = None,
        _candidate_limit: int | None = None,
    ) -> list[VectorMatch]:
        if not query.strip() or not 1 <= top_k <= 500:
            raise ValueError("query must not be blank and top_k must be within 1..500")
        if active_version_ids == frozenset():
            return []
        limit = _candidate_limit or max(top_k, self.policy.candidate_k)
        routes: dict[str, list[VectorMatch]] = {}
        if self.policy.strategy != "sparse_only" and (
            self.policy.strategy != "hybrid_rrf" or self.policy.vector_weight > 0
        ):
            routes["vector"] = self._vector.search(
                query, top_k=limit, active_version_ids=active_version_ids
            )
        if self.policy.strategy != "vector_only" and (
            self.policy.strategy != "hybrid_rrf" or self.policy.sparse_weight > 0
        ):
            routes["sparse"] = self._lexical.search(
                query, top_k=limit, active_version_ids=active_version_ids
            )
        if self.policy.strategy == "hybrid_rrf" and self.policy.graph_weight > 0:
            assert self._graph_candidates is not None
            routes["graph"] = self._graph_candidates.search(
                query, top_k=limit, active_version_ids=active_version_ids
            )
        lookup = getattr(self._lexical, "get_chunks", None)
        local_sources: dict[str, VectorChunk] | None = (
            lookup([m.chunk.chunk_id for r in ("vector", "graph") for m in routes.get(r, [])])
            if callable(lookup)
            else None
        )
        candidates: dict[str, VectorMatch] = {}
        contributions: dict[str, list[dict[str, Any]]] = {}
        for route, matches in routes.items():
            seen = set()
            rank = 0
            for match in matches:
                if route in {"vector", "graph"} and local_sources is not None:
                    source = local_sources.get(match.chunk.chunk_id)
                    if (
                        source is None
                        or source.document_id != match.chunk.document_id
                        or source.version_id != match.chunk.version_id
                        or source.content != match.chunk.content
                        or source.metadata.get("locator") != match.chunk.metadata.get("locator")
                    ):
                        continue
                    if route == "graph" and "graph_retrieval" in match.chunk.metadata:
                        source = replace(
                            source,
                            metadata={
                                **source.metadata,
                                "graph_retrieval": match.chunk.metadata["graph_retrieval"],
                            },
                        )
                    match = replace(match, chunk=source)
                if (
                    match.chunk.chunk_id in seen
                    or (
                        active_version_ids is not None
                        and match.chunk.version_id not in active_version_ids
                    )
                    or len(match.chunk.content) > self.policy.max_context_chars
                    or (route == "vector" and match.score < self.policy.vector_min_score)
                    or (route == "sparse" and match.score <= 0)
                ):
                    continue
                seen.add(match.chunk.chunk_id)
                rank += 1
                key = match.chunk.chunk_id
                # Local lexical records are the ingested source copy, preferred to mirror payloads.
                if key not in candidates or route == "sparse":
                    old = candidates.get(key)
                    if old is not None and (
                        old.chunk.version_id != match.chunk.version_id
                        or old.chunk.document_id != match.chunk.document_id
                        or old.chunk.content != match.chunk.content
                    ):
                        # A conflicting mirror payload cannot vote for the local source identity.
                        contributions[key] = []
                    candidates[key] = match
                if route == "graph" and "graph_retrieval" in match.chunk.metadata:
                    candidate = candidates[key]
                    candidates[key] = replace(
                        candidate,
                        chunk=replace(
                            candidate.chunk,
                            metadata={
                                **candidate.chunk.metadata,
                                "graph_retrieval": match.chunk.metadata["graph_retrieval"],
                            },
                        ),
                    )
                contributions.setdefault(key, []).append(
                    {
                        "route": route,
                        "rank": rank,
                        "raw_score": match.score,
                        "rrf": 1 / (self.policy.rrf_k + rank),
                    }
                )
        single_source = (
            self.policy.single_source_unlimited
            and active_version_ids is not None
            and len(active_version_ids) == 1
            and len({m.chunk.document_id for m in candidates.values()}) == 1
        )
        document_limit = 0 if single_source else self.policy.max_per_document
        # An explicit scope gives an upper bound; do not widen merely because
        # the diversity policy makes the requested Top K impossible.
        attainable_k = top_k
        if document_limit and active_version_ids is not None:
            attainable_k = min(top_k, len(active_version_ids) * document_limit)
        fused = []
        for key, match in candidates.items():
            parts = contributions[key]
            if self.policy.strategy == "vector_only":
                score = max(-1.0, min(1.0, match.score))
            elif self.policy.strategy == "sparse_only":
                # BM25 scores have no cosine scale. Preserve raw score; expose bounded rank score.
                score = 1 / parts[0]["rank"]
            else:
                weights = {
                    "vector": self.policy.vector_weight,
                    "sparse": self.policy.sparse_weight,
                    "graph": self.policy.graph_weight,
                }
                score = sum(float(part["rrf"]) * weights[str(part["route"])] for part in parts) / (
                    (
                        self.policy.vector_weight
                        + self.policy.sparse_weight
                        + self.policy.graph_weight
                    )
                    / (self.policy.rrf_k + 1)
                )
            metadata = {
                **match.chunk.metadata,
                "retrieval": {
                    "algorithm": self.policy.identity,
                    "policy": asdict(self.policy),
                    "reranker": self._reranker.identity,
                    "effective_max_per_document": document_limit,
                    "document_limit_reason": "explicit_single_source"
                    if single_source
                    else "configured_diversity",
                    "attainable_top_k_upper_bound": attainable_k,
                    "score_semantics": "ranking_signal_not_probability",
                    "route_candidates": {r: len(items) for r, items in routes.items()},
                    "contributions": parts,
                    "graph_source": match.chunk.metadata.get("graph_retrieval"),
                },
            }
            fused.append(VectorMatch(replace(match.chunk, metadata=metadata), score))
        fused.sort(key=lambda item: (-item.score, item.chunk.chunk_id))
        # Bound the fused pool and whole-text reranking payload independently of final Top K.
        fused = fused[:limit]
        rerank_input: list[RankedText] = []
        used_characters = 0
        budget_exhausted = False
        for match in fused:
            fits = (
                not budget_exhausted
                and used_characters
                + len(str(match.chunk.metadata.get("display_content_zh", match.chunk.content)))
                <= self.policy.rerank_char_budget
            )
            detail = match.chunk.metadata["retrieval"]
            detail["rerank_budget_included"] = fits
            detail["fused_candidate_count"] = len(fused)
            if fits:
                rerank_input.append(
                    RankedText(
                        match.chunk.chunk_id,
                        str(match.chunk.metadata.get("display_content_zh", match.chunk.content)),
                        match.score,
                    )
                )
                used_characters += len(
                    str(match.chunk.metadata.get("display_content_zh", match.chunk.content))
                )
            else:
                budget_exhausted = True
        # Reranking sees only source children, before larger parent context is filled.
        by_key = {item.chunk.chunk_id: item for item in fused}
        ranked = self._reranker.rerank(query, rerank_input)
        ranked_ids = list(dict.fromkeys(item.key for item in ranked if item.key in by_key))
        ranked_ids.extend(key for key in by_key if key not in ranked_ids)
        selected: list[VectorMatch] = []
        document_counts: Counter[str] = Counter()
        selected_ids: set[str] = set()
        context_cache: dict[tuple[str, str], ParentContext | None] = {}
        for key in ranked_ids:
            match = self._expand(by_key[key], context_cache)
            duplicate = next((prior for prior in selected if _covered_source(match, prior)), None)
            if duplicate is not None:
                duplicate.chunk.metadata["retrieval"].setdefault(
                    "duplicate_source_ranges", []
                ).append(
                    {
                        "chunk_id": match.chunk.chunk_id,
                        "locator": match.chunk.metadata.get("locator"),
                    }
                )
                continue
            if match.chunk.chunk_id in selected_ids or (
                document_limit and document_counts[match.chunk.document_id] >= document_limit
            ):
                continue
            selected.append(match)
            selected_ids.add(match.chunk.chunk_id)
            document_counts[match.chunk.document_id] += 1
            if len(selected) == top_k:
                break
        if (
            len(selected) < attainable_k
            and limit < 500
            and (len(candidates) > limit or any(len(items) >= limit for items in routes.values()))
        ):
            return self.search(
                query,
                top_k=top_k,
                active_version_ids=active_version_ids,
                _candidate_limit=min(500, limit * 2),
            )
        return selected


def _covered_source(candidate: VectorMatch, selected: VectorMatch) -> bool:
    """Collapse only same-version identical or fully contained anchored evidence.

    Independent sources, revisions, conditions and partly overlapping tails remain available.
    """
    a, b = candidate.chunk, selected.chunk
    if (a.document_id, a.version_id) != (b.document_id, b.version_id):
        return False
    if a.metadata.get("document_metadata") != b.metadata.get("document_metadata"):
        return False
    if a.content == b.content:
        return True
    al, bl = a.metadata.get("locator", {}), b.metadata.get("locator", {})
    if not isinstance(al, Mapping) or not isinstance(bl, Mapping):
        return False
    ac, bc = al.get("coordinates", {}), bl.get("coordinates", {})
    if not isinstance(ac, Mapping) or not isinstance(bc, Mapping):
        return False
    points = ac.get("start"), ac.get("end"), bc.get("start"), bc.get("end")
    if not all(type(p) is int for p in points):
        return False
    start, end, lower, upper = (cast(int, p) for p in points)
    return (
        al.get("kind") == bl.get("kind")
        and ac.get("page") == bc.get("page")
        and lower <= start < end <= upper
        and len(b.content) == upper - lower
        and b.content[start - lower : end - lower] == a.content
    )
