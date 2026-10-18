"""Budget-matched query expansion around existing BM25/E5 candidate stores.

Development only. Reuse the existing outer ManualEvidenceRetriever for route fusion,
source validation, ranking and dedup. No graph, model calls or new index are needed.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
from typing import Any

from aerodiagnosis.ports import VectorChunk, VectorMatch, VectorStore
from aerodiagnosis.retrieval.query_planning import (
    build_query_plan,
    candidate_quotas,
    plan_record,
    planner_identity,
)


class PlannedCandidateStore:
    def __init__(
        self,
        store: VectorStore,
        *,
        terminology: bool = True,
        decompose: bool = False,
        candidate_budget: int = 50,
        audit_sink: Callable[[dict[str, Any]], None] | None = None,
    ) -> None:
        if not 1 <= candidate_budget <= 500:
            raise ValueError("Candidate budget must be within 1..500")
        self.store = store
        self.audit_sink = audit_sink
        self.terminology, self.decompose = terminology, decompose
        self.candidate_budget = candidate_budget

    @property
    def backend_name(self) -> str:
        return (
            self.store.backend_name
            + ":"
            + planner_identity(
                terminology=self.terminology,
                decompose=self.decompose,
                candidate_budget=self.candidate_budget,
            )
        )

    @property
    def embedding_identity(self) -> str:
        return self.store.embedding_identity

    def count(self) -> int:
        return self.store.count()

    def upsert(self, chunks: Sequence[VectorChunk]) -> int:
        return self.store.upsert(chunks)

    def delete_document(self, document_id: str) -> int:
        return self.store.delete_document(document_id)

    def get_chunks(self, chunk_ids: Sequence[str]) -> dict[str, VectorChunk]:
        getter = getattr(self.store, "get_chunks", None)
        if callable(getter):
            return dict(getter(chunk_ids))
        single = getattr(self.store, "get_chunk", None)
        if not callable(single):
            raise TypeError("Local mirror source lookup is required")
        return {key: chunk for key in chunk_ids if (chunk := single(key)) is not None}

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        if not 1 <= top_k <= 500:
            raise ValueError("top_k must be within 1..500")
        if active_version_ids == frozenset():
            return []
        budget = min(top_k, self.candidate_budget)
        plan = build_query_plan(
            query,
            terminology=self.terminology,
            decompose=self.decompose,
            max_queries=min(3, budget),
        )
        quotas = candidate_quotas(budget, len(plan.variants))
        candidates: dict[str, VectorMatch] = {}
        votes: dict[str, list[dict[str, Any]]] = {}
        signatures: dict[str, tuple[str, str, str, object]] = {}
        conflicts: set[str] = set()
        for variant, quota in zip(plan.variants, quotas, strict=True):
            found = self.store.search(
                variant.text, top_k=quota, active_version_ids=active_version_ids
            )
            seen = set()
            rank = 0
            for match in found[:quota]:
                chunk = match.chunk
                if (
                    chunk.chunk_id in seen
                    or match.score <= 0
                    or (
                        active_version_ids is not None
                        and chunk.version_id not in active_version_ids
                    )
                ):
                    continue
                seen.add(chunk.chunk_id)
                rank += 1
                signature = (
                    chunk.document_id,
                    chunk.version_id,
                    chunk.content,
                    chunk.metadata.get("locator"),
                )
                if chunk.chunk_id in signatures and signatures[chunk.chunk_id] != signature:
                    conflicts.add(chunk.chunk_id)
                    continue
                signatures[chunk.chunk_id] = signature
                candidates.setdefault(chunk.chunk_id, match)
                votes.setdefault(chunk.chunk_id, []).append(
                    {
                        "kind": variant.kind,
                        "query": variant.text,
                        "rank": rank,
                        "raw_score": match.score,
                        "rrf": 1 / (60 + rank),
                    }
                )
        ranked = sorted(
            (k for k in candidates if k not in conflicts),
            key=lambda k: (-sum(v["rrf"] for v in votes[k]), k),
        )
        if self.audit_sink is not None:
            self.audit_sink(
                {
                    "plan": plan_record(plan),
                    "requested_limits": list(quotas),
                    "total_requested_candidates": sum(quotas),
                    "votes": votes,
                    "ordered_chunk_ids": ranked,
                    "conflicts": sorted(conflicts),
                }
            )
        output = []
        for key in ranked:
            match = candidates[key]
            detail = {
                "plan": plan_record(plan),
                "requested_limits": list(quotas),
                "total_requested_candidates": sum(quotas),
                "candidate_budget": self.candidate_budget,
                "contributions": votes[key],
                "conflicting_source_ids_discarded": len(conflicts),
            }
            output.append(
                replace(
                    match,
                    chunk=replace(
                        match.chunk, metadata={**match.chunk.metadata, "query_plan": detail}
                    ),
                )
            )
        return output
