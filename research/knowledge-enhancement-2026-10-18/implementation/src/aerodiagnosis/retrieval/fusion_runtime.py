"""Run a frozen fusion policy against real candidate stores, without evaluation labels."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.ports.vector import CandidateSearch
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked, select_fused


class ConfiguredFusionRetriever:
    def __init__(
        self,
        *,
        stores: Mapping[str, CandidateSearch],
        weights: Mapping[str, float],
        budgets: Mapping[str, int],
        load_chunks: Callable[[Sequence[str]], dict[str, VectorChunk]],
        token_count: Callable[[VectorChunk], int],
        mode: str = "rrf",
    ) -> None:
        if not stores or set(stores) != set(weights) or set(stores) != set(budgets):
            raise ValueError("Exactly matching candidate channels required")
        if (
            any(type(n) is not int or n < 1 for n in budgets.values())
            or sum(budgets.values()) != 150
        ):
            raise ValueError("Frozen comparison requires source candidate ceiling150")
        fuse_ranked({name: [] for name in stores}, weights)
        select_fused([], token_count=token_count, mode=mode)
        self.stores, self.weights, self.budgets = dict(stores), dict(weights), dict(budgets)
        self.load, self.token_count, self.mode = load_chunks, token_count, mode
        payload = json.dumps(
            dict(weights=self.weights, budgets=self.budgets, mode=mode), sort_keys=True
        )
        self.identity = "configured_fusion@1:" + hashlib.sha256(payload.encode()).hexdigest()
        self.last_trace: dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return self.identity

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        if not query.strip() or not 1 <= top_k <= 10:
            raise ValueError("Frozen final evidence limit is1..10")
        self.last_trace = {"routes": [], "requested_candidates": 0}
        if not active_version_ids:
            return []
        raw = {
            name: store.search(
                query, top_k=self.budgets[name], active_version_ids=active_version_ids
            )
            for name, store in self.stores.items()
        }
        authoritative = self.load(
            sorted({m.chunk.chunk_id for matches in raw.values() for m in matches})
        )
        routes: dict[str, list[VectorMatch]] = {}
        for name, matches in raw.items():
            if len(matches) > self.budgets[name]:
                raise ValueError("Candidate store exceeded its route budget")
            valid = []
            for match in matches:
                source = authoritative.get(match.chunk.chunk_id)
                if source is None or source.version_id not in active_version_ids:
                    continue
                if (source.document_id, source.version_id, source.content) != (
                    match.chunk.document_id,
                    match.chunk.version_id,
                    match.chunk.content,
                ):
                    continue
                graph = match.chunk.metadata.get("graph_retrieval")
                chunk = (
                    replace(source, metadata={**source.metadata, "graph_retrieval": graph})
                    if graph
                    else source
                )
                valid.append(VectorMatch(chunk, match.score))
            routes[name] = valid
            self.last_trace["routes"].append(
                dict(
                    route=name,
                    requested=self.budgets[name],
                    returned=len(matches),
                    accepted=len(valid),
                    chunk_ids=[m.chunk.chunk_id for m in valid],
                    graph_trace=dict(getattr(self.stores[name], "last_trace", {})),
                )
            )
        fused = fuse_ranked(routes, self.weights)
        selected, trace = select_fused(
            fused, token_count=self.token_count, mode=self.mode, top_k=top_k
        )
        self.last_trace.update(trace)
        self.last_trace["requested_candidates"] = sum(self.budgets.values())
        self.last_trace["actual_candidates"] = sum(len(matches) for matches in raw.values())
        self.last_trace["post_fusion_pool"] = [c.chunk.chunk_id for c in fused[:50]]
        return [
            VectorMatch(
                replace(
                    c.chunk,
                    metadata={
                        **c.chunk.metadata,
                        "fusion_retrieval": {
                            "policy": self.identity,
                            "rrf_score": c.score,
                            "selection_score": score,
                            "route_ranks": dict(c.route_ranks),
                            "mode": self.mode,
                        },
                    },
                ),
                score,
            )
            for c, score in selected
        ]
