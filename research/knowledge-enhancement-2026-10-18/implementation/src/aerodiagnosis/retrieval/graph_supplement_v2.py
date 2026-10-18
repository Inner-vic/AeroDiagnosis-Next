"""Experimental query-directed graph support, preserving all primary text in append mode.

No labels or desired answers enter retrieval. Source authority is rechecked by v1.
Auxiliary evidence consumes the same total token ceiling as the primary evidence.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from aerodiagnosis.ports import GraphStore, VectorChunk, VectorMatch
from aerodiagnosis.ports.vector import CandidateSearch
from aerodiagnosis.retrieval.fusion_policy import same_source_scope
from aerodiagnosis.retrieval.gated_graph import (
    GraphSupplementPolicy,
    SourceGate,
    consensus_scope,
    entity_in_query,
    intended_relations,
    protected_supplement,
)
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates
from aerodiagnosis.retrieval.manual import _covered_source


@dataclass(frozen=True)
class Focus:
    direction: Literal["boundary", "incoming", "outgoing", "other"]


def query_focus(query: str) -> Focus:
    # Positive causal paths cannot establish an unknown cause or a negative boundary.
    if any(
        t in query
        for t in (
            "不确定",
            "反证",
            "未查明",
            "未能确定",
            "尚未确定",
            "排除",
            "确定到",
            "无法确定",
            "证据不足",
            "不能确定",
            "未知根因",
            "假说",
        )
    ):
        return Focus("boundary")
    if any(
        t in query
        for t in (
            "机理",
            "机制",
            "后果",
            "后续",
            "如何导致",
            "如何引起",
            "怎样导致",
            "怎样影响",
            "串联",
            "如何连接",
        )
    ) or ("从" in query and "到" in query):
        return Focus("outgoing")
    if any(t in query for t in ("原因", "根因", "为什么", "为何", "归因")):
        return Focus("incoming")
    return Focus("other")


def focus_reason(query: str, path: dict[str, Any]) -> str | None:
    focus = query_focus(query).direction
    if focus == "boundary":
        return "uncertainty_boundary_abstain"
    names = path.get("entities", [])
    if not names:
        return "missing_focus_endpoint"
    if focus == "incoming" and not entity_in_query(query, names[-1]):
        return "root_requires_terminal_anchor"
    if focus == "outgoing" and not entity_in_query(query, names[0]):
        return "mechanism_requires_initial_anchor"
    return None


class FocusGate(SourceGate):
    def _path_reason(self, query: str, source: VectorChunk, path: dict[str, Any]) -> str | None:
        return super()._path_reason(query, source, path) or focus_reason(query, path)


def append_support(
    *,
    baseline: Sequence[VectorMatch],
    bundles: Sequence[Sequence[VectorMatch]],
    token_count: Callable[[VectorChunk], int],
    token_budget: int = 4096,
    max_per_document: int = 7,
) -> tuple[list[VectorMatch], dict[str, Any]]:
    """Keep every primary hit, then append atomic source bundles up to two slots.

    A missing auxiliary slot never authorizes dropping or truncating primary evidence.
    Source containment is tested only within equal applicability/section scope.
    """
    result = list(baseline)
    sizes = [token_count(h.chunk) for h in result]
    counts = Counter(h.chunk.document_id for h in result)
    if (
        len(result) > 10
        or any(n < 0 for n in sizes)
        or sum(sizes) > token_budget
        or max(counts.values(), default=0) > max_per_document
    ):
        raise ValueError("Invalid primary evidence budget")
    selected = {h.chunk.chunk_id for h in result}
    tokens = sum(sizes)
    added: list[str] = []
    rejects: Counter[str] = Counter()
    for bundle in bundles:
        new = list(
            {h.chunk.chunk_id: h for h in bundle if h.chunk.chunk_id not in selected}.values()
        )
        if not new:
            continue
        if len(added) + len(new) > 2 or len(result) + len(new) > 12:
            rejects["addition_slots"] += 1
            continue
        projected = counts + Counter(h.chunk.document_id for h in new)
        if max(projected.values(), default=0) > max_per_document:
            rejects["document_cap"] += 1
            continue
        if any(
            same_source_scope(h.chunk, old.chunk) and _covered_source(h, old)
            for i, h in enumerate(new)
            for old in [*result, *new[:i]]
        ):
            rejects["contained_source"] += 1
            continue
        costs = [token_count(h.chunk) for h in new]
        if any(n < 0 for n in costs):
            raise ValueError("Negative token cost")
        if tokens + sum(costs) > token_budget:
            rejects["token_budget"] += 1
            continue
        result.extend(new)
        added.extend(h.chunk.chunk_id for h in new)
        selected.update(h.chunk.chunk_id for h in new)
        counts, tokens = projected, tokens + sum(costs)
    return result, dict(
        added=added,
        displaced_text_tail=[],
        context_tokens=tokens,
        budget_rejections=dict(rejects),
        protected_prefix_identical=True,
        primary_count=len(baseline),
        auxiliary_count=len(added),
    )


class DirectedGraphSupport:
    """Research wrapper. Main text is unchanged in append mode; graph failure closes."""

    def __init__(
        self,
        *,
        text: CandidateSearch,
        graph: GraphStore,
        load_chunks: Callable[[Sequence[str]], dict[str, VectorChunk]],
        token_count: Callable[[VectorChunk], int],
        focus: bool = True,
        mode: Literal["append", "replace"] = "append",
        max_per_document: int = 7,
    ) -> None:
        if mode not in {"append", "replace"}:
            raise ValueError("Unknown evidence area")
        self.text, self.graph, self.load, self.count = text, graph, load_chunks, token_count
        self.focus, self.mode, self.doc_cap = focus, mode, max_per_document
        self.last_trace: dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return f"directed_graph_support@2:{self.mode}:focus={self.focus}:doc={self.doc_cap}"

    def search(
        self, query: str, *, top_k: int = 12, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        if not query.strip() or not 1 <= top_k <= (12 if self.mode == "append" else 10):
            raise ValueError("Invalid query or evidence limit")
        self.last_trace = dict(
            policy=self.backend_name,
            graph_requested=0,
            graph_returned=0,
            focus=query_focus(query).direction,
            reason="no_active_scope",
        )
        if not active_version_ids:
            return []
        baseline = self.text.search(query, top_k=10, active_version_ids=active_version_ids)
        if top_k <= (10 if self.mode == "append" else 5):
            self.last_trace["reason"] = "protected_primary"
            return baseline[:top_k]
        scope = consensus_scope(baseline)
        if scope is None:
            self.last_trace["reason"] = "no_source_consensus"
            return baseline[:top_k]
        if self.focus and query_focus(query).direction == "boundary":
            self.last_trace["reason"] = "uncertainty_boundary_abstain"
            return baseline[:top_k]
        if not intended_relations(query):
            self.last_trace["reason"] = "no_relation_intent"
            return baseline[:top_k]
        self.last_trace.update(source_scope=list(scope), reason="gated", graph_requested=25)
        try:
            nodes, edges = self.graph.snapshot(active_version_ids=frozenset({scope[1]}), limit=2000)
            if len(nodes) >= 2000 or len(edges) >= 4000:
                raise ValueError("Truncated graph snapshot")
            keys = {
                str(p.get("chunk_id", ""))
                for e in edges
                for p in [e.properties, *e.properties.get("supporting_evidence", [])]
            }
            keys.update(str(n.properties.get("entity_link", {}).get("chunk_id", "")) for n in nodes)
            chunks = self.load(sorted(keys - {""}))
            raw = GraphPathCandidates(self.graph, self.load).search(
                query, top_k=25, active_version_ids=frozenset({scope[1]})
            )
            self.last_trace["graph_returned"] = len(raw)
            policy = GraphSupplementPolicy(max_per_document=self.doc_cap)
            gate = (FocusGate if self.focus else SourceGate)(
                nodes=nodes, edges=edges, chunks=chunks, scope=scope, policy=policy
            )
            protected = {h.chunk.chunk_id for h in baseline[: 10 if self.mode == "append" else 5]}
            by_id = {h.chunk.chunk_id: h for h in raw}
            bundles = []
            seen = set()
            for h in raw:
                for path in gate.paths(query, h):
                    signature = tuple(path["edge_ids"])
                    if signature in seen:
                        continue
                    seen.add(signature)
                    ids = path["source_chunks"]
                    if any(k not in by_id and k not in protected for k in ids):
                        gate.rejected["incomplete_candidate_bundle"] += 1
                        continue
                    bundles.append(
                        [
                            VectorMatch(
                                replace(
                                    chunks[k],
                                    metadata={
                                        **chunks[k].metadata,
                                        "graph_supplement": {
                                            "policy": self.backend_name,
                                            "path": path,
                                            "certainty_is_not_probability": True,
                                        },
                                    },
                                ),
                                by_id[k].score,
                            )
                            for k in ids
                            if k not in protected
                        ]
                    )
            if self.mode == "append":
                final, trace = append_support(
                    baseline=baseline,
                    bundles=bundles,
                    token_count=self.count,
                    max_per_document=self.doc_cap,
                )
            else:
                final, trace = protected_supplement(
                    baseline=baseline, bundles=bundles, token_count=self.count, policy=policy
                )
            self.last_trace.update(
                trace, gate_rejections=dict(gate.rejected), eligible_bundles=len(bundles)
            )
            return final[:top_k]
        except Exception as error:
            self.last_trace.update(
                reason="graph_unavailable_text_fallback", error_type=type(error).__name__
            )
            return baseline[:top_k]
