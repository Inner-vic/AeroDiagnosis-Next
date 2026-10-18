"""Optional source-bound graph supplements with an immutable text prefix.

Confidence means explicit source/review rules, never an estimated probability.
No query labels, desired hop counts or evaluation answers enter this module.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any, Literal

from aerodiagnosis.application.fault_graph import graph_edge_is_active
from aerodiagnosis.ports import GraphEdge, GraphNode, GraphStore, VectorChunk, VectorMatch
from aerodiagnosis.ports.vector import CandidateSearch
from aerodiagnosis.retrieval.chinese import normalize_query, query_terms
from aerodiagnosis.retrieval.fusion_policy import same_source_scope
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates, _quote_valid, entity_identity
from aerodiagnosis.retrieval.manual import _covered_source

_CAUSAL = frozenset(
    {
        "CAUSES",
        "CONTRIBUTES_TO",
        "POSSIBLY_CAUSED_BY",
        "HAS_MECHANISM",
        "INDICATES",
        "ASSOCIATED_WITH",
    }
)
_ACTION = frozenset(
    {
        "HAS_LIMITATION",
        "APPLIES_TO",
        "REQUIRES_ACTION",
        "RECOMMENDS",
        "CHECK_WITH",
        "MITIGATED_BY",
        "MITIGATES",
        "HAS_RECOMMENDED_PERIOD",
        "HAS_SCHEDULED_MODIFICATION",
    }
)
_OBSERVATION = frozenset({"HAS_FINDING", "HAS_HISTORY", "FOUND_IN", "INDICATES"})
_ASSERTED = frozenset({"established", "observed"})
_QUALIFIED = frozenset(
    {
        "possible",
        "suspected",
        "likely",
        "potential",
        "plausible",
        "probable",
        "conditional",
        "initial_hypothesis",
        "conditional_mechanism",
        "suggested_probable",
        "appeared_or_inferred",
    }
)


def intended_relations(query: str) -> frozenset[str]:
    relations: set[str] = set()
    if any(
        t in query
        for t in (
            "为什么",
            "原因",
            "导致",
            "后果",
            "机理",
            "机制",
            "如何连接",
            "串联",
            "归因",
            "关联",
            "怎样影响",
        )
    ):
        relations.update(_CAUSAL)
    if any(
        t in query
        for t in (
            "限制",
            "条件",
            "适用",
            "时限",
            "强制",
            "间隔",
            "要求",
            "措施",
            "检查",
            "建议",
            "维修",
            "改装",
        )
    ):
        relations.update(_ACTION)
    if any(t in query for t in ("发现", "观察", "拆检", "征象", "告警")):
        relations.update(_OBSERVATION)
    return frozenset(relations)


def entity_in_query(query: str, name: str) -> bool:
    q, n = normalize_query(query), normalize_query(name).strip()
    if not n or n in {"发动机", "叶片", "故障", "损伤", "失效", "裂纹", "检查", "制造商"}:
        return False
    if len(n) >= 4 and n in q:
        return True
    # Require two actual lexical units, not a single generic character/bigram.
    units = query_terms(n)
    matched = {t for t in units if t in q}
    return len(matched) >= 2 and len(matched) / max(1, len(units)) >= 0.5


@dataclass(frozen=True)
class GraphSupplementPolicy:
    gate: Literal["source", "strict", "uncertainty"] = "strict"
    protect: Literal[5, 10] = 5
    max_additions: int = 2
    token_budget: int = 4096
    max_per_document: int = 5

    def __post_init__(self) -> None:
        if self.gate not in {"source", "strict", "uncertainty"} or self.protect not in {5, 10}:
            raise ValueError("Unknown graph supplement policy")
        if not 1 <= self.max_additions <= 2 or self.token_budget < 1 or self.max_per_document < 1:
            raise ValueError("Invalid supplement budget")


def consensus_scope(baseline: Sequence[VectorMatch]) -> tuple[str, str] | None:
    votes = Counter((h.chunk.document_id, h.chunk.version_id) for h in baseline[:3])
    if not votes:
        return None
    scope, count = votes.most_common(1)[0]
    return scope if count >= 2 else None


class SourceGate:
    """Recheck source authority, every edge and mention, scope and forward continuity."""

    def __init__(
        self,
        *,
        nodes: Sequence[GraphNode],
        edges: Sequence[GraphEdge],
        chunks: Mapping[str, VectorChunk],
        scope: tuple[str, str],
        policy: GraphSupplementPolicy,
    ) -> None:
        self.nodes = {n.node_id: n for n in nodes}
        self.edges = {e.edge_id: e for e in edges}
        self.chunks, self.scope, self.policy = dict(chunks), scope, policy
        self.rejected: Counter[str] = Counter()

    def paths(self, query: str, match: VectorMatch) -> list[dict[str, Any]]:
        source = self.chunks.get(match.chunk.chunk_id)
        if (
            source is None
            or (source.document_id, source.version_id, source.content)
            != (match.chunk.document_id, match.chunk.version_id, match.chunk.content)
            or (source.document_id, source.version_id) != self.scope
            or not math.isfinite(match.score)
        ):
            self.rejected["candidate_authority_or_scope"] += 1
            return []
        result = []
        for path in match.chunk.metadata.get("graph_retrieval", {}).get("paths", []):
            reason = self._path_reason(query, source, path)
            if reason:
                self.rejected[reason] += 1
            else:
                result.append(path)
        return result

    def _path_reason(self, query: str, source: VectorChunk, path: dict[str, Any]) -> str | None:
        ids = path.get("edge_ids", [])
        if (
            not ids
            or len(ids) > 3
            or len(set(ids)) != len(ids)
            or path.get("direction") != "forward"
        ):
            return "path_shape"
        edges = [self.edges.get(i) for i in ids]
        previous = None
        all_nodes = []
        expected_chunks = []
        for edge in edges:
            if edge is None:
                return "missing_edge"
            p = edge.properties
            if (
                edge.version_id != self.scope[1]
                or edge.source_ref != "document:" + self.scope[0]
                or p.get("document_id") != self.scope[0]
                or p.get("validation_status") != "necessary_source_checks_passed"
                or p.get("independent_source_review")
                not in {"accepted", "revise", "revised", "qualified"}
                or p.get("polarity") != "positive"
                or not graph_edge_is_active(edge)
            ):
                return "edge_authority_or_scope"
            pair = [self.nodes.get(edge.source_id), self.nodes.get(edge.target_id)]
            identities = []
            for node in pair:
                if (
                    node is None
                    or node.version_id != edge.version_id
                    or node.source_ref != edge.source_ref
                ):
                    return "entity_authority"
                identity = entity_identity(node, self.chunks)
                if not identity.startswith("entity:"):
                    return "unverified_entity"
                identities.append(identity)
                all_nodes.append(node)
            if previous is not None and previous != identities[0]:
                return "disconnected_direction"
            previous = identities[1]
            for proof in [p, *p.get("supporting_evidence", [])]:
                chunk = self.chunks.get(str(proof.get("chunk_id", "")))
                if (
                    chunk is None
                    or (chunk.document_id, chunk.version_id) != self.scope
                    or not _quote_valid(chunk, proof)
                ):
                    return "invalid_source_quote"
                expected_chunks.append(chunk.chunk_id)
            if self.policy.gate != "source":
                certainty = p.get("certainty")
                qualified_query = any(
                    t in query for t in ("可能", "疑似", "不确定", "假说", "尚未", "是否", "能否")
                )
                if certainty not in _ASSERTED and not (
                    self.policy.gate == "uncertainty"
                    and qualified_query
                    and certainty in _QUALIFIED
                ):
                    return "certainty_not_eligible"
                if edge.relation not in intended_relations(query):
                    return "relation_intent_mismatch"
        present_edges = [e for e in edges if e is not None]
        expected_names = [all_nodes[0].name] + [n.name for n in all_nodes[1::2]]
        if (
            path.get("hops") != len(present_edges)
            or path.get("entities") != expected_names
            or path.get("relations") != [e.relation for e in present_edges]
            or path.get("certainty")
            != [e.properties.get("certainty", "unknown") for e in present_edges]
            or path.get("qualifiers") != [e.properties.get("qualifiers", {}) for e in present_edges]
        ):
            return "path_metadata_mismatch"
        if (
            list(dict.fromkeys(expected_chunks)) != path.get("source_chunks")
            or source.chunk_id not in expected_chunks
        ):
            return "path_evidence_mismatch"
        if self.policy.gate != "source" and not any(
            entity_in_query(query, n.name) for n in all_nodes
        ):
            return "weak_entity_match"
        return None


def protected_supplement(
    *,
    baseline: Sequence[VectorMatch],
    bundles: Sequence[Sequence[VectorMatch]],
    token_count: Callable[[VectorChunk], int],
    policy: GraphSupplementPolicy,
) -> tuple[list[VectorMatch], dict[str, Any]]:
    """Never remove/reorder protected text hits; source bundles are atomic.

    Top5 is protected, not Top10: optional additions can displace the text tail.
    protect=10 is a no-eviction control that fills only genuinely spare slots/tokens.
    """
    result = list(baseline[: policy.protect])
    sizes = {h.chunk.chunk_id: token_count(h.chunk) for h in baseline}
    if (
        any(n < 0 for n in sizes.values())
        or sum(sizes[h.chunk.chunk_id] for h in baseline) > policy.token_budget
    ):
        raise ValueError("Baseline violates token budget")
    if any(
        n > policy.max_per_document for n in Counter(h.chunk.document_id for h in baseline).values()
    ):
        raise ValueError("Baseline violates document budget")
    selected = {h.chunk.chunk_id for h in result}
    tokens = sum(sizes[h.chunk.chunk_id] for h in result)
    counts = Counter(h.chunk.document_id for h in result)
    added: list[str] = []
    rejections: Counter[str] = Counter()
    for bundle in bundles:
        new = list(
            {h.chunk.chunk_id: h for h in bundle if h.chunk.chunk_id not in selected}.values()
        )
        if not new:
            continue
        if len(added) + len(new) > policy.max_additions or len(result) + len(new) > 10:
            rejections["addition_slots"] += 1
            continue
        projected = counts + Counter(h.chunk.document_id for h in new)
        if any(n > policy.max_per_document for n in projected.values()):
            rejections["document_cap"] += 1
            continue
        if any(
            same_source_scope(h.chunk, old.chunk) and _covered_source(h, old)
            for i, h in enumerate(new)
            for old in [*result, *new[:i]]
        ):
            rejections["contained_source"] += 1
            continue
        costs = [token_count(h.chunk) for h in new]
        if any(n < 0 for n in costs):
            raise ValueError("Negative token cost")
        if tokens + sum(costs) > policy.token_budget:
            rejections["token_budget"] += 1
            continue
        result.extend(new)
        selected.update(h.chunk.chunk_id for h in new)
        added.extend(h.chunk.chunk_id for h in new)
        counts, tokens = projected, tokens + sum(costs)
    for h in baseline[policy.protect :]:
        if h.chunk.chunk_id in selected or len(result) >= 10:
            continue
        cost = sizes[h.chunk.chunk_id]
        if (
            counts[h.chunk.document_id] >= policy.max_per_document
            or tokens + cost > policy.token_budget
        ):
            continue
        result.append(h)
        selected.add(h.chunk.chunk_id)
        counts[h.chunk.document_id] += 1
        tokens += cost
    assert [h.chunk.chunk_id for h in result[: min(policy.protect, len(baseline))]] == [
        h.chunk.chunk_id for h in baseline[: policy.protect]
    ]
    return result, dict(
        added=added,
        displaced_text_tail=[
            h.chunk.chunk_id for h in baseline if h.chunk.chunk_id not in selected
        ],
        context_tokens=tokens,
        budget_rejections=dict(rejections),
        protected_prefix_identical=True,
    )


class GatedGraphRetriever:
    """Optional wrapper around the unchanged primary text retriever; fails closed to text."""

    def __init__(
        self,
        *,
        text: CandidateSearch,
        graph: GraphStore,
        load_chunks: Callable[[Sequence[str]], dict[str, VectorChunk]],
        token_count: Callable[[VectorChunk], int],
        policy: GraphSupplementPolicy | None = None,
    ) -> None:
        self.text, self.graph, self.load, self.count, self.policy = (
            text,
            graph,
            load_chunks,
            token_count,
            policy or GraphSupplementPolicy(),
        )
        self.last_trace: dict[str, Any] = {}

    @property
    def backend_name(self) -> str:
        return (
            f"{self.text.backend_name}:protected_graph@1:{self.policy.gate}:{self.policy.protect}"
        )

    def search(
        self, query: str, *, top_k: int = 5, active_version_ids: frozenset[str] | None = None
    ) -> list[VectorMatch]:
        if not query.strip() or not 1 <= top_k <= 10:
            raise ValueError("Query and top_k1..10 required")
        self.last_trace = dict(
            policy=self.backend_name, graph_requested=0, graph_returned=0, reason="no_active_scope"
        )
        if not active_version_ids:
            return []
        baseline = self.text.search(query, top_k=10, active_version_ids=active_version_ids)
        scope = consensus_scope(baseline)
        if top_k <= 5 or scope is None:
            self.last_trace["reason"] = "protected_top5" if top_k <= 5 else "no_source_consensus"
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
            gate = SourceGate(
                nodes=nodes, edges=edges, chunks=chunks, scope=scope, policy=self.policy
            )
            by_id = {h.chunk.chunk_id: h for h in raw}
            protected = {h.chunk.chunk_id for h in baseline[: self.policy.protect]}
            bundles = []
            for h in raw:
                for path in gate.paths(query, h):
                    ids = path["source_chunks"]
                    if any(k not in by_id and k not in protected for k in ids):
                        gate.rejected["incomplete_candidate_bundle"] += 1
                        continue
                    bundle = []
                    for k in ids:
                        if k in protected:
                            continue
                        source = chunks[k]
                        bundle.append(
                            VectorMatch(
                                replace(
                                    source,
                                    metadata={
                                        **source.metadata,
                                        "graph_supplement": {
                                            "policy": "protected_graph@1",
                                            "path": path,
                                            "certainty_is_not_probability": True,
                                        },
                                    },
                                ),
                                by_id[k].score,
                            )
                        )
                    bundles.append(bundle)
            final, trace = protected_supplement(
                baseline=baseline, bundles=bundles, token_count=self.count, policy=self.policy
            )
            self.last_trace.update(
                trace, gate_rejections=dict(gate.rejected), eligible_bundles=len(bundles)
            )
            return final[:top_k]
        # Optional graph adapters have backend-specific exception classes.
        # The primary text call remains outside this fallback boundary.
        except Exception as error:
            self.last_trace.update(
                reason="graph_unavailable_text_fallback", error_type=type(error).__name__
            )
            return baseline[:top_k]
