"""Experimental query-scoped atomic path selection; no answer labels accepted."""

from __future__ import annotations

import math
import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from aerodiagnosis.application.fault_graph import graph_edge_is_active
from aerodiagnosis.ports import GraphEdge, GraphNode, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.graph_paths import _quote_valid, entity_identity

IDENTIFIER = re.compile(r"(?<![A-Za-z0-9])[A-Z]{2,6}-[0-9]{3,8}(?![A-Za-z0-9])", re.I)
CAUSAL = frozenset({"CAUSES", "CONTRIBUTES_TO", "HAS_MECHANISM"})


def identifier_scope(query: str, chunks: Mapping[str, VectorChunk]) -> frozenset[str] | None:
    """None means no identifier constraint; empty means an explicit unknown identifier."""
    requested = {s.upper() for s in IDENTIFIER.findall(query)}
    if not requested:
        return None
    return frozenset(
        c.version_id
        for c in chunks.values()
        if requested & {s.upper() for s in IDENTIFIER.findall(c.content)}
    )


def causal_intent(query: str) -> bool:
    if any(
        s in query
        for s in ("能否", "是否", "是不是", "足以", "确定", "反推", "仅凭", "一定", "归咎", "边界")
    ):
        return False
    return any(
        s in query
        for s in (
            "因果",
            "机理",
            "机制",
            "路径",
            "作用链",
            "导致",
            "引起",
            "引发",
            "传导",
            "传递",
            "演变",
            "中间环节",
        )
    ) or ("从" in query and "到" in query)


def endpoint_match(query: str, name: str) -> bool:
    query_ids = {s.upper() for s in IDENTIFIER.findall(query)}
    entity_ids = {s.upper() for s in IDENTIFIER.findall(name)}
    if query_ids and entity_ids and not entity_ids <= query_ids:
        return False
    core = IDENTIFIER.sub("", name).strip().removeprefix("的")
    return len(core) >= 4 and core in query


@dataclass(frozen=True)
class PathBundle:
    chunks: tuple[str, ...]
    edges: tuple[str, ...]
    score: float
    hops: int


def verified_bundles(
    query: str,
    graph_matches: Sequence[VectorMatch],
    *,
    nodes: Mapping[str, GraphNode],
    edges: Mapping[str, GraphEdge],
    chunks: Mapping[str, VectorChunk],
    max_hops: int = 3,
) -> tuple[list[PathBundle], dict[str, int]]:
    rejected: Counter[str] = Counter()
    if not causal_intent(query):
        return [], {"noncausal_or_boundary": 1}
    available = {m.chunk.chunk_id for m in graph_matches}
    found: dict[tuple[str, ...], PathBundle] = {}
    identities = {key: entity_identity(n, dict(chunks)) for key, n in nodes.items()}
    for match in graph_matches:
        if not math.isfinite(match.score):
            rejected["nonfinite_score"] += 1
            continue
        for path in match.chunk.metadata.get("graph_retrieval", {}).get("paths", []):
            ids = path.get("edge_ids", [])
            if (
                not 2 <= len(ids) <= max_hops
                or len(set(ids)) != len(ids)
                or path.get("direction") != "forward"
            ):
                rejected["shape"] += 1
                continue
            selected = [edges.get(key) for key in ids]
            if any(e is None for e in selected):
                rejected["missing_edge"] += 1
                continue
            selected = [e for e in selected if e is not None]
            proof_ids: list[str] = []
            expected_names: list[str] = []
            scope = None
            previous = None
            reason = None
            for edge in selected:
                assert edge is not None
                source, target = nodes.get(edge.source_id), nodes.get(edge.target_id)
                p = edge.properties
                if (
                    source is None
                    or target is None
                    or not graph_edge_is_active(edge)
                    or edge.relation not in CAUSAL
                    or p.get("polarity") != "positive"
                    or p.get("certainty") not in {"established", "observed"}
                    or p.get("validation_status") != "necessary_source_checks_passed"
                ):
                    reason = "edge_semantics_or_authority"
                    break
                edge_scope = (edge.source_ref, edge.version_id)
                if scope is not None and scope != edge_scope:
                    reason = "mixed_scope"
                    break
                scope = edge_scope
                if any((n.source_ref, n.version_id) != scope for n in (source, target)):
                    reason = "entity_scope"
                    break
                left, right = identities[source.node_id], identities[target.node_id]
                if (
                    not left.startswith("entity:")
                    or not right.startswith("entity:")
                    or (previous is not None and previous != left)
                ):
                    reason = "disconnected_or_unverified"
                    break
                previous = right
                if not expected_names:
                    expected_names.append(source.name)
                expected_names.append(target.name)
                for proof in [p, *p.get("supporting_evidence", [])]:
                    chunk = chunks.get(str(proof.get("chunk_id", "")))
                    if (
                        chunk is None
                        or ("document:" + chunk.document_id, chunk.version_id) != scope
                        or not _quote_valid(chunk, proof)
                    ):
                        reason = "invalid_quote"
                        break
                    proof_ids.append(chunk.chunk_id)
                if reason:
                    break
            if reason:
                rejected[reason] += 1
                continue
            ordered = tuple(dict.fromkeys(proof_ids))
            if (
                path.get("source_chunks") != list(ordered)
                or path.get("entities") != expected_names
                or path.get("hops") != len(ids)
                or path.get("relations") != [e.relation for e in selected if e is not None]
                or path.get("certainty")
                != [e.properties.get("certainty", "unknown") for e in selected if e is not None]
                or path.get("qualifiers")
                != [e.properties.get("qualifiers", {}) for e in selected if e is not None]
            ):
                rejected["metadata_mismatch"] += 1
                continue
            if not endpoint_match(query, expected_names[0]) or not endpoint_match(
                query, expected_names[-1]
            ):
                rejected["endpoint_mismatch"] += 1
                continue
            if not set(ordered) <= available:
                rejected["outside_graph_candidate_budget"] += 1
                continue
            key = tuple(ids)
            candidate = PathBundle(ordered, key, float(match.score), len(ids))
            if key not in found or candidate.score > found[key].score:
                found[key] = candidate
    return sorted(found.values(), key=lambda b: (-b.score, -b.hops, b.edges)), dict(rejected)


def atomic_select(
    baseline: Sequence[VectorMatch],
    bundles: Sequence[PathBundle],
    *,
    chunks: Mapping[str, VectorChunk],
    token_count: Callable[[VectorChunk], int],
    protect: int = 1,
    top_k: int = 10,
    token_budget: int = 4096,
    per_document: int = 5,
) -> tuple[list[VectorMatch], dict[str, Any]]:
    """Promote one complete verified path after a protected prefix, or promote none.

    Preserve the prefix exactly. A rejected path consumes no slots. This policy
    guarantees atomic insertion, not preservation of every original Top10 hit.
    """
    if not 0 <= protect <= top_k or top_k < 1 or token_budget < 1 or per_document < 1:
        raise ValueError("Invalid atomic selection policy")
    if len({m.chunk.chunk_id for m in baseline}) != len(baseline):
        raise ValueError("Duplicate baseline authority")
    sizes = [token_count(m.chunk) for m in baseline]
    if (
        len(baseline) > top_k
        or any(s < 0 for s in sizes)
        or sum(sizes) > token_budget
        or any(n > per_document for n in Counter(m.chunk.document_id for m in baseline).values())
    ):
        raise ValueError("Baseline violates budget")
    result = list(baseline[:protect])
    counts = Counter(m.chunk.document_id for m in result)
    cost = sum(token_count(m.chunk) for m in result)
    if cost > token_budget or any(n > per_document for n in counts.values()):
        raise ValueError("Protected prefix violates budget")
    rejected: Counter[str] = Counter()
    adopted = None
    for bundle in bundles:
        ids = tuple(dict.fromkeys(bundle.chunks))
        if not ids or any(k not in chunks for k in ids):
            rejected["missing_bundle_chunk"] += 1
            continue
        existing = {m.chunk.chunk_id for m in result}
        new = [chunks[k] for k in ids if k not in existing]
        sizes = [token_count(c) for c in new]
        if any(s < 0 for s in sizes):
            raise ValueError("Negative token count")
        if (
            len(result) + len(new) > top_k
            or cost + sum(sizes) > token_budget
            or any(
                n > per_document for n in (counts + Counter(c.document_id for c in new)).values()
            )
        ):
            rejected["atomic_budget"] += 1
            continue
        result += [VectorMatch(c, bundle.score) for c in new]
        counts.update(c.document_id for c in new)
        cost += sum(sizes)
        adopted = bundle
        break
    if adopted is None:
        return list(baseline[:top_k]), dict(
            adopted=False, budget_rejections=dict(rejected), protected_prefix=protect
        )
    seen = {m.chunk.chunk_id for m in result}
    for m in baseline:
        if m.chunk.chunk_id in seen:
            continue
        size = token_count(m.chunk)
        if size < 0:
            raise ValueError("Negative token count")
        if len(result) >= top_k:
            break
        if counts[m.chunk.document_id] >= per_document or cost + size > token_budget:
            continue
        result.append(m)
        seen.add(m.chunk.chunk_id)
        counts[m.chunk.document_id] += 1
        cost += size
    assert set(adopted.chunks) <= {m.chunk.chunk_id for m in result}
    return result, dict(
        adopted=True,
        edge_ids=list(adopted.edges),
        path_chunks=list(adopted.chunks),
        path_hops=adopted.hops,
        protected_prefix=protect,
        context_tokens=cost,
        budget_rejections=dict(rejected),
        promoted_path_complete_rank=max(
            i for i, m in enumerate(result, 1) if m.chunk.chunk_id in adopted.chunks
        ),
    )
