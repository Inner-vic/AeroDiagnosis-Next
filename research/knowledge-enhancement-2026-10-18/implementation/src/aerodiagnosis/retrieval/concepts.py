"""Source-scoped exact concept membership; never merge event assertions.

Inspired by Neo4j GraphRAG's exact-match resolver. Unlike its destructive
APOC merge, membership only discovers source candidates and proves no causality.
"""

from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from collections.abc import Sequence
from math import log1p

from aerodiagnosis.ports import GraphEdge, GraphNode
from aerodiagnosis.retrieval.chinese import normalize_query, ranked_entities

POLICY = "source-scoped-concepts@1"


def concept_record(node: GraphNode) -> dict[str, str]:
    label = normalize_query(node.name).strip()
    role = str(node.properties.get("role", "unknown"))
    qualifiers = node.properties.get("qualifiers", {})
    signature = json.dumps(qualifiers, ensure_ascii=False, sort_keys=True)
    # Unknown physical roles cannot identify a shared object. Keep each mention
    # separate until the source supplies an explicit role.
    discriminator = (
        node.node_id if node.kind in {"Engine", "Component"} and role == "unknown" else ""
    )
    key = json.dumps(
        [POLICY, node.version_id, node.kind, label, role, signature, discriminator],
        ensure_ascii=False,
    )
    return {
        "node_id": node.node_id,
        "concept_id": hashlib.sha256(key.encode()).hexdigest(),
        "label": label,
        "kind": node.kind,
        "role": role,
        "version_id": node.version_id,
        "policy": POLICY,
    }


def guided_edges(
    query: str,
    nodes: Sequence[GraphNode],
    edges: Sequence[GraphEdge],
    memberships: Sequence[dict[str, str]],
) -> list[tuple[GraphEdge, float, tuple[str, ...]]]:
    """One membership hop, then original assertions, with inverse hub weighting."""
    known = {n.node_id for n in nodes}
    by_concept: dict[str, list[str]] = defaultdict(list)
    concepts: dict[str, str] = {}
    for row in memberships:
        if row["node_id"] in known:
            by_concept[row["concept_id"]].append(row["node_id"])
            concepts[row["node_id"]] = row["concept_id"]
    signals: dict[str, float] = {}
    paths: dict[str, set[str]] = defaultdict(set)
    seeds = ranked_entities(query, nodes)
    maximum = max((score for _, score in seeds), default=1)
    for node, score in seeds[:32]:
        signals[node.node_id] = max(signals.get(node.node_id, 0), score / maximum)
        key = concepts.get(node.node_id)
        if key is None:
            continue
        members = by_concept[key]
        # Bound traversal fanout and downweight common concepts.
        if len(members) > 32:
            continue
        for other in members:
            signals[other] = max(signals.get(other, 0), score / maximum / (1 + log1p(len(members))))
            paths[other].add(key)
    # One assertion hop plus a concept membership hop can locate a following
    # assertion. These are candidate discovery links, not a composed causal claim.
    propagated: dict[str, float] = {}
    for edge in edges:
        signal = max(signals.get(edge.source_id, 0), signals.get(edge.target_id, 0))
        if signal <= 0:
            continue
        for endpoint in (edge.source_id, edge.target_id):
            key = concepts.get(endpoint)
            if key is None or len(by_concept[key]) > 32:
                continue
            for other in by_concept[key]:
                propagated[other] = max(propagated.get(other, 0), 0.5 * signal)
                paths[other].add(key)
    for identity, signal in propagated.items():
        signals[identity] = max(signals.get(identity, 0), signal)
    result = []
    for edge in edges:
        signal = max(signals.get(edge.source_id, 0), signals.get(edge.target_id, 0))
        if signal > 0:
            result.append(
                (edge, signal, tuple(sorted(paths[edge.source_id] | paths[edge.target_id])))
            )
    return sorted(result, key=lambda item: (-item[1], item[0].edge_id))
