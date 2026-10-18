import json
from dataclasses import replace
from pathlib import Path

import pytest

from aerodiagnosis.ports import GraphEdge, GraphNode, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.atomic_path_fusion import (
    PathBundle,
    atomic_select,
    causal_intent,
    endpoint_match,
    identifier_scope,
    verified_bundles,
)
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates

ROOT = Path(
    "D:/codex/worktrees/knowledge-upgrade-integration/AeroDiagnosis/evaluation/controlled_scenario_v1"
)


@pytest.fixture
def example():
    data = json.loads((ROOT / "corpus.json").read_text(encoding="utf8"))
    g = json.loads((ROOT / "graph.json").read_text(encoding="utf8"))
    doc = data["documents"][0]
    chunks = {
        c["chunk_id"]: VectorChunk(**c)
        for c in data["chunks"]
        if c["document_id"] == doc["document_id"]
    }
    nodes = {
        n["node_id"]: GraphNode(**n) for n in g["nodes"] if n["version_id"] == doc["version_id"]
    }
    edges = {
        e["edge_id"]: GraphEdge(**e) for e in g["edges"] if e["version_id"] == doc["version_id"]
    }

    class Graph:
        backend_name = "test_source"

        def snapshot(self, *, active_version_ids=None, limit=2000):
            return list(nodes.values()), list(edges.values())

    root = nodes[edges[doc["document_id"] + ":rule:0"].source_id].name
    end = nodes[edges[doc["document_id"] + ":rule:2"].target_id].name
    query = f"从{root}到{end},请说明物理因果路径。"
    matches = GraphPathCandidates(Graph(), lambda ks: {k: chunks[k] for k in ks}).search(
        query, top_k=50, active_version_ids=frozenset({doc["version_id"]})
    )
    return dict(
        doc=doc,
        chunks=chunks,
        nodes=nodes,
        edges=edges,
        query=query,
        matches=matches,
        root=root,
        end=end,
    )


def bundles(e, **overrides):
    kw = {k: e[k] for k in ["nodes", "edges", "chunks"]}
    kw.update(overrides)
    return verified_bundles(e["query"], e["matches"], **kw)


def test_known_and_unknown_scope_are_distinct(example):
    assert identifier_scope("查询一般异常", example["chunks"]) is None
    assert identifier_scope(example["doc"]["bench"].lower(), example["chunks"]) == frozenset(
        {example["doc"]["version_id"]}
    )
    assert identifier_scope("QB-00000", example["chunks"]) == frozenset()


@pytest.mark.parametrize(
    "q", ["物理因果如何联系", "从甲故障到乙结果", "怎样经由中间环节产生结果", "请还原传递过程"]
)
def test_natural_causal_intents(q):
    assert causal_intent(q)


@pytest.mark.parametrize(
    "q", ["是不是就一定导致它", "仅凭结果足以确定原因吗", "能否沿因果路径逆推", "采样周期是多少"]
)
def test_boundary_or_parameter_is_not_positive_path(q):
    assert not causal_intent(q)


def test_endpoint_rejects_wrong_identifier(example):
    assert endpoint_match(example["query"], example["root"])
    assert not endpoint_match(
        example["query"], example["root"].replace(example["doc"]["bench"], "QB-00000")
    )


def test_valid_source_path_is_complete(example):
    found, _ = bundles(example)
    assert found and found[0].hops == 3 and len(found[0].chunks) == 3


def test_quote_tampering_is_rejected(example):
    edges = dict(example["edges"])
    key = example["doc"]["document_id"] + ":rule:1"
    edges[key] = replace(
        edges[key], properties={**edges[key].properties, "evidence_text": "not in original source"}
    )
    found, rejected = bundles(example, edges=edges)
    assert not found and rejected["invalid_quote"] > 0


def test_uncertain_edge_is_not_promoted(example):
    edges = dict(example["edges"])
    key = example["doc"]["document_id"] + ":rule:1"
    edges[key] = replace(edges[key], properties={**edges[key].properties, "certainty": "possible"})
    assert not bundles(example, edges=edges)[0]


def test_missing_middle_candidate_rejects_entire_path(example):
    full, _ = bundles(example)
    middle = full[0].chunks[1]
    found, rejected = verified_bundles(
        example["query"],
        [m for m in example["matches"] if m.chunk.chunk_id != middle],
        nodes=example["nodes"],
        edges=example["edges"],
        chunks=example["chunks"],
    )
    assert not found and rejected["outside_graph_candidate_budget"] > 0


def test_disconnected_edge_cannot_be_composed(example):
    edges = dict(example["edges"])
    key = example["doc"]["document_id"] + ":rule:1"
    edges[key] = replace(
        edges[key], source_id=edges[example["doc"]["document_id"] + ":rule:2"].target_id
    )
    assert not bundles(example, edges=edges)[0]


def test_whole_path_after_protected_text(example):
    found, _ = bundles(example)
    path = found[0]
    other = next(c for k, c in example["chunks"].items() if k not in path.chunks)
    baseline = [VectorMatch(other, 0.9), VectorMatch(example["chunks"][path.chunks[0]], 0.8)]
    result, trace = atomic_select(
        baseline, found, chunks=example["chunks"], token_count=lambda c: 1
    )
    assert result[0] == baseline[0]
    assert set(path.chunks) <= {m.chunk.chunk_id for m in result[:5]}
    assert trace["adopted"] and trace["promoted_path_complete_rank"] == 4


def test_insufficient_token_budget_never_adds_partial_path(example):
    found, _ = bundles(example)
    path = found[0]
    baseline = [VectorMatch(example["chunks"][path.chunks[0]], 0.8)]
    result, trace = atomic_select(
        baseline, found, chunks=example["chunks"], token_count=lambda c: 1, token_budget=2
    )
    assert (
        result == baseline
        and not trace["adopted"]
        and trace["budget_rejections"]["atomic_budget"] > 0
    )


def test_per_document_budget_is_atomic(example):
    found, _ = bundles(example)
    path = found[0]
    other = next(c for k, c in example["chunks"].items() if k not in path.chunks)
    baseline = [VectorMatch(other, 0.9)]
    result, trace = atomic_select(
        baseline, found, chunks=example["chunks"], token_count=lambda c: 1, per_document=3
    )
    assert result == baseline and not trace["adopted"]


def test_bad_first_bundle_does_not_consume_good_bundle_slots(example):
    found, _ = bundles(example)
    bad = PathBundle(("nonexistent",), ("missing",), 1.0, 2)
    baseline = [VectorMatch(example["chunks"][found[0].chunks[0]], 0.8)]
    result, trace = atomic_select(
        baseline, [bad, *found], chunks=example["chunks"], token_count=lambda c: 1
    )
    assert trace["adopted"] and set(found[0].chunks) <= {m.chunk.chunk_id for m in result}


def test_empty_paths_preserve_baseline(example):
    baseline = [VectorMatch(c, 0.5) for c in list(example["chunks"].values())[:5]]
    assert (
        atomic_select(baseline, [], chunks=example["chunks"], token_count=lambda c: 1)[0]
        == baseline
    )


def test_invalid_baseline_budget_fails_closed(example):
    baseline = [VectorMatch(c, 0.5) for c in list(example["chunks"].values())[:5]]
    with pytest.raises(ValueError, match="Baseline violates budget"):
        atomic_select(
            baseline, [], chunks=example["chunks"], token_count=lambda c: 100, token_budget=10
        )
