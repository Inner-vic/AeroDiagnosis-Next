from dataclasses import replace

import pytest
from tests.unit.test_gated_graph import Broken, Text, baseline, data
from tests.unit.test_graph_paths import Graph

from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.graph_supplement_v2 import (
    DirectedGraphSupport,
    append_support,
    focus_reason,
    query_focus,
)


def test_direction_anchors_and_boundary_abstention():
    path = {"entities": ["轴承润滑不足", "轴承过热损坏", "封严失效泄漏"]}
    assert focus_reason("轴承过热损坏的原因是什么?", path) == "root_requires_terminal_anchor"
    assert focus_reason("封严失效泄漏的根因是什么?", path) is None
    assert focus_reason("轴承润滑不足后如何导致泄漏?请说明机理", path) is None
    assert (
        focus_reason("封严失效泄漏后如何导致烟雾?请说明机理", path)
        == "mechanism_requires_initial_anchor"
    )
    assert query_focus("根本原因能确定到什么程度?有哪些反证?").direction == "boundary"
    assert focus_reason("轴承过热损坏原因有哪些不确定性", path) == "uncertainty_boundary_abstain"


def test_append_removes_quota_collision_without_evicting_primary():
    text = baseline()
    text = [
        replace(h, chunk=replace(h.chunk, document_id="d")) if i < 5 else h
        for i, h in enumerate(text)
    ]
    extras = [
        VectorMatch(VectorChunk("e" + str(i), "d", "v", "extra" + str(i)), 0.5) for i in range(3)
    ]
    out, trace = append_support(
        baseline=text, bundles=[[extras[0], extras[1]], [extras[2]]], token_count=lambda c: 1
    )
    assert out[:10] == text
    assert len(out) == 12 and trace["added"] == ["e0", "e1"]
    assert trace["displaced_text_tail"] == []
    assert trace["budget_rejections"]["addition_slots"] == 1


def test_append_atomic_budget_and_invalid_negative_cost():
    text = baseline()
    extras = [
        VectorMatch(VectorChunk("e" + str(i), "d", "v", "extra" + str(i)), 0.5) for i in range(2)
    ]
    out, trace = append_support(
        baseline=text, bundles=[extras], token_count=lambda c: 1, token_budget=11
    )
    assert out == text and trace["budget_rejections"]["token_budget"] == 1
    with pytest.raises(ValueError):
        append_support(baseline=text, bundles=[extras], token_count=lambda c: -1)


def test_boundary_never_calls_graph_and_primary_failure_not_hidden():
    text, graph = Text(), Broken()
    r = DirectedGraphSupport(
        text=text, graph=graph, load_chunks=lambda ids: {}, token_count=lambda c: 1
    )
    assert (
        r.search("轴承失效的原因有哪些不确定性?", active_version_ids=frozenset({"v"})) == text.hits
    )
    assert graph.calls == 0
    assert r.last_trace["reason"] == "uncertainty_boundary_abstain"
    assert r.search("为什么state0会导致state1?", active_version_ids=frozenset({"v"})) == text.hits
    assert r.last_trace["reason"] == "graph_unavailable_text_fallback"


def test_source_verified_graph_append_and_focus_ablation():
    chunks, nodes, edges = data()
    text = Text()
    r = DirectedGraphSupport(
        text=text,
        graph=Graph(nodes, edges),
        load_chunks=lambda ids: {k: chunks[k] for k in ids if k in chunks},
        token_count=lambda c: 1,
    )
    hits = r.search("state0后如何导致state1?请说明机理", active_version_ids=frozenset({"v"}))
    assert hits[:10] == text.hits and r.last_trace["added"]
    assert len(hits) <= 12
    assert (
        r.search("state0后如何导致state1?请说明机理", top_k=10, active_version_ids=frozenset({"v"}))
        == text.hits
    )
    edges[0] = replace(edges[0], properties={**edges[0].properties, "canonical_sha256": "forged"})
    broken = DirectedGraphSupport(
        text=text,
        graph=Graph(nodes, edges[:1]),
        load_chunks=lambda ids: chunks,
        token_count=lambda c: 1,
    )
    assert (
        broken.search("state0后如何导致state1?请说明机理", active_version_ids=frozenset({"v"}))
        == text.hits
    )
