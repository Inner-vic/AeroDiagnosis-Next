from dataclasses import replace

import pytest
from tests.unit.test_graph_paths import Graph, fixture

from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.gated_graph import (
    GatedGraphRetriever,
    GraphSupplementPolicy,
    SourceGate,
    protected_supplement,
)
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates


def data(certainty="established"):
    chunks, nodes, edges = fixture()
    edges = [
        replace(
            e,
            properties={
                **e.properties,
                "polarity": "positive",
                "independent_source_review": "accepted",
                "certainty": certainty,
            },
        )
        for e in edges
    ]
    return chunks, nodes, edges


def check(
    query="为什么state0会导致state1?",
    *,
    change=None,
    node_change=None,
    certainty="established",
    gate="strict",
):
    chunks, nodes, edges = data(certainty)
    graph = Graph(nodes, edges)
    hits = GraphPathCandidates(graph, lambda ids: chunks).search(
        "state0", top_k=25, active_version_ids=frozenset({"v"})
    )
    h = next(h for h in hits if h.chunk.chunk_id == "0")
    if change:
        edges = [
            replace(e, properties={**e.properties, **change}) if e.edge_id == "0" else e
            for e in edges
        ]
    if node_change:
        n = nodes[0]
        nodes[0] = replace(
            n, properties={"entity_link": {**n.properties["entity_link"], **node_change}}
        )
    source = SourceGate(
        nodes=nodes,
        edges=edges,
        chunks=chunks,
        scope=("d", "v"),
        policy=GraphSupplementPolicy(gate=gate),
    )
    return source.paths(query, h), source


def test_confidence_requires_source_entity_and_relation_intent():
    assert check()[0]
    assert not check(query="state0的编号是多少?")[0]
    assert not check(query="为什么发动机失效?")[0]
    assert check(query="为什么发动机失效?", gate="source")[0]


@pytest.mark.parametrize(
    "change",
    [
        {"polarity": "negative"},
        {"polarity": None},
        {"deprecated": True},
        {"independent_source_review": "pending"},
        {"validation_status": "pending"},
        {"document_id": "other"},
        {"canonical_sha256": "forged"},
        {"canonical_start": 99},
        {"evidence_text": "fabricated"},
        {"certainty": "unresolved"},
    ],
)
def test_gate_fails_closed_on_invalid_or_uncertain_edges(change):
    assert not check(change=change)[0]


@pytest.mark.parametrize(
    "proof",
    [
        {"review_status": "pending"},
        {"surface": "invented"},
        {"document_id": "another"},
        {"canonical_sha256": "wrong"},
        {"role": "unknown"},
    ],
)
def test_gate_requires_verified_source_mentions(proof):
    assert not check(node_change=proof)[0]


def test_possible_is_preserved_and_only_allowed_for_explicit_uncertainty_question():
    assert not check(certainty="possible")[0]
    assert not check(certainty="possible", gate="uncertainty")[0]
    paths, _ = check(query="state0是否可能导致state1?", certainty="possible", gate="uncertainty")
    assert paths and paths[0]["certainty"][0] == "possible"


def baseline():
    return [
        VectorMatch(
            VectorChunk("b" + str(i), "d" if i < 3 else "d" + str(i), "v", "text" + str(i)),
            1 - i * 0.01,
        )
        for i in range(10)
    ]


def test_atomic_bundle_protects_prefix_scores_and_obeys_budgets():
    text = baseline()
    extras = [VectorMatch(VectorChunk(str(i), "d", "v", "source" + str(i)), 0.8) for i in range(2)]
    final, trace = protected_supplement(
        baseline=text, bundles=[extras], token_count=lambda c: 1, policy=GraphSupplementPolicy()
    )
    assert final[:5] == text[:5]
    assert [h.chunk.chunk_id for h in final[5:7]] == ["0", "1"]
    assert trace["displaced_text_tail"] == ["b8", "b9"]
    assert len(final) == 10 and trace["context_tokens"] == 10
    rejected, trace = protected_supplement(
        baseline=text,
        bundles=[extras],
        token_count=lambda c: 10 if c.chunk_id in {"0", "1"} else 1,
        policy=GraphSupplementPolicy(token_budget=10),
    )
    assert rejected == text and trace["added"] == []
    assert trace["budget_rejections"]["token_budget"] == 1


def test_no_eviction_control_and_candidate_duplicate_do_not_change_text():
    text = baseline()
    extra = VectorMatch(VectorChunk("g", "d", "v", "extra"), 0.5)
    out, trace = protected_supplement(
        baseline=text,
        bundles=[[text[0]], [extra]],
        token_count=lambda c: 1,
        policy=GraphSupplementPolicy(protect=10),
    )
    assert out == text and trace["added"] == []


class Text:
    backend_name = "text"

    def __init__(self):
        self.hits = baseline()

    def search(self, query, *, top_k, active_version_ids):
        return self.hits[:top_k]


class AdapterUnavailable(Exception):
    pass


class Broken:
    backend_name = "broken"

    def __init__(self, error_type=RuntimeError):
        self.calls = 0
        self.error_type = error_type

    def snapshot(self, **kwargs):
        self.calls += 1
        raise self.error_type("optional backend offline")


@pytest.mark.parametrize("error_type", [RuntimeError, AdapterUnavailable])
def test_runtime_skips_top5_graph_and_falls_back_when_backend_unavailable(error_type):
    text, graph = Text(), Broken(error_type)
    r = GatedGraphRetriever(
        text=text, graph=graph, load_chunks=lambda ids: {}, token_count=lambda c: 1
    )
    assert r.search("why", top_k=5, active_version_ids=frozenset({"v"})) == text.hits[:5]
    assert graph.calls == 0
    assert r.search("why", top_k=10, active_version_ids=frozenset({"v"})) == text.hits
    assert r.last_trace["reason"] == "graph_unavailable_text_fallback"
    assert r.search("why", top_k=10) == []
    assert r.last_trace["reason"] == "no_active_scope"


def test_runtime_integrates_verified_graph_without_reading_labels():
    chunks, nodes, edges = data()
    text = Text()
    r = GatedGraphRetriever(
        text=text,
        graph=Graph(nodes, edges),
        load_chunks=lambda ids: {k: chunks[k] for k in ids if k in chunks},
        token_count=lambda c: 1,
    )
    hits = r.search("为什么state0会导致state1?", top_k=10, active_version_ids=frozenset({"v"}))
    assert hits[:5] == text.hits[:5]
    assert r.last_trace["added"]
    assert len(r.last_trace["added"]) <= 2
    for h in hits:
        if "graph_supplement" in h.chunk.metadata:
            assert h.chunk.content == chunks[h.chunk.chunk_id].content
            assert h.chunk.metadata["graph_supplement"]["path"]["direction"] == "forward"


def test_gate_rejects_forged_path_certainty_without_erasing_qualifiers():
    chunks, nodes, edges = data("possible")
    raw = GraphPathCandidates(Graph(nodes, edges), lambda ids: chunks).search(
        "state0", top_k=25, active_version_ids=frozenset({"v"})
    )
    h = next(h for h in raw if h.chunk.chunk_id == "0")
    paths = [
        {**p, "certainty": ["established"] * p["hops"]}
        for p in h.chunk.metadata["graph_retrieval"]["paths"]
    ]
    forged = replace(
        h.chunk,
        metadata={
            **h.chunk.metadata,
            "graph_retrieval": {**h.chunk.metadata["graph_retrieval"], "paths": paths},
        },
    )
    gate = SourceGate(
        nodes=nodes,
        edges=edges,
        chunks=chunks,
        scope=("d", "v"),
        policy=GraphSupplementPolicy(gate="uncertainty"),
    )
    assert not gate.paths("state0是否可能导致state1?", VectorMatch(forged, h.score))
    assert gate.rejected["path_metadata_mismatch"]


def test_no_source_consensus_does_not_query_graph():
    text, graph = Text(), Broken()
    text.hits = [
        VectorMatch(replace(h.chunk, document_id="unique" + str(i)), h.score)
        for i, h in enumerate(text.hits)
    ]
    r = GatedGraphRetriever(
        text=text, graph=graph, load_chunks=lambda ids: {}, token_count=lambda c: 1
    )
    assert r.search("原因是什么", top_k=10, active_version_ids=frozenset({"v"})) == text.hits
    assert graph.calls == 0 and r.last_trace["reason"] == "no_source_consensus"
