from dataclasses import replace

import pytest

from aerodiagnosis.ports import GraphEdge, GraphNode, VectorChunk
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates, entity_identity


class Graph:
    backend_name = "memory-test"

    def __init__(self, nodes, edges):
        self.nodes, self.edges = nodes, edges

    def snapshot(self, **kwargs):
        return self.nodes, self.edges


def fixture():
    chunks, nodes, edges = {}, [], []
    for i in range(3):
        text = f"state{i} causes state{i + 1}"
        c = VectorChunk(
            str(i),
            "d",
            "v",
            text,
            {
                "locator": {
                    "coordinates": {
                        "start": i * 100,
                        "end": i * 100 + len(text),
                        "canonical_sha256": "sha",
                    }
                }
            },
        )
        chunks[c.chunk_id] = c
        proof = {
            "chunk_id": c.chunk_id,
            "document_id": "d",
            "canonical_start": i * 100,
            "canonical_end": i * 100 + len(text),
            "canonical_sha256": "sha",
            "evidence_text": text,
        }
        for j in (i, i + 1):
            n = GraphNode(
                f"{i}-{j}",
                f"state{j}",
                "Fault",
                "",
                "document:d",
                "v",
                {
                    "entity_link": {
                        **proof,
                        "surface": f"state{j}",
                        "role": f"role{j}",
                        "kind": "Fault",
                        "review_status": "source_verified_nonexpert",
                    }
                },
            )
            nodes.append(n)
        edges.append(
            GraphEdge(
                str(i),
                f"{i}-{i}",
                f"{i}-{i + 1}",
                "CAUSES",
                "document:d",
                "v",
                properties={
                    **proof,
                    "validation_status": "necessary_source_checks_passed",
                    "certainty": "possible" if i == 1 else "established",
                },
            )
        )
    return chunks, nodes, edges


def test_resolves_source_mentions_and_returns_three_distinct_original_chunks():
    chunks, nodes, edges = fixture()
    assert entity_identity(nodes[1], chunks) == entity_identity(nodes[2], chunks)
    route = GraphPathCandidates(
        Graph(nodes, edges), lambda ids: {i: chunks[i] for i in ids if i in chunks}
    )
    hits = route.search("state0 state3", top_k=10, active_version_ids=frozenset({"v"}))
    assert {h.chunk.chunk_id for h in hits} == {"0", "1", "2"}
    paths = [p for h in hits for p in h.chunk.metadata["graph_retrieval"]["paths"]]
    long = next(p for p in paths if p["hops"] == 3)
    assert long["edge_ids"] == ["0", "1", "2"]
    assert long["certainty"] == ["established", "possible", "established"]
    assert long["not_a_composed_causal_conclusion"]
    assert all(h.chunk.content == chunks[h.chunk.chunk_id].content for h in hits)


@pytest.mark.parametrize(
    "field,value",
    [
        ("role", "unknown"),
        ("surface", "fabricated"),
        ("review_status", "pending"),
        ("kind", "Engine"),
        ("document_id", "other"),
        ("canonical_sha256", "wrong"),
        ("canonical_start", 99),
    ],
)
def test_unsupported_identity_never_merges(field, value):
    chunks, nodes, _ = fixture()
    proof = {**nodes[1].properties["entity_link"], field: value}
    node = replace(nodes[1], properties={"entity_link": proof})
    assert entity_identity(node, chunks).startswith("mention:")


def test_role_and_version_boundaries_and_direction():
    chunks, nodes, edges = fixture()
    other = replace(nodes[2], version_id="other")
    assert entity_identity(other, chunks) != entity_identity(nodes[1], chunks)
    proof = {**nodes[2].properties["entity_link"], "role": "different_component"}
    assert entity_identity(
        replace(nodes[2], properties={"entity_link": proof}), chunks
    ) != entity_identity(nodes[1], chunks)
    reversed_edge = replace(edges[1], source_id=edges[1].target_id, target_id=edges[1].source_id)
    route = GraphPathCandidates(
        Graph(nodes, [edges[0], reversed_edge, edges[2]]), lambda ids: chunks
    )
    hits = route.search("state0", active_version_ids=frozenset({"v"}))
    assert not any(
        p["hops"] == 3 for h in hits for p in h.chunk.metadata["graph_retrieval"]["paths"]
    )


@pytest.mark.parametrize(
    "change",
    [
        {"polarity": "negative"},
        {"deprecated": True},
        {"canonical_sha256": "forged"},
        {"evidence_text": "forged"},
    ],
)
def test_bad_edge_cannot_complete_path(change):
    chunks, nodes, edges = fixture()
    edges[1] = replace(edges[1], properties={**edges[1].properties, **change})
    route = GraphPathCandidates(Graph(nodes, edges), lambda ids: chunks)
    hits = route.search("state0 state3", active_version_ids=frozenset({"v"}))
    assert not any(
        p["hops"] == 3 for h in hits for p in h.chunk.metadata["graph_retrieval"]["paths"]
    )


def test_explicit_scope_and_visit_budget():
    chunks, nodes, edges = fixture()
    route = GraphPathCandidates(Graph(nodes, edges), lambda ids: chunks, max_visits=1)
    assert route.search("state0") == []
    route.search("state0 state3", active_version_ids=frozenset({"v"}))
    assert route.last_trace["visits"] == 1
    assert route.last_trace["budget_exhausted"]


def test_supplementary_evidence_is_validated_and_counted_as_original_chunk():
    chunks, nodes, edges = fixture()
    extra = VectorChunk(
        "extra",
        "d",
        "v",
        "recommended",
        {"locator": {"coordinates": {"start": 400, "end": 411, "canonical_sha256": "sha"}}},
    )
    chunks["extra"] = extra
    proof = {
        "chunk_id": "extra",
        "canonical_start": 400,
        "canonical_end": 411,
        "canonical_sha256": "sha",
        "evidence_text": "recommended",
    }
    edges[0] = replace(edges[0], properties={**edges[0].properties, "supporting_evidence": [proof]})
    route = GraphPathCandidates(Graph(nodes, edges), lambda ids: chunks)
    hits = route.search("state0", top_k=10, active_version_ids=frozenset({"v"}))
    assert "extra" in {h.chunk.chunk_id for h in hits}
    broken = replace(
        edges[0],
        properties={
            **edges[0].properties,
            "supporting_evidence": [{**proof, "evidence_text": "forged"}],
        },
    )
    route = GraphPathCandidates(Graph(nodes, [broken, *edges[1:]]), lambda ids: chunks)
    hits = route.search("state0", top_k=10, active_version_ids=frozenset({"v"}))
    assert not hits


def test_path_configuration_is_opt_in(monkeypatch, tmp_path):
    from aerodiagnosis.config import RuntimeSettings

    assert (
        RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "db").manual_graph_mode
        == "entity"
    )
    assert (
        RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "db").manual_graph_weight
        == 0
    )
    monkeypatch.setenv("AERODIAGNOSIS_MANUAL_GRAPH_MODE", "path")
    assert RuntimeSettings.from_env(tmp_path).manual_graph_mode == "path"
    with pytest.raises(ValueError):
        RuntimeSettings(
            runtime_dir=tmp_path, database_path=tmp_path / "db", manual_graph_mode="unbounded"
        )
