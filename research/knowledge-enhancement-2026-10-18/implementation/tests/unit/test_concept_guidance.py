from aerodiagnosis.ports import GraphEdge, GraphNode
from aerodiagnosis.retrieval.concepts import concept_record, guided_edges


def node(identity, name, kind="Failure", version="v1", role="unknown"):
    return GraphNode(identity, name, kind, "", "doc", version, {"role": role})


def test_physical_unknown_roles_and_versions_never_merge():
    first = node("a", "压气机", "Component")
    assert (
        concept_record(first)["concept_id"]
        != concept_record(node("b", "压气机", "Component"))["concept_id"]
    )
    assert (
        concept_record(node("a", "裂纹"))["concept_id"]
        != concept_record(node("b", "裂纹", version="v2"))["concept_id"]
    )
    assert (
        concept_record(node("a", "裂纹", role="occurrence"))["concept_id"]
        != concept_record(node("b", "裂纹", role="comparison"))["concept_id"]
    )


def test_memberships_discover_real_assertions_without_inventing_relations():
    nodes = [node("a", "疲劳裂纹"), node("b", "疲劳裂纹"), node("c", "功率丧失")]
    edge = GraphEdge("e", "b", "c", "CAUSES", "doc", "v1", 1, {})
    result = guided_edges("疲劳裂纹", nodes, [edge], [concept_record(n) for n in nodes])
    assert result[0][0] is edge
    assert result[0][2]
    assert len(result) == 1


def test_one_assertion_then_concept_finds_downstream_source_without_claiming_causality():
    nodes = [
        node("a", "疲劳裂纹"),
        node("b", "叶片断裂"),
        node("c", "叶片断裂"),
        node("d", "功率丧失"),
    ]
    edges = [
        GraphEdge("first", "a", "b", "CAUSES", "doc", "v1", 1, {}),
        GraphEdge("second", "c", "d", "CAUSES", "doc", "v1", 1, {}),
    ]
    result = guided_edges("疲劳裂纹", nodes, edges, [concept_record(n) for n in nodes])
    assert {r[0].edge_id for r in result} == {"first", "second"}
    assert next(r for r in result if r[0].edge_id == "second")[1] <= 0.5
    different = [*nodes[:2], node("c", "叶片断裂", role="comparison"), nodes[3]]
    scoped = guided_edges("疲劳裂纹", different, edges, [concept_record(n) for n in different])
    assert {r[0].edge_id for r in scoped} == {"first"}
