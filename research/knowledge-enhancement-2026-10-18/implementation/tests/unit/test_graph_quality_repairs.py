from __future__ import annotations

from dataclasses import replace

from aerodiagnosis.adapters.docling_structure import structured_records
from aerodiagnosis.application.graph_quality import normalize_graph
from aerodiagnosis.domain import DiagnosisCommand
from aerodiagnosis.ports import GraphEdge, GraphNode, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.chinese import ranked_entities
from aerodiagnosis.tools.diagnostic import DiagnosticToolset


class Tokenizer:
    def encode(self, text, **kwargs):
        return list(text)


def test_cross_page_continuation_survives_heading_and_has_both_pages():
    data = {
        "texts": [
            {
                "self_ref": "p1",
                "label": "text",
                "text": "Please remove the",
                "prov": [{"page_no": 1}],
            },
            {"self_ref": "h", "label": "section_header", "text": "CHAPTER 6"},
            {
                "self_ref": "p2",
                "label": "text",
                "text": "vibration sensitive components and inspect them.",
                "prov": [{"page_no": 2}],
            },
        ],
        "body": {"children": [{"$ref": "p1"}, {"$ref": "h"}, {"$ref": "p2"}]},
    }
    records = structured_records(data, Tokenizer(), 400)
    assert records[0][0] == "Please remove the vibration sensitive components and inspect them."
    assert records[0][1]["pages"] == [1, 2]


def test_table_preserves_header_and_whole_row_even_over_budget():
    cells = [
        {
            "start_row_offset_idx": r,
            "end_row_offset_idx": r + 1,
            "start_col_offset_idx": c,
            "end_col_offset_idx": c + 1,
            "text": text,
            "column_header": r == 0,
        }
        for r, row in enumerate([["Fault", "Check"], ["Cracked blade", "Measure thickness"]])
        for c, text in enumerate(row)
    ]
    data = {
        "tables": [
            {
                "self_ref": "t",
                "label": "table",
                "data": {"table_cells": cells, "num_rows": 2, "num_cols": 2},
            }
        ],
        "body": {"children": [{"$ref": "t"}]},
    }
    records = structured_records(data, Tokenizer(), 5)
    assert records[0][0] == "Fault | Check\nCracked blade | Measure thickness"
    assert records[0][1]["contains_table"]


def test_direction_repairs_are_source_based_and_idempotent():
    nodes = [
        GraphNode(
            "a", "转子不平衡", "Cause", "", "d", "v", {"source_surface": "Rotor mass imbalance"}
        ),
        GraphNode("b", "质量中心偏心", "Cause", "", "d", "v", {"source_surface": "eccentricity"}),
    ]
    edge = GraphEdge(
        "e",
        "a",
        "b",
        "CAUSES",
        "d",
        "v",
        properties={"evidence_text": "Rotor mass imbalance is due to eccentricity."},
    )
    ns, es = normalize_graph(nodes, [edge])
    assert (es[0].source_id, es[0].target_id) == ("b", "a")
    assert normalize_graph(ns, es) == (ns, es)
    assert es[0].properties["evidence_text"] == edge.properties["evidence_text"]


def test_defect_types_do_not_reclassify_detectors_or_filters():
    nodes = [
        GraphNode("a", "风扇叶片损伤", "Component", "", "d", "v"),
        GraphNode("b", "裂纹检测器", "Component", "", "d", "v"),
        GraphNode("c", "颗粒过滤器", "Component", "", "d", "v"),
    ]
    fixed, _ = normalize_graph(nodes, [])
    assert [n.kind for n in fixed] == ["Failure", "Component", "Component"]


def test_chinese_natural_question_and_compressor_alias_match_short_entity():
    node = GraphNode("n", "压气机外来沉积物", "Cause", "", "d", "v")
    assert ranked_entities("FAA培训文本中压缩机的外来沉积物有何影响?", [node])[0][0] == node


class Cases:
    backend_name = "none"

    def search(self, *args, **kwargs):
        return []


class Graph:
    backend_name = "test"

    def __init__(self, negative=False):
        self.nodes = [
            GraphNode("a", "压气机污染", "Cause", "", "d", "v"),
            GraphNode("b", "效率下降", "Symptom", "", "d", "v"),
        ]
        self.edge = GraphEdge(
            "e",
            "a",
            "b",
            "CAUSES",
            "d",
            "v",
            properties={
                "evidence_text": "Deposits reduce efficiency.",
                "chunk_id": "c",
                "polarity": "negative" if negative else "positive",
            },
        )

    def snapshot(self, **kwargs):
        return self.nodes, [self.edge]


class Vector:
    handles_reranking = True
    backend_name = "test"
    embedding_identity = "test"

    def __init__(self):
        self.chunk = VectorChunk(
            "c",
            "d",
            "v",
            "Deposits reduce efficiency.",
            {"display_content_zh": "压气机污染降低效率。"},
        )

    def get_chunk(self, key):
        return self.chunk if key == "c" else None

    def search(self, *args, **kwargs):
        return [VectorMatch(self.chunk, 0.9)]


def test_guided_route_keeps_full_source_and_edge_audit_without_duplicate_hit():
    tools = DiagnosticToolset(
        vector_store=Vector(), graph_store=Graph(), case_store=Cases(), source_guided_graph=True
    )
    result = tools.hybrid_search(
        command=DiagnosisCommand(session_id="s", question="压气机污染有哪些影响?"),
        query="压气机 污染",
        active_version_ids=frozenset({"v"}),
    )
    assert len(result.hits) == 1
    assert result.hits[0].evidence.excerpt == "Deposits reduce efficiency."
    assert result.hits[0].evidence.locator["graph_guidance"]["edge_ids"] == ["e"]
    assert {p.route for p in result.hits[0].contributions} == {"manual", "graph"}


def test_negative_graph_claim_cannot_promote_a_source():
    tools = DiagnosticToolset(
        vector_store=Vector(),
        graph_store=Graph(negative=True),
        case_store=Cases(),
        source_guided_graph=True,
    )
    result = tools.hybrid_search(
        command=DiagnosisCommand(session_id="s", question="压气机污染有哪些影响?"),
        query="压气机 污染",
        active_version_ids=frozenset({"v"}),
    )
    assert all(p.route != "graph" for h in result.hits for p in h.contributions)


def test_stale_chunk_cannot_be_returned_as_current_graph_evidence():
    vector = Vector()
    vector.chunk = replace(vector.chunk, version_id="old")
    tools = DiagnosticToolset(
        vector_store=vector, graph_store=Graph(), case_store=Cases(), source_guided_graph=True
    )
    assert (
        tools.execute(
            "traverse_fault_graph",
            command=DiagnosisCommand(session_id="s", question="压气机污染有哪些影响?"),
            query="压气机 污染",
            active_version_ids=frozenset({"v"}),
        )
        == ()
    )


def test_picture_container_body_text_is_not_lost():
    data = {
        "pictures": [{"self_ref": "pic", "label": "picture", "children": [{"$ref": "text"}]}],
        "texts": [
            {
                "self_ref": "text",
                "label": "text",
                "text": "Inspect the shroud and measure thickness at the worn area.",
                "prov": [{"page_no": 1}],
            }
        ],
        "body": {"children": [{"$ref": "pic"}]},
    }
    records = structured_records(data, Tokenizer(), 400)
    assert records[0][0] == "Inspect the shroud and measure thickness at the worn area."
    assert records[0][1]["pages"] == [1]


def test_list_items_are_not_sentence_continuations():
    data = {
        "texts": [
            {"self_ref": "a", "label": "text", "text": "The alternatives are:"},
            {"self_ref": "b", "label": "list_item", "text": "seal applied to a loose union"},
            {"self_ref": "c", "label": "list_item", "text": "seal applied to a tight union"},
        ],
        "body": {"children": [{"$ref": "a"}, {"$ref": "b"}, {"$ref": "c"}]},
    }
    records = structured_records(data, Tokenizer(), 400)
    assert "loose union seal" not in "\n".join(text for text, _ in records)
    assert "loose union\nseal" in "\n".join(text for text, _ in records)
