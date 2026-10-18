"""Source-aware graph repairs; never infer causes from a token match alone."""

from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import replace

from aerodiagnosis.ports import GraphEdge, GraphNode


def normalize_graph(
    nodes: Sequence[GraphNode], edges: Sequence[GraphEdge]
) -> tuple[list[GraphNode], list[GraphEdge]]:
    by_id = {n.node_id: n for n in nodes}
    repaired = []
    for edge in edges:
        source, target = by_id[edge.source_id], by_id[edge.target_id]
        quote = str(edge.properties.get("evidence_text", ""))
        swap = False
        if edge.relation == "CAUSES":
            marker = re.search(
                r"\b(?:is|are)\s+(?:generally\s+)?(?:the result of|due to)\b", quote, re.I
            )
            if marker:
                # Extracted surface spans are retained separately before Chinese naming.
                subject = str(source.properties.get("source_surface", source.name))
                object_ = str(target.properties.get("source_surface", target.name))
                left, right = quote[: marker.start()], quote[marker.end() :]
                swap = (
                    subject.casefold() in left.casefold() and object_.casefold() in right.casefold()
                )
        if edge.relation == "REQUIRES_ACTION":
            swap = source.kind == "Action" and target.kind in {
                "Failure",
                "Condition",
                "Component",
                "Symptom",
            }
        relation = edge.relation
        if relation == "REQUIRES_ACTION" and re.search(
            r"(?:remedied|corrected|repaired) by", quote, re.I
        ):
            relation = "MITIGATED_BY"
        if relation == "INDICATES" and "attracts" in quote.casefold():
            relation = "ATTRACTS"
        repaired.append(
            replace(
                edge,
                source_id=edge.target_id if swap else edge.source_id,
                target_id=edge.source_id if swap else edge.target_id,
                relation=relation,
                properties={
                    **edge.properties,
                    "quality_policy": "source-roles@1",
                    **(
                        {"predicate_repair": {"previous_relation": edge.relation}}
                        if relation != edge.relation
                        else {}
                    ),
                    **(
                        {
                            "direction_repair": {
                                "previous_source": edge.source_id,
                                "previous_target": edge.target_id,
                            }
                        }
                        if swap
                        else {}
                    ),
                },
            )
        )
    final_nodes = []
    for node in nodes:
        kind, name = node.kind, node.name
        if kind == "Component" and re.search(r"(?:损伤|损坏|裂纹|松动)$", name):
            kind = "Failure"
        if kind == "Cause" and re.search(r"裂纹或外来物损伤", name):
            kind = "Failure"
        if kind in {"Component", "Symptom"} and name.endswith("颗粒"):
            kind = "Material"
        if kind == "Failure" and "拒收" in name:
            kind = "Action"
        if (
            kind == "Mechanism"
            and str(node.properties.get("source_surface", "")).casefold() == "gas"
        ):
            kind, name = "Material", "起动气体"
        final_nodes.append(
            replace(
                node,
                kind=kind,
                name=name,
                properties={
                    **node.properties,
                    "name_zh": name,
                    "quality_policy": "source-roles@1",
                    **({"type_repair": {"previous_kind": node.kind}} if kind != node.kind else {}),
                },
            )
        )
    return final_nodes, repaired
