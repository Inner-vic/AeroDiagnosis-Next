# ruff: noqa: RUF001

from __future__ import annotations

import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence

from aerodiagnosis.ports import GraphEdge, GraphNode

_COMPONENT_TERMS = {
    "fan": ("风扇", "fan"),
    "compressor": ("压气机", "compressor"),
    "turbine": ("涡轮", "turbine"),
    "fuel_nozzle": ("燃油喷嘴", "fuel nozzle"),
    "bearing": ("轴承", "bearing"),
    "sensor": ("传感器", "sensor"),
}
_SYMPTOM_TERMS = {
    "stall": ("压气机喘振", "喘振", "surge", "stall"),
    "egt_high": ("EGT 异常升高", "排气温度升高", "EGT rise", "egt high"),
    "vibration": ("风扇振动异常", "振动升高", "vibration"),
}
_CAUSE_TERMS = {
    "stall": ("压气机喘振", "喘振", "stall"),
    "fouling": ("叶片积垢", "积垢", "污染", "fouling"),
    "erosion": ("叶片侵蚀", "侵蚀", "erosion"),
    "fod": ("外物损伤", "外物撞击", "FOD"),
    "sensor_chain": ("测量链异常", "传感链异常", "sensor chain"),
}
_SOLUTION_TERMS = {
    "inspect": ("孔探检查", "检查叶片", "孔探", "borescope"),
    "wash": ("压气机清洗", "清洗", "wash"),
    "verify_trend": ("趋势复核", "趋势确认", "trend review"),
}


def _normalized(value: str) -> str:
    return unicodedata.normalize("NFKC", value).casefold()


def _has_any(text: str, terms: Sequence[str]) -> bool:
    folded = _normalized(text)
    return any(_normalized(term) in folded for term in terms)


def _first_match(text: str, mapping: Mapping[str, Sequence[str]]) -> tuple[str, str] | None:
    for key, terms in mapping.items():
        if _has_any(text, terms):
            return key, terms[0]
    return None


def _entity_id(kind: str, label: str) -> str:
    return hashlib.sha256(f"{kind}\0{label}".encode()).hexdigest()


def _edge_id(relation: str, source_id: str, target_id: str) -> str:
    return hashlib.sha256(
        f"{relation}\0{source_id}\0{target_id}".encode()
    ).hexdigest()


class KnowledgeGraphExtractor:
    """Rule-based entity and relation extraction for structured aviation text."""

    def extract(
        self,
        *,
        source_ref: str,
        version_id: str,
        chunks: Sequence[str],
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        sentences = re.split(r"(?<=[。！？.!?])\s*|\n+", "\n".join(chunks))
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}
        equipment = self._node(
            "equipment",
            source_ref,
            version_id,
            "航空发动机",
            "Equipment",
            "通用航空发动机系统。",
            nodes,
        )

        for sentence in sentences:
            cleaned = sentence.strip()
            if not cleaned:
                continue
            component = _first_match(cleaned, _COMPONENT_TERMS)
            symptom = _first_match(cleaned, _SYMPTOM_TERMS)
            cause = _first_match(cleaned, _CAUSE_TERMS)
            solution = _first_match(cleaned, _SOLUTION_TERMS)
            if component is None:
                continue
            component_node = self._node(
                f"component:{component[0]}",
                source_ref,
                version_id,
                component[1],
                "Component",
                f"从文本中识别出的部件：{component[1]}。",
                nodes,
            )
            self._edge(
                f"has_component:{component[0]}",
                equipment.node_id,
                component_node.node_id,
                "HAS_COMPONENT",
                source_ref,
                version_id,
                1.0,
                edges,
            )
            if symptom is not None:
                symptom_node = self._node(
                    f"symptom:{symptom[0]}",
                    source_ref,
                    version_id,
                    symptom[1],
                    "Symptom",
                    f"从文本中识别出的症状：{symptom[1]}。",
                    nodes,
                )
                self._edge(
                    f"has_symptom:{component[0]}:{symptom[0]}",
                    component_node.node_id,
                    symptom_node.node_id,
                    "HAS_SYMPTOM",
                    source_ref,
                    version_id,
                    0.82,
                    edges,
                )
                if cause is not None:
                    cause_node = self._node(
                        f"cause:{cause[0]}",
                        source_ref,
                        version_id,
                        cause[1],
                        "Cause",
                        f"从文本中识别出的可能原因：{cause[1]}。",
                        nodes,
                    )
                    self._edge(
                        f"caused_by:{symptom[0]}:{cause[0]}",
                        symptom_node.node_id,
                        cause_node.node_id,
                        "CAUSED_BY",
                        source_ref,
                        version_id,
                        0.78,
                        edges,
                    )
                    if solution is not None:
                        solution_node = self._node(
                            f"solution:{solution[0]}",
                            source_ref,
                            version_id,
                            solution[1],
                            "Solution",
                            f"从文本中识别出的检查或处置：{solution[1]}。",
                            nodes,
                        )
                        self._edge(
                            f"solved_by:{cause[0]}:{solution[0]}",
                            cause_node.node_id,
                            solution_node.node_id,
                            "SOLVED_BY",
                            source_ref,
                            version_id,
                            0.8,
                            edges,
                        )

        return sorted(nodes.values(), key=lambda node: node.node_id), sorted(
            edges.values(), key=lambda edge: edge.edge_id
        )

    @staticmethod
    def _node(
        prefix: str,
        source_ref: str,
        version_id: str,
        name: str,
        kind: str,
        description: str,
        nodes: dict[str, GraphNode],
    ) -> GraphNode:
        node_id = _entity_id(kind, name)
        node = GraphNode(
            node_id=node_id,
            name=name,
            kind=kind,
            description=description,
            source_ref=source_ref,
            version_id=version_id,
            properties={"extraction": "lexical@1", "normalized": True},
        )
        nodes[node_id] = node
        return node

    @staticmethod
    def _edge(
        prefix: str,
        source_id: str,
        target_id: str,
        relation: str,
        source_ref: str,
        version_id: str,
        confidence: float,
        edges: dict[str, GraphEdge],
    ) -> None:
        edge_id = _edge_id(relation, source_id, target_id)
        edges[edge_id] = GraphEdge(
            edge_id=edge_id,
            source_id=source_id,
            target_id=target_id,
            relation=relation,
            source_ref=source_ref,
            version_id=version_id,
            confidence=confidence,
            properties={"extraction": "lexical@1", "normalized": True},
        )
