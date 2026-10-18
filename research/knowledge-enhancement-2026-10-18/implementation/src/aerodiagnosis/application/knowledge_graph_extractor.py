# ruff: noqa: RUF001
"""Conservative offline extraction of source-scoped, evidence-bearing claims."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

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
    "surge": ("喘振", "surge"),
    "stall": ("失速", "stall"),
    "egt_high": ("EGT 异常升高", "排气温度升高", "EGT rise", "egt high"),
    "vibration": ("振动升高", "风扇振动异常", "vibration"),
}
_CAUSE_TERMS = {
    "surge": ("喘振", "surge"),
    "stall": ("失速", "stall"),
    "fouling": ("叶片积垢", "积垢", "污染", "fouling"),
    "erosion": ("叶片侵蚀", "侵蚀", "erosion"),
    "fod": ("外物损伤", "外物撞击", "FOD"),
    "sensor_chain": ("测量链异常", "传感链异常", "sensor chain"),
}
_CHECK_TERMS = {
    "inspect": ("孔探检查", "检查叶片", "检查压气机叶片", "孔探", "borescope"),
    "verify_trend": ("趋势复核", "趋势确认", "trend review"),
}
_SOLUTION_TERMS = {"wash": ("压气机清洗", "清洗", "wash")}
_CAUSAL = re.compile(
    r"导致|引起|造成|源于|由于|可能是|原因是|caused\s+by|due\s+to|"
    r"leads?\s+to|results?\s+in|causes?",
    re.IGNORECASE,
)
_NEGATION = re.compile(
    r"未|无|不|否认|排除|并非|不能|\b(?:not|no|without|never|cannot|ruled\s+out)\b",
    re.IGNORECASE,
)
_POSSIBLE = re.compile(r"可能|疑似|可导致|\b(?:may|might|can|could|possible)\b", re.IGNORECASE)


def _digest(*values: str) -> str:
    return hashlib.sha256("\0".join(values).encode()).hexdigest()


@dataclass(frozen=True)
class _Mention:
    key: str
    label: str
    start: int
    end: int


def _mentions(text: str, terms: Mapping[str, Sequence[str]]) -> list[_Mention]:
    result = []
    for key, aliases in terms.items():
        pattern = "|".join(
            (r"(?<!\w)" + re.escape(alias) + r"(?!\w)") if alias.isascii() else re.escape(alias)
            for alias in sorted(aliases, key=len, reverse=True)
        )
        for match in re.finditer(pattern, text, re.IGNORECASE):
            result.append(_Mention(key, aliases[0], match.start(), match.end()))
    return sorted(result, key=lambda mention: (mention.start, mention.end, mention.key))


class KnowledgeGraphExtractor:
    """Emit only unambiguous lexical claims; scores are heuristics, not probabilities.

    Nodes are source/version occurrences. semantic_id groups vocabulary identities
    without overwriting their provenance or equating different physical engines.
    """

    identity = "lexical@2"

    def extract(
        self,
        *,
        source_ref: str,
        version_id: str,
        chunks: Sequence[str],
        chunk_locators: Sequence[Mapping[str, Any]] | None = None,
        chunk_ids: Sequence[str] | None = None,
    ) -> tuple[list[GraphNode], list[GraphEdge]]:
        if chunk_locators is not None and len(chunk_locators) != len(chunks):
            raise ValueError("chunk_locators must align with chunks")
        if chunk_ids is not None and len(chunk_ids) != len(chunks):
            raise ValueError("chunk_ids must align with chunks")
        nodes: dict[str, GraphNode] = {}
        edges: dict[str, GraphEdge] = {}

        def node(kind: str, mention: _Mention) -> GraphNode:
            semantic_key = f"{kind.casefold()}:{mention.key}"
            identity = _digest(source_ref, version_id, semantic_key)
            value = GraphNode(
                node_id=identity,
                name=mention.label,
                kind=kind,
                description=f"Source-scoped {kind} mention: {mention.label}.",
                source_ref=source_ref,
                version_id=version_id,
                properties={
                    "extraction": "lexical@2",
                    "semantic_id": _digest(semantic_key),
                    "semantic_key": semantic_key,
                    "review_status": "machine_candidate",
                },
            )
            if identity in nodes:
                return nodes[identity]
            nodes[identity] = value
            return value

        equipment = node("Equipment", _Mention("engine", "航空发动机", 0, 0))
        for chunk_index, chunk in enumerate(chunks):
            for sentence_index, match in enumerate(
                re.finditer(r"[^。！？.!?\n]+[。！？.!?]?", chunk)
            ):
                raw = match.group()
                sentence = raw.strip()
                if not sentence:
                    continue
                start = match.start() + len(raw) - len(raw.lstrip())
                properties: dict[str, Any] = {
                    "extraction": "lexical@2",
                    "evidence_text": sentence,
                    "chunk_index": chunk_index,
                    "sentence_index": sentence_index,
                    "sentence_start": start,
                    "sentence_end": start + len(sentence),
                    "polarity": "affirmed",
                    "modality": "possible" if _POSSIBLE.search(sentence) else "asserted",
                    "confidence_kind": "heuristic",
                    "review_status": "machine_candidate",
                    "conditions": sentence,
                }
                if chunk_ids is not None:
                    properties["chunk_id"] = chunk_ids[chunk_index]
                if chunk_locators is not None:
                    properties["locator"] = dict(chunk_locators[chunk_index])

                def edge(
                    source: GraphNode,
                    target: GraphNode,
                    relation: str,
                    evidence: Mapping[str, Any] = properties,
                    position: str = f"{chunk_index}:{sentence_index}:{start}:{sentence}",
                ) -> None:
                    identity = _digest(
                        version_id, relation, source.node_id, target.node_id, position
                    )
                    edges[identity] = GraphEdge(
                        edge_id=identity,
                        source_id=source.node_id,
                        target_id=target.node_id,
                        relation=relation,
                        source_ref=source_ref,
                        version_id=version_id,
                        confidence=0.8,
                        properties=dict(evidence),
                    )

                components = _mentions(sentence, _COMPONENT_TERMS)
                symptoms = _mentions(sentence, _SYMPTOM_TERMS)
                causes = _mentions(sentence, _CAUSE_TERMS)
                checks = _mentions(sentence, _CHECK_TERMS)
                solutions = _mentions(sentence, _SOLUTION_TERMS)
                for kind, mentions in (
                    ("Component", components),
                    ("Symptom", symptoms),
                    ("Cause", causes),
                    ("Check", checks),
                    ("Solution", solutions),
                ):
                    for mention in mentions:
                        node(kind, mention)

                def note(
                    reason: str,
                    entity_groups: tuple[tuple[str, list[_Mention]], ...] = (
                        ("Component", components),
                        ("Symptom", symptoms),
                        ("Cause", causes),
                    ),
                ) -> None:
                    for kind, mentions in entity_groups:
                        for mention in mentions:
                            value = node(kind, mention)
                            reasons = list(value.properties.get("skipped_relation_reasons", []))
                            if reason not in reasons:
                                reasons.append(reason)
                            nodes[value.node_id] = replace(
                                value,
                                properties={
                                    **value.properties,
                                    "skipped_relation_reasons": reasons,
                                },
                            )

                if _NEGATION.search(sentence):
                    # Skip the whole sentence until negation scope can be verified.
                    note("negation_or_exclusion")
                    continue
                if not _CAUSAL.search(sentence) and causes and symptoms:
                    note("no_explicit_causal_trigger")
                for component in components:
                    edge(equipment, node("Component", component), "HAS_COMPONENT")

                claimed_causes: dict[str, _Mention] = {}
                for trigger in _CAUSAL.finditer(sentence):
                    reverse = " ".join(trigger.group().casefold().split()) in {
                        "源于",
                        "由于",
                        "可能是",
                        "原因是",
                        "caused by",
                        "due to",
                    }
                    left, right = (symptoms, causes) if reverse else (causes, symptoms)
                    separators = list(re.finditer(r"[,，;；]", sentence))
                    clause_start = max(
                        (item.end() for item in separators if item.end() <= trigger.start()),
                        default=0,
                    )
                    clause_end = min(
                        (item.start() for item in separators if item.start() >= trigger.end()),
                        default=len(sentence),
                    )
                    subjects = {
                        item.key: item
                        for item in left
                        if clause_start <= item.start and item.end <= trigger.start()
                    }
                    objects = {
                        item.key: item
                        for item in right
                        if trigger.end() <= item.start and item.end <= clause_end
                    }
                    # Multiple entities on either side require a richer parser; no cross product.
                    if len(subjects) != 1 or len(objects) != 1:
                        note("ambiguous_or_missing_causal_arguments")
                        continue
                    subject = next(iter(subjects.values()))
                    obj = next(iter(objects.values()))
                    symptom, cause = (subject, obj) if reverse else (obj, subject)
                    if symptom.key == cause.key:
                        continue
                    edge(node("Symptom", symptom), node("Cause", cause), "CAUSED_BY")
                    claimed_causes[cause.key] = cause
                    if len({item.key for item in components}) == 1:
                        edge(
                            node("Component", components[0]),
                            node("Symptom", symptom),
                            "HAS_SYMPTOM",
                        )
                if (
                    len(claimed_causes) == 1
                    and len(list(_CAUSAL.finditer(sentence))) == 1
                    and len({item.key for item in symptoms} - claimed_causes.keys()) == 1
                    and checks
                    and len({item.key for item in components}) <= 1
                    and re.search(
                        r"应|建议|检查|复核|\b(?:check|inspect|review|recommend|should)\b",
                        sentence,
                        re.IGNORECASE,
                    )
                ):
                    cause = next(iter(claimed_causes.values()))
                    for check in checks:
                        edge(node("Cause", cause), node("Check", check), "CHECKED_BY")
                # Require an explicit action -> mitigation verb -> cause span.
                # A remedy plus an unrelated "remove" clause is insufficient.
                for trigger in re.finditer(
                    r"去除|清除|缓解|\b(?:remove|mitigate|reduce)s?\b", sentence, re.IGNORECASE
                ):
                    separators = list(re.finditer(r"[,，;；]", sentence))
                    clause_start = max(
                        (item.end() for item in separators if item.end() <= trigger.start()),
                        default=0,
                    )
                    clause_end = min(
                        (item.start() for item in separators if item.start() >= trigger.end()),
                        default=len(sentence),
                    )
                    actions = {
                        item.key: item
                        for item in solutions
                        if clause_start <= item.start and item.end <= trigger.start()
                    }
                    targets = {
                        item.key: item
                        for item in causes
                        if trigger.end() <= item.start and item.end <= clause_end
                    }
                    if len(actions) == len(targets) == 1:
                        action = next(iter(actions.values()))
                        target = next(iter(targets.values()))
                        edge(node("Cause", target), node("Solution", action), "MITIGATED_BY")

        return sorted(nodes.values(), key=lambda value: value.node_id), sorted(
            edges.values(),
            key=lambda value: value.edge_id,
        )
