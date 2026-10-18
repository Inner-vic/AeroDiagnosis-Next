"""Shared Chinese query normalization and lexical relevance."""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from math import log

from aerodiagnosis.ports import GraphNode


def normalize_query(value: str) -> str:
    text = unicodedata.normalize("NFKC", value).casefold()
    for source, target in [
        ("压缩机", "压气机"),
        ("启动", "起动"),
        ("外罩", "护罩"),
        ("egt", "排气温度"),
    ]:
        text = text.replace(source, target)
    return text


def query_terms(value: str) -> set[str]:
    text = normalize_query(value)
    for phrase in [
        "根据",
        "提供",
        "资料",
        "培训",
        "文本",
        "来源",
        "什么",
        "哪些",
        "如何",
        "是否",
        "能否",
        "关于",
        "中的",
        "faa",
        "条件",
        "措施",
        "以及",
        "相关",
        "列出",
        "说明",
    ]:
        text = text.replace(phrase, " ")
    terms = set(re.findall(r"[a-z0-9_]{2,}", text))
    for span in re.findall(r"[\u3400-\u9fff]+", text):
        terms.update(span[i : i + 2] for i in range(len(span) - 1))
    return terms


def relevance(query: str, text: str) -> float:
    terms = query_terms(query)
    candidate = normalize_query(text)
    return sum(term in candidate for term in terms) / max(1, len(terms))


def ranked_entities(query: str, nodes: Sequence[GraphNode]) -> list[tuple[GraphNode, float]]:
    terms = query_terms(query)
    if not terms:
        return []
    names = {n.node_id: normalize_query(n.name) for n in nodes}
    frequency = Counter(term for term in terms for name in names.values() if term in name)

    def score(node: GraphNode) -> float:
        name = names[node.node_id]
        matched = [term for term in terms if term in name]
        return sum(log(1 + len(nodes) / (1 + frequency[t])) for t in matched)

    return sorted(
        ((node, score(node)) for node in nodes if score(node) > 0),
        key=lambda row: (-row[1], row[0].node_id),
    )
