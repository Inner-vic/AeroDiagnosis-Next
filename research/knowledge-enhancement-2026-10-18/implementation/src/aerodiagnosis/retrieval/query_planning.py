"""Offline development query plans; never replace the user's original query.

Inspired by MAKG's preserve-original query decomposition. Rules are deliberately
conservative; this module is not wired into the frozen production or v4 policies.
"""

from __future__ import annotations

# ruff: noqa: RUF001 -- Match original Chinese punctuation and mathematical signs exactly.
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from typing import Any, Literal

POLICY = "conservative-query-plan@1"
# Existing aviation terminology, not learned from held-out questions.
ALIASES = (("压缩机", "压气机"), ("启动", "起动"))
_PROTECTED = re.compile(
    r"[A-Za-z][A-Za-z0-9]*(?:[-_/][A-Za-z0-9]+)*|[−+-]?\d+(?:\.\d+)?(?:%|°C|℃|rpm)?"
)
_QUALIFIERS = re.compile(
    r"如果|除非|只有|否则|若|当|时|未|不|无|非|可能|尚|未知|仅|必须|[<>≤≥]|大于|小于|低于|高于"
)
_EGT = re.compile(r"(?<![A-Za-z0-9_/-])EGT(?![A-Za-z0-9_/-])", re.I)


@dataclass(frozen=True, slots=True)
class QueryVariant:
    text: str
    kind: Literal["original", "terminology", "subquestion"]


@dataclass(frozen=True, slots=True)
class QueryPlan:
    original: str
    variants: tuple[QueryVariant, ...]
    skipped: tuple[str, ...] = ()
    policy: str = POLICY


def protected_tokens(query: str) -> tuple[str, ...]:
    return tuple(sorted(set(m.group().casefold() for m in _PROTECTED.finditer(query))))


def build_query_plan(
    query: str, *, terminology: bool = True, decompose: bool = False, max_queries: int = 3
) -> QueryPlan:
    if not query.strip() or not 1 <= max_queries <= 3:
        raise ValueError("A query and a maximum of 1..3 queries are required")
    original = query.strip()
    variants = [QueryVariant(original, "original")]
    skipped = []
    # Split only explicit shared context followed by multiple complete questions.
    # No implied context resolution, causal rewriting or assertion generation.
    if decompose:
        parts = re.split(r"[：:]", original, maxsplit=1)
        if len(parts) == 2 and parts[0].strip() and not _QUALIFIERS.search(original):
            questions = re.findall(r"[^？?]+[？?]", parts[1])
            remainder = re.sub(r"[^？?]+[？?]", "", parts[1]).strip()
            proposals = [parts[0] + "：" + q.strip() for q in questions]
            if (
                not remainder
                and all(
                    not re.match(
                        r"^(如何|为什么|怎样|能否|是否|怎么办|还有|它|其|该|这|那)", q.strip()
                    )
                    for q in questions
                )
                and 2 <= len(proposals) <= max_queries - 1
                and all(protected_tokens(q) == protected_tokens(original) for q in proposals)
            ):
                variants.extend(QueryVariant(q, "subquestion") for q in proposals)
            else:
                skipped.append("incomplete_questions_or_unshared_identifiers_or_budget")
        else:
            skipped.append("shared_context_required_or_qualified_query")
    if terminology and len(variants) < max_queries:
        expanded = original
        for source, target in ALIASES:
            expanded = expanded.replace(source, target)
        # Keep the acronym itself, including its casing, alongside the Chinese term.
        expanded = _EGT.sub(lambda m: "排气温度（" + m.group() + "）", expanded)
        if expanded != original:
            if protected_tokens(expanded) != protected_tokens(original):
                raise ValueError("Terminology expansion changed protected identifiers or values")
            variants.append(QueryVariant(expanded, "terminology"))
    unique = {v.text: v for v in variants}
    return QueryPlan(original, tuple(unique.values()), tuple(skipped))


def candidate_quotas(total: int, count: int) -> tuple[int, ...]:
    """Prioritize the original; sum of requested candidate limits never grows."""
    if total < 1 or not 1 <= count <= 3 or count > total:
        raise ValueError("Candidate budget must cover every query")
    if count == 1:
        return (total,)
    original = (total + 1) // 2
    remaining = total - original
    if remaining < count - 1:
        original = total - (count - 1)
        remaining = count - 1
    base, extra = divmod(remaining, count - 1)
    return (original, *(base + (i < extra) for i in range(count - 1)))


def planner_identity(*, terminology: bool, decompose: bool, candidate_budget: int) -> str:
    payload = {
        "policy": POLICY,
        "aliases": ALIASES,
        "egt": "preserve acronym plus 排气温度",
        "terminology": terminology,
        "decompose": decompose,
        "candidate_budget": candidate_budget,
        "max_queries": 3,
        "rrf_k": 60,
    }
    return (
        POLICY
        + ":"
        + hashlib.sha256(
            json.dumps(payload, ensure_ascii=False, sort_keys=True).encode()
        ).hexdigest()
    )


def plan_record(plan: QueryPlan) -> dict[str, Any]:
    return asdict(plan)
