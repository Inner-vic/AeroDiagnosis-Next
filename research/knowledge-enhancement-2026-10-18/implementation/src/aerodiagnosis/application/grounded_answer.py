"""Exact display-text spans and conservative uncertainty gates; no gold access.

Offsets refer to the supplied display text, not the untranslated source file.
Passing these structural gates does not prove semantic entailment.
"""

from __future__ import annotations

# ruff: noqa: RUF001 -- Exact Chinese sentence punctuation is significant.
import hashlib
import re
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

from aerodiagnosis.application.evidence_semantics import ALTERNATIVE, relevant_propositions

UNCERTAIN = re.compile(
    r"可能|不一定|未必|疑似|推测|(?:未能|无法|不能|没有|未)[^。！？,，;；]{0,24}(?:确定|查明|确认)|不确定|尚不明确|尚不清楚|原因不明|未知|\b(?:may|might|could|likely|probabl[ey]|possibly|suspected|undetermined|unknown)\b",
    re.I,
)
STRONG_CERTAINTY = re.compile(r"一定|必然|已证实|\b(?:certainly|definitely|confirmed)\b", re.I)
CAUSAL = re.compile(r"导致|引起|造成|起因|源于|原因|\b(?:caus\w*|due to|result\w* in)\b", re.I)
UNRESOLVED = re.compile(
    r"(?:未能|无法|不能|没有|未)[^。！？,，;；]{0,24}(?:确定|查明|确认)|原因不明|"
    r"\b(?:not determined|could not determine|no evidence)\b",
    re.I,
)


@dataclass(frozen=True)
class EvidenceSpan:
    span_id: str
    evidence_id: str
    start: int
    end: int
    text: str
    text_sha256: str


def spans_for_evidence(evidence: list[dict[str, str]]) -> tuple[EvidenceSpan, ...]:
    output = []
    identities = set()
    for item in evidence:
        identity, text = item["id"], item["text"]
        if identity in identities:
            raise ValueError("Duplicate evidence identity")
        identities.add(identity)
        # Preserve exact source spelling and whitespace inside every sentence.
        boundaries = [0] + [
            m.end() for m in re.finditer(r"[。！？\n]|(?<=[.!?])\s+(?=[A-Z])", text)
        ]
        if boundaries[-1] != len(text):
            boundaries.append(len(text))
        sha = hashlib.sha256(text.encode()).hexdigest()
        for lo, hi in pairwise(boundaries):
            fragment = text[lo:hi]
            start = lo + len(fragment) - len(fragment.lstrip())
            end = hi - len(fragment) + len(fragment.rstrip())
            if end <= start:
                continue
            value = text[start:end]
            key = hashlib.sha256(f"{identity}:{start}:{end}:{sha}".encode()).hexdigest()
            output.append(EvidenceSpan(key, identity, start, end, value, sha))
    return tuple(output)


def uncertainty_issues(statement: str, quotes: list[str]) -> list[str]:
    # Exclude month names in dates from the English modal test.
    source = re.sub(
        r"\b(?:\d{1,2}\s+May|May\s+\d{1,2})\b",
        "",
        " ".join(relevant_propositions(statement, quotes)),
    )
    problems = []
    strong_statement = re.sub(
        r"(?:不|非|未)\s*(?:一定|必然|已证实)|\b(?:not|never)\s+(?:certainly|definitely|confirmed)\b",
        "",
        statement,
        flags=re.I,
    )
    if UNCERTAIN.search(source) and (
        not UNCERTAIN.search(statement) or STRONG_CERTAINTY.search(strong_statement)
    ):
        problems.append("source_uncertainty_removed")
    if UNRESOLVED.search(source) and CAUSAL.search(statement) and not UNRESOLVED.search(statement):
        problems.append("unresolved_cause_presented_as_identified")
    if (
        ALTERNATIVE.search(source)
        and re.search(r"表明|表示|可能性|indicat|possibil|either", source, re.I)
        and not (UNCERTAIN.search(statement) or ALTERNATIVE.search(statement))
    ):
        problems.append("source_alternatives_presented_as_simultaneous_facts")
    return problems


def assemble_answer(value: dict[str, Any], spans: tuple[EvidenceSpan, ...]) -> dict[str, Any]:
    by_id = {s.span_id: s for s in spans}
    raw = value.get("claims")
    if not isinstance(raw, list):
        raise ValueError("Claims must be a list")
    if not isinstance(value.get("insufficient_evidence"), bool):
        raise ValueError("Explicit boolean answerability required")
    accepted: list[dict[str, Any]] = []
    issues: list[dict[str, Any]] = []
    for index, claim in enumerate(raw):
        if not isinstance(claim, dict):
            issues.append({"claim_index": index, "reason": "invalid_claim"})
            continue
        statement, refs = claim.get("text"), claim.get("support_span_ids")
        if (
            not isinstance(statement, str)
            or not statement.strip()
            or not isinstance(refs, list)
            or not refs
        ):
            issues.append({"claim_index": index, "reason": "missing_text_or_spans"})
            continue
        if any(not isinstance(k, str) or k not in by_id for k in refs) or len(set(refs)) != len(
            refs
        ):
            issues.append({"claim_index": index, "reason": "unknown_or_duplicate_span"})
            continue
        selected = [by_id[k] for k in refs]
        problems = uncertainty_issues(statement, [s.text for s in selected])
        if problems:
            issues.extend({"claim_index": index, "reason": reason} for reason in problems)
            continue
        accepted.append(
            {
                "text": statement,
                "support_span_ids": refs,
                "evidence_ids": list(dict.fromkeys(s.evidence_id for s in selected)),
                "citations": [
                    {
                        "span_id": s.span_id,
                        "evidence_id": s.evidence_id,
                        "start": s.start,
                        "end": s.end,
                        "text_sha256": s.text_sha256,
                        "text_kind": "provided_display_text",
                        "support_quote": s.text,
                    }
                    for s in selected
                ],
            }
        )
    # Fail the whole output on a rejected factual claim; never silently improve a
    # score by dropping inconvenient facts. A bounded repair may be explicit.
    valid = not issues and bool(accepted or value["insufficient_evidence"])
    abstention = value.get("abstention_reason", "")
    if value["insufficient_evidence"] and (
        not isinstance(abstention, str) or not abstention.strip()
    ):
        issues.append({"reason": "missing_abstention_reason"})
        valid = False
    return {
        "answer": "\n".join(c["text"] for c in accepted)
        + (
            ("\n" if accepted else "") + "现有证据不足以完整回答该问题。"
            if value["insufficient_evidence"] and isinstance(abstention, str)
            else ""
        ),
        "insufficient_evidence": value["insufficient_evidence"],
        "claims": accepted,
        "gate_valid": valid,
        "gate_issues": issues,
        "policy": "source-span-and-uncertainty@2",
    }
