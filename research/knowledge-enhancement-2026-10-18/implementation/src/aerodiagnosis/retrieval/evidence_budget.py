"""Auditable evidence packing around the existing source-span/token-budget contract."""

from __future__ import annotations

from typing import Any

from aerodiagnosis.application.grounded_context import pack_spans
from aerodiagnosis.domain import EvidenceItem


def pack_evidence_audited(
    items: list[EvidenceItem], tokenizer: Any, budget: int = 2048
) -> dict[str, Any]:
    if type(budget) is not int or budget < 1:
        raise ValueError("A positive whole-token budget is required")
    if len({i.evidence_id for i in items}) != len(items):
        raise ValueError("Duplicate evidence IDs must be resolved before packing")
    if any(i.excerpt_truncated for i in items):
        raise ValueError("Truncated source evidence cannot enter whole-source packing")
    bundle = pack_spans(items, tokenizer, budget)
    if bundle["context_tokens"] > budget:
        raise ValueError("Budget cannot fit the empty evidence envelope")
    selected = {entry["id"] for entry in bundle["packed"]}
    bundle["selection_audit"] = {
        "policy": "whole-evidence-budget-audit@1",
        "budget_tokens": budget,
        "context_tokens": bundle["context_tokens"],
        "selected_ids": [i.evidence_id for i in items if i.evidence_id in selected],
        "omitted": [
            {
                "evidence_id": i.evidence_id,
                "document_id": i.document_id,
                "version_id": i.version_id,
                "reason": "whole_evidence_exceeds_remaining_token_budget",
            }
            for i in items
            if i.evidence_id not in selected
        ],
        "source_rewritten": False,
        "semantic_completeness_evaluated": False,
    }
    return bundle
