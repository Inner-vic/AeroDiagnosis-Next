"""Fair evidence payloads: sentence selectors and source text share one token budget."""

from __future__ import annotations

import json
from dataclasses import asdict
from typing import Any

from aerodiagnosis.application.grounded_answer import (
    EvidenceSpan,
    assemble_answer,
    spans_for_evidence,
)


def pack_spans(items: list[Any], tokenizer: Any, budget: int = 2048) -> dict[str, Any]:
    packed: list[dict[str, Any]] = []
    catalog: list[dict[str, Any]] = []
    spans: list[EvidenceSpan] = []
    aliases: dict[str, str] = {}
    used = len(tokenizer.encode('{"evidence":[]}', add_special_tokens=False))
    for item in items:
        text = str(item.locator.get("display_text_zh") or item.excerpt)
        entry = {"id": item.evidence_id, "text": text}
        records = spans_for_evidence([entry])
        selectors = [
            {"id": f"s{len(spans) + index + 1}", "text": span.text}
            for index, span in enumerate(records)
        ]
        model_entry: dict[str, Any] = {"source_id": f"e{len(packed) + 1}", "spans": selectors}
        notice = item.locator.get("source_integrity_warning")
        if isinstance(notice, str) and notice:
            model_entry["quality_notice"] = notice
        source = getattr(item, "document_metadata", {})
        if source:
            model_entry["source"] = {
                k: source.get(k) for k in ["source_url", "source_revision", "source_type"]
            }
            entry["source"] = model_entry["source"]
        size = len(
            tokenizer.encode(
                json.dumps(
                    {"evidence": [*catalog, model_entry]}, ensure_ascii=False, sort_keys=True
                ),
                add_special_tokens=False,
                verbose=False,
            )
        )
        if size > budget:
            continue
        used = size
        packed.append(entry)
        catalog.append(model_entry)
        spans.extend(records)
        aliases.update({s["id"]: span.span_id for s, span in zip(selectors, records, strict=True)})
    return {
        "packed": packed,
        "catalog": catalog,
        "spans": tuple(spans),
        "span_records": [asdict(s) for s in spans],
        "aliases": aliases,
        "context_tokens": used,
    }


def assemble_selected(value: dict[str, Any], bundle: dict[str, Any]) -> dict[str, Any]:
    # The model sees compact selectors; original offsets, IDs and hashes are
    # assigned only by the program. Unknown selectors remain invalid.
    claims = []
    raw_claims = value.get("claims")
    if not isinstance(raw_claims, list) or type(value.get("insufficient_evidence")) is not bool:
        return {
            "answer": "",
            "claims": [],
            "insufficient_evidence": False,
            "gate_valid": False,
            "gate_issues": [{"reason": "invalid_output_schema"}],
            "policy": "source-span-and-uncertainty@2",
        }
    for claim in raw_claims:
        if isinstance(claim, dict):
            refs = claim.get("support_span_ids")
            if isinstance(refs, list):
                claim = {
                    **claim,
                    "support_span_ids": [
                        bundle["aliases"].get(k, k) if isinstance(k, str) else k for k in refs
                    ],
                }
        claims.append(claim)
    return assemble_answer({**value, "claims": claims}, bundle["spans"])
