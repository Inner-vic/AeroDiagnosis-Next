"""Repair source-only development copies and replay old candidates as regression, not holdout."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from aerodiagnosis.adapters.llm.chinese_document import (
    ChineseDocumentDisplay,
    TranslationQualityError,
)
from aerodiagnosis.adapters.llm.research_client import ResearchClient
from aerodiagnosis.application.evidence_semantics import translation_issues
from aerodiagnosis.application.grounded_answer import EvidenceSpan
from aerodiagnosis.application.grounded_context import assemble_selected
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import VectorChunk


def main():
    root = Path.cwd()
    original = root / ".runtime/experiments/holdout-v2-final-20261014"
    out = root / ".runtime/quality-v3"
    out.mkdir(exist_ok=True)
    data = json.loads((original / "snapshot.json").read_text(encoding="utf-8"))
    profile = RuntimeSettings.from_env(root / ".runtime/holdout-v2/profile")
    assert profile.default_llm_api_key is not None
    client = ResearchClient(
        api_key=profile.default_llm_api_key.get_secret_value(),
        runtime=profile.knowledge_budget_runtime,
        cap_cny=35,
        model="deepseek-flash",
    )
    translator = ChineseDocumentDisplay(client, stage="supplemental_review")
    decisions = []
    prior = {}
    for row in sorted(
        data["chunks"],
        key=lambda c: (c["document_id"], c["metadata"]["locator"]["coordinates"]["start"]),
    ):
        previous = prior.get(row["document_id"])
        metadata = dict(row["metadata"])
        # A colon-introduced continuation/list obtains source context, not test labels.
        if previous and previous["content"].rstrip().endswith(":"):
            metadata["source_semantic_context_before"] = previous["content"][-1500:]
        prior[row["document_id"]] = row
        text = metadata["display_content_zh"]
        context = metadata.get("source_semantic_context_before", "")
        issues = translation_issues(row["content"], text, context)
        if not issues:
            continue
        chunk = VectorChunk(
            row["chunk_id"], row["document_id"], row["version_id"], row["content"], metadata
        )
        try:
            changed = translator.enrich([chunk])[0]
        except TranslationQualityError as exc:
            decisions.append(
                {
                    "chunk_id": row["chunk_id"],
                    "source_content": row["content"],
                    "source_context_before": context,
                    "original_translation": text,
                    "replacement_translation": exc.candidate_translation,
                    "issues": list(exc.quality_issues),
                    "status": "requires_independent_source_review",
                    "source_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
                }
            )
            continue
        decisions.append(
            {
                "chunk_id": row["chunk_id"],
                "source_content": row["content"],
                "source_context_before": context,
                "original_translation": text,
                "replacement_translation": changed.metadata["display_content_zh"],
                "issues": issues,
                "source_sha256": hashlib.sha256(row["content"].encode()).hexdigest(),
            }
        )
        row["metadata"] = dict(changed.metadata)
    (out / "development-source-snapshot-repaired.json").write_text(
        json.dumps(data, ensure_ascii=False), encoding="utf-8"
    )
    (out / "source-repair-decisions.json").write_text(
        json.dumps(decisions, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    replay = []
    for path in sorted((original / "answers").glob("*.json")):
        row = json.loads(path.read_text(encoding="utf-8"))
        arms = {}
        for name, arm in row["arms"].items():
            spans = tuple(EvidenceSpan(**s) for s in arm["span_records"])
            selectors = [s["id"] for entry in arm["span_catalog"] for s in entry["spans"]]
            bundle = {
                "spans": spans,
                "aliases": {k: s.span_id for k, s in zip(selectors, spans, strict=True)},
            }
            assembled = assemble_selected(arm["raw_reply"], bundle)
            arms[name] = {
                "old_gate_valid": arm["answer"]["gate_valid"],
                "new_gate_valid": assembled["gate_valid"],
                "new_gate_issues": assembled["gate_issues"],
            }
        replay.append({"sample_id": row["id"], "arms": arms})
    result = {
        "basis": "old candidate replay on old supplied spans; development regression only",
        "paid_generation_reruns": 0,
        "source_repair_candidates": len(decisions),
        "ready_translation_repairs": sum(
            x.get("status") != "requires_independent_source_review" for x in decisions
        ),
        "pending_source_reviews": sum(
            x.get("status") == "requires_independent_source_review" for x in decisions
        ),
        "results": replay,
        "gate_failures": {
            name: sum(not r["arms"][name]["new_gate_valid"] for r in replay)
            for name in ["text", "text_graph"]
        },
        "budget": {k: v for k, v in client.summary().items() if k != "receipts"},
    }
    (out / "development-regression.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({k: v for k, v in result.items() if k != "results"})


if __name__ == "__main__":
    main()
