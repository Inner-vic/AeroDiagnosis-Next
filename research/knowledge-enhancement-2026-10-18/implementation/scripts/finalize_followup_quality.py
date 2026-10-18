"""Apply independently source-reviewed translations; no test facts enter the corpus."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese source integrity notice.
from dataclasses import replace
from pathlib import Path

from evaluate_followup_holdout import digest, read, save, snapshot

from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.config import RuntimeSettings


def main():
    root = Path.cwd()
    if (root / "evaluation/gas_path_holdout_v2_candidate/freeze.json").exists():
        raise ValueError("No quality changes after freezing")
    review_path = root / ".runtime/holdout-v2/review-translation-adjudicated.json"
    review = read(review_path)
    prior = root / ".runtime/experiments/holdout-v2-paired-20261014"
    record = read(prior / "quality-adjudication.json")
    if {r["id"] for r in review["items"]} != set(record["unresolved_translation_ids"]):
        raise ValueError("Independent source adjudication incomplete")
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v2/profile")
    vector = SQLiteVectorStore(settings.database_path)
    lexical = SQLiteLexicalStore(settings.database_path)
    changes = []
    decisions = []
    for item in review["items"]:
        if item["decision"] not in ["keep_original", "use_proposed", "replace"]:
            raise ValueError("Unsupported quality decision")
        chunk = vector.get_chunk(item["id"])
        if chunk is None:
            raise ValueError("Source chunk unavailable")
        chinese = item["final_chinese"]
        if not isinstance(chinese, str) or len(chinese) < 20:
            raise ValueError("Full Chinese translation required")
        metadata = {
            **chunk.metadata,
            "display_content_zh": chinese,
            "quality_review_status": "independent_source_reviewed_before_predictions",
            "independent_source_review_sha256": digest(review),
            "independent_source_review_reason": item["reason"],
            "human_expert_reviewed": False,
        }
        # Figure labels and a damaged source definition remain visible as source
        # limitations; no reconstructed equations or additional facts are inserted.
        if item["id"].startswith(("2e291790cf", "8e4390afdc", "cea87d1ad9")):
            metadata["source_integrity_warning"] = (
                "部分图注或定义措辞存在原始排版歧义；仅使用可辨正文，不推补图示。"
            )
        changes.append(replace(chunk, metadata=metadata))
        decisions.append({"id": item["id"], "decision": item["decision"], "reason": item["reason"]})
    vector.upsert(changes)
    lexical.upsert(changes)
    out = root / ".runtime/experiments/holdout-v2-final-20261014"
    out.mkdir(parents=True, exist_ok=True)
    final, graph = snapshot(settings, out)
    save(
        out / "quality-adjudication.json",
        {
            **record,
            "snapshot_sha256": digest(final),
            "unresolved_major": 0,
            "unresolved_translation_ids": [],
            "final_source_decisions": decisions,
            "independent_review_sha256": digest(review),
            "independent_source_review_expert": False,
            "source_review_limit": (
                "AI source review; reviewer had previously seen questions, no predictions"
            ),
            "source_text_changed_by_test_gold": False,
        },
    )
    save(
        out / "source-review-provenance.json",
        {
            "prior_quality_directory": str(prior),
            "independent_review_path": str(review_path),
            "final_snapshot_sha256": digest(final),
            "no_system_predictions": True,
        },
    )
    graph.close()
    print("Source quality finalized; seven unresolved findings explicitly adjudicated")


if __name__ == "__main__":
    main()
