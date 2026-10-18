# ruff: noqa: E501
"""Adopt full source-only semantic review and freeze graph before question authoring."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
from pathlib import Path

from prepare_strict_graph_holdout import OUT, ROOT, anchor, digest, read, save, sha

REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/strict-source-review/semantic-review"
)


def adopt():
    assert not (ROOT / "graph-freeze.json").exists()
    data = read(ROOT / "translated-corpus-candidate.json")
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    items = []
    corrections = []
    reviewed = []
    reviews = []
    for doc in data["documents"]:
        review = read(REVIEW / (doc["document_id"] + ".json"))
        assert (
            review["document_id"] == doc["document_id"]
            and review["source_sha256"] == doc["source_sha256"]
        )
        assert (
            not review["expert_reviewed"]
            and not review["questions_seen"]
            and not review["predictions_seen"]
        )
        assert review["translation_review_complete"] and set(review["reviewed_chunk_ids"]) == set(
            doc["chunk_ids"]
        )
        raw = read(OUT / "extraction" / (doc["document_id"] + ".json"))
        assert {r["assertion_index"] for r in review["assertions"]} == set(
            range(len(raw["assertions"]))
        )
        assert len(review["assertions"]) == len(raw["assertions"])
        for correction in review["translation_corrections"]:
            key = correction["chunk_id"]
            assert key in doc["chunk_ids"]
            c = chunks[key]
            text = correction["text_zh"]
            assert isinstance(text, str) and text.strip()
            corrections.append(
                {
                    "document_id": doc["document_id"],
                    "chunk_id": key,
                    "before_sha256": sha(c["metadata"]["display_content_zh"]),
                    "after_sha256": sha(text),
                    "reason_zh": correction["reason_zh"],
                }
            )
            c["metadata"]["display_content_zh"] = text
        for key in doc["chunk_ids"]:
            c = chunks[key]
            assert c["metadata"]["display_translation_source_sha256"] == sha(c["content"])
            c["metadata"]["translation_review_status"] = "source_semantic_checked_nonexpert"
        for decision in review["assertions"]:
            assert decision["decision"] in {"accepted", "revise", "reject"}
            assert decision["in_anchored_candidate"]
            items.append({"document_id": doc["document_id"], **decision})
        reviewed.append(doc["document_id"])
        reviews.append(review)
    save(ROOT / "translated-corpus-final.json", data)
    save(
        ROOT / "translation-adoption.json",
        {
            "expert_reviewed": False,
            "all_chunks_independently_reviewed": True,
            "chunks": len(chunks),
            "corrections": corrections,
            "questions_seen": False,
            "predictions_seen": False,
            "document_reviews": reviews,
        },
    )
    path = ROOT / "graph-source-semantic-review.json"
    save(
        path,
        {
            "expert_reviewed": False,
            "questions_seen": False,
            "predictions_seen": False,
            "reviewed_document_ids": reviewed,
            "items": items,
        },
    )
    anchor(str(path))
    audit = read(ROOT / "source-graph-review-adoption-audit.json")
    unexpected = [
        r
        for r in audit["rejected"]
        if r["reason"] != "not_reviewed_or_rejected_by_independent_source_review"
    ]
    save(OUT / "semantic-adoption-unresolved.json", {"unexpected_anchor_failures": unexpected})
    if unexpected:
        raise ValueError(
            "Source-reviewed assertions still have anchor/schema errors; resolve before graph freeze"
        )
    graph = read(ROOT / "source-graph-final.json")
    save(
        ROOT / "graph-freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "before_questions": True,
            "expert_reviewed": False,
            "source_freeze_sha256": digest(ROOT / "source-freeze.json"),
            "source_graph_sha256": digest(ROOT / "source-graph-final.json"),
            "source_review_sha256": digest(path),
            "translation_sha256": digest(ROOT / "translated-corpus-final.json"),
            "translation_review_sha256": digest(ROOT / "translation-adoption.json"),
            "raw_graph_sha256": digest(ROOT / "source-graph-candidate.json"),
            "nodes": len(graph["nodes"]),
            "edges": len(graph["edges"]),
            "reviewed_documents": len(reviewed),
            "curation": "source-only independent AI nonexpert, no author questions/gold/strategy/rankings shown",
        },
    )
    print(
        "Frozen reviewed graph",
        len(graph["edges"]),
        "edges; translation corrections",
        len(corrections),
        flush=True,
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.parse_args()
    adopt()
