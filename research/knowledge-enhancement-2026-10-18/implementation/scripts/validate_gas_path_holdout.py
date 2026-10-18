"""Validate local holdout artifacts without running retrieval or an LLM."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--require-frozen", action="store_true")
    args = parser.parse_args()
    root = ROOT / "evaluation/gas_path_holdout_v1_candidate"
    queries = [
        json.loads(s) for s in (root / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    blind = [
        json.loads(s)
        for s in (root / "blind_queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    generation = [
        json.loads(s)
        for s in (root / "generation_queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    manifest = json.loads((root / "source_manifest.json").read_text(encoding="utf-8"))
    sources = {x["document_id"]: x for x in manifest["documents"]}
    assert len(queries) == len(blind) == len(generation)
    assert len({x["sample_id"] for x in queries}) == len(queries)
    assert len({x["query"] for x in queries}) == len(queries)
    checked = 0
    for row, reviewer_row, generator_row in zip(queries, blind, generation, strict=True):
        assert row["sample_id"] == reviewer_row["sample_id"] == generator_row["sample_id"]
        assert row["query"] == reviewer_row["query"] == generator_row["query"]
        assert row["query_language"] == "zh"
        assert set(generator_row) == {"sample_id", "query", "query_language"}
        assert not any(
            k in reviewer_row
            for k in ["reference_answer", "minimal_facts", "evidence", "answerability"]
        )
        signed = {k: v for k, v in row.items() if k != "annotation_sha256"}
        assert row["annotation_sha256"] == sha(
            json.dumps(signed, ensure_ascii=False, sort_keys=True)
        ), row["sample_id"]
        for ev in row["evidence"]:
            doc = ev["document_id"]
            text = (root / "sources" / f"{doc}.txt").read_text(encoding="utf-8")
            assert sha(text) == ev["canonical_sha256"] == sources[doc]["canonical_sha256"]
            blocks = json.loads(
                (root / "sources" / f"{doc}.blocks.json").read_text(encoding="utf-8")
            )
            block = next(x for x in blocks if x["block_id"] == ev["block_id"])
            excerpt = text[ev["canonical_start"] : ev["canonical_end"]]
            assert excerpt == block["text"]
            assert sha(excerpt) == ev["text_sha256"]
            assert ev["source_url"] == sources[doc]["url"]
            assert ev["canonical_start"] == block["start"] and ev["canonical_end"] == block["end"]
            checked += 1
    if args.require_frozen:
        freeze = json.loads((root / "freeze.json").read_text(encoding="utf-8"))
        assert freeze["queries_sha256"] == sha((root / "queries.jsonl").read_text(encoding="utf-8"))
        assert freeze["knowledge_coverage_verified"] is True
        assert freeze["unresolved_reviews"] == 0
        assert all(x["review_status"] == "ai_cross_source_verified" for x in queries)
    print(
        json.dumps(
            {
                "queries": len(queries),
                "evidence_bindings_checked": checked,
                "generation_gold_leakage": False,
                "annotation_hashes_valid": True,
                "frozen": (root / "freeze.json").exists(),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
