"""Create answer-hidden original-source packets without system predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    prepared = root / ".runtime/holdout-v3/prepared-v5"
    if (dataset / "freeze.json").exists():
        raise ValueError("Packets must precede predictions")

    def read(p):
        return json.loads(p.read_text(encoding="utf-8"))

    qs = [
        json.loads(s) for s in (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    out = root / ".runtime/holdout-v3/answer-hidden"
    out.mkdir(exist_ok=True)
    groups = {}
    for q in qs:
        groups.setdefault(q["document_id"], []).append(
            {"sample_id": q["sample_id"], "query": q["query"]}
        )
    for d in read(prepared / "catalog.json")["documents"]:
        source = read(prepared / d["parsed_path"])["canonical_content"]
        packet = {
            "document_id": d["original_source_id"],
            "source_original": source,
            "source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "questions": groups[d["original_source_id"]],
            "instructions": (
                "Independently reconstruct answer from saved original source. "
                "No labels or predictions supplied. For each sample return reviewable, "
                "answer_zh, required_parts, contiguous source_quotes, certainty, issues. "
                "Do not invent missing figures or universal maintenance authority."
            ),
        }
        (out / (d["original_source_id"] + ".json")).write_text(
            json.dumps(packet, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    print("Answer-hidden packets", len(groups), "questions", len(qs))


if __name__ == "__main__":
    main()
