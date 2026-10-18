"""Fixed source-only selection of engine sections for the new independent study."""

from __future__ import annotations

import json
import re
from pathlib import Path

import prepare_holdout_knowledge as prep


def select_blocks(row, blocks):
    if row["publisher"] == "NASA":
        selected = [
            b for b in blocks if len(b["text"]) > 400 and not b["text"].startswith(("#", "!["))
        ][:1]
    else:
        kept = []
        active = False
        for b in blocks:
            value = b["text"]
            if value.startswith("#"):
                active = bool(
                    re.search(
                        r"engine|compressor|turbine|blade|bearing|fuel|oil|ignition|ice|icing|"
                        r"analysis|finding|examination|safety action",
                        value,
                        re.I,
                    )
                )
            if (
                active or re.search(r"\bengine\b|\bcompressor\b|\bturbine\b", value, re.I)
            ) and not re.search(r"cookie|privacy|sign up|subscribe|JavaScript", value, re.I):
                kept.append(b)
        # Keep whole blocks; source-only fixed cap, not selected using future questions.
        selected = []
        used = 0
        for b in kept:
            if used + len(b["text"]) > 12000 and selected:
                continue
            selected.append(b)
            used += len(b["text"])
        if not selected:
            selected = [b for b in blocks if len(b["text"]) > 400][:3]
    if not selected:
        raise ValueError("No engine source text selected")
    return selected, {
        "source_body_scope": (
            "engine-related saved paragraphs, fixed 12000 character whole-block cap"
        ),
        "source_only": True,
        "private_use_glyph_blocks": sum(
            bool(re.search(r"[\ue000-\uf8ff]", b["text"])) for b in selected
        ),
        "image_references_not_loaded": sum(b["text"].count("![") for b in selected),
        "input_blocks": len(blocks),
        "selected_blocks": len(selected),
        "discarded": [
            {"block_id": b["block_id"], "reason": "nonengine_navigation_or_fixed_source_cap"}
            for b in blocks
            if b not in selected
        ],
        "equation_and_figure_recovery": "not established; no inferred facts",
    }


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    if (dataset / "queries.jsonl").exists():
        raise ValueError("Prepare sources before questions")
    p = dataset / "source_manifest.json"
    manifest = json.loads(p.read_text(encoding="utf-8"))
    for row in manifest["documents"]:
        if row["status"] not in {
            "captured_pending_scope_review",
            "adopted_saved_engine_scope_before_questions",
        }:
            raise ValueError("Capture incomplete")
        row["candidate_used"] = True
        row["status"] = "adopted_saved_engine_scope_before_questions"
    p.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    prep.select_blocks = select_blocks
    prep.POLICY = "source-only-engine-selection-v3@1"
    out = root / ".runtime/holdout-v3/prepared-v5"
    prep.run(dataset, out, root / ".runtime/holdout-v3/profile")
    p = out / "catalog.json"
    catalog = json.loads(p.read_text(encoding="utf-8"))
    catalog["namespace"] = "holdout-v3"
    for row in catalog["documents"]:
        row["document_id"] = row["document_id"].replace("holdout-v1-", "holdout-v3-", 1)
    p.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared new sources", catalog["totals"], flush=True)


if __name__ == "__main__":
    main()
