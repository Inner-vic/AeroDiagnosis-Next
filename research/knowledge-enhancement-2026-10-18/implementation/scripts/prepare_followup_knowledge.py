"""Prepare independently captured source bodies before annotations exist."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import prepare_holdout_knowledge as preparation
from prepare_holdout_knowledge import run
from prepare_holdout_knowledge import select_blocks as original_select


def select_blocks(row, blocks):
    if row["publisher"] == "NASA":
        # NTRS's title/abstract block; neither navigation nor metadata is a full paper.
        chosen = [
            b for b in blocks if len(b["text"]) > 500 and not b["text"].startswith(("![", "#"))
        ]
        chosen = chosen[:1]
        return chosen, {
            "input_blocks": len(blocks),
            "selected_blocks": len(chosen),
            "discarded": [
                {"block_id": b["block_id"], "reason": "publisher_metadata_or_navigation"}
                for b in blocks
                if b not in chosen
            ],
            "image_references_not_loaded": 0,
            "private_use_glyph_blocks": 0,
        }
    if row["document_id"] == "ao-2010-089":
        # Only the captured final-report summary, never mix later interim headings.
        chosen = [b for b in blocks[4:] if b["start"] < blocks[15]["start"]]
        return chosen, {
            "input_blocks": len(blocks),
            "selected_blocks": len(chosen),
            "discarded": [
                {"block_id": b["block_id"], "reason": "not_final_report_summary"}
                for b in blocks
                if b not in chosen
            ],
            "image_references_not_loaded": 0,
            "private_use_glyph_blocks": 0,
        }
    chosen, audit = original_select(row, blocks)
    # A source-side engine chapter selection does not use questions or gold.
    if row["document_id"] in {"a12p0008", "a08p0125"}:
        keep = []
        relevant = False
        for b in chosen:
            if b["text"].startswith("#"):
                title = b["text"].lower()
                relevant = any(
                    t in title
                    for t in [
                        "engine",
                        "compressor",
                        "blade",
                        "fuel control",
                        "power turbine",
                        "findings",
                        "safety action",
                    ]
                )
            if relevant:
                keep.append(b)
        audit["source_body_scope"] = "all_engine_system_sections_and_findings"
        audit["selected_blocks"] = len(keep)
        audit["discarded"].extend(
            {"block_id": b["block_id"], "reason": "non_engine_section_source_only_selection"}
            for b in chosen
            if b not in keep
        )
        return keep, audit
    return chosen, audit


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument(
        "--source-root", type=Path, default=Path("evaluation/gas_path_holdout_v2_candidate")
    )
    cli.add_argument("--output", type=Path, default=Path(".runtime/holdout-v2/prepared"))
    args = cli.parse_args()
    if (args.source_root / "queries.jsonl").exists():
        raise ValueError("Source preparation must precede question authoring")
    preparation.select_blocks = select_blocks
    run(
        args.source_root.resolve(),
        args.output.resolve(),
        Path(".runtime/holdout-v2/profile").resolve(),
    )
    catalog_path = args.output / "catalog.json"
    catalog = json.loads(catalog_path.read_text(encoding="utf-8"))
    catalog["namespace"] = "holdout-v2"
    for row in catalog["documents"]:
        row["document_id"] = row["document_id"].replace("holdout-v1-", "holdout-v2-", 1)
    catalog_path.write_text(json.dumps(catalog, ensure_ascii=False, indent=2), encoding="utf-8")
    print("Prepared totals", catalog["totals"], flush=True)


if __name__ == "__main__":
    main()
