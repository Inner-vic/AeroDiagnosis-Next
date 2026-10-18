"""Check reviewed evidence survives source preparation; no query retrieval or predictions."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt


def normalized(text: str) -> str:
    renderer = MarkdownIt("commonmark", {"html": False}).enable("table")
    plain = BeautifulSoup(renderer.render(text), "html.parser").get_text(" ")
    plain = re.sub(r"^\s*(?:\d+[.)]|[*•-])\s+", "", plain)
    plain = plain.split("Abstract:")[0] if plain.lstrip().startswith("摘要") else plain
    plain = unicodedata.normalize("NFKC", plain).replace("\u00ad", "").replace("\u200b", "")
    plain = plain.translate(
        str.maketrans(
            {
                "\u2019": "'",
                "\u2018": "'",
                "“": '"',
                "”": '"',
                "\u2010": "-",
                "\u2013": "-",
                "—": "-",
            }
        )
    )
    plain = re.sub(r"\s+", "", plain)
    return re.sub(r"(?<=[A-Za-z])-(?=[A-Za-z])", "", plain)


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--dataset", type=Path, default=Path("evaluation/gas_path_holdout_v1_candidate")
    )
    cli.add_argument("--prepared", type=Path, default=Path(".runtime/holdout-v1/prepared"))
    args = cli.parse_args()
    questions = [
        json.loads(s)
        for s in (args.dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    catalog = json.loads((args.prepared / "catalog.json").read_text(encoding="utf-8"))
    by = {d["original_source_id"]: d for d in catalog["documents"]}
    loaded = {}
    results = []
    for row in questions:
        checks = []
        for ev in row["evidence"]:
            doc = ev["document_id"]
            if doc not in loaded:
                text = (args.dataset / "sources" / f"{doc}.txt").read_text(encoding="utf-8")
                parsed = json.loads(
                    (args.prepared / by[doc]["parsed_path"]).read_text(encoding="utf-8")
                )
                loaded[doc] = (text, normalized(parsed["canonical_content"]), parsed)
            text, canonical, parsed = loaded[doc]
            original = text[ev["canonical_start"] : ev["canonical_end"]]
            if hashlib.sha256(original.encode()).hexdigest() != ev["text_sha256"]:
                raise ValueError("Gold evidence snapshot changed")
            needle = normalized(original)
            checks.append(
                {
                    "block_id": ev["block_id"],
                    "whole_evidence_block_survives_normalized": needle in canonical,
                    "matched_chunk_indices": [
                        i
                        for i, c in enumerate(parsed["chunks"])
                        if needle in normalized(c["content"])
                    ],
                }
            )
        results.append(
            {
                "sample_id": row["sample_id"],
                "source_layer": row["source_layer"],
                "all_bound_evidence_blocks_survive": all(
                    c["whole_evidence_block_survives_normalized"] for c in checks
                ),
                "checks": checks,
            }
        )
    out = {
        "status": "preparation_coverage_only_not_retrieval_score",
        "questions": len(results),
        "all_bound_evidence_blocks_survive": sum(
            x["all_bound_evidence_blocks_survive"] for x in results
        ),
        "unmatched_questions": [
            x["sample_id"] for x in results if not x["all_bound_evidence_blocks_survive"]
        ],
        "results": results,
        "normalization": (
            "Markdown visible text + NFKC + punctuation/whitespace/list marker normalization; "
            "Chinese abstract projection excludes duplicate English abstract, no inferred facts"
        ),
        "llm_calls": 0,
        "system_predictions": 0,
    }
    (args.prepared / "evidence-coverage.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
