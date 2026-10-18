"""Reproducible source-span audit for the new gas-path research scope.

Heuristics only propose review candidates. No keyword or document-level engine
tag assigns an accepted gas-path scope. Unreviewed page/record content remains
needs_review even when a selected passage on that page has been source-audited.
The original frozen corpus and evaluation labels are read-only inputs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

LABELS = {"core", "directly_related", "auxiliary", "distractor", "needs_review"}
SIGNALS = {
    "compressor": r"compressor|压气机",
    "surge_stall": r"\bsurge\b|\bstall\b|喘振|失速",
    "combustion": r"combustion|燃烧室",
    "turbine_airfoil": r"turbine blade|nozzle vane|涡轮叶片|导叶",
    "inlet_distortion_icing": r"inlet.{0,30}(distort|ic)|进气.{0,20}(畸变|结冰)",
    "measurement": r"thermocouple|sensor|EGT|传感|热电偶",
    "mechanical_background": r"bearing|chip detector|fatigue|starter|轴承|疲劳",
    "reciprocating_candidate": r"carburetor|reciprocating|活塞式|化油器",
}
DEFINITIONS = {
    "core": (
        "Explicit gas-path mechanism, component damage/performance, symptoms tied to it, or "
        "associated inspection/mitigation; applies to the audited passage only."
    ),
    "directly_related": (
        "Controls, measurement, fuel supply, applicability or observed symptoms directly "
        "needed to interpret a gas-path problem, without proving a gas-path mechanism."
    ),
    "auxiliary": (
        "Engine-health, mechanical reliability, general principles or document governance "
        "background outside the core mechanism."
    ),
    "distractor": (
        "Source content outside the turbine gas-path scope that can confuse retrieval; "
        "retain separately for robustness."
    ),
    "needs_review": (
        "Unreviewed scope, mixed/ambiguous content, or structure/figure quality needing "
        "independent verification."
    ),
}
COVERAGE_GAPS = [
    {
        "topic": "quantitative_compressor_performance",
        "status": "not_established_in_audited_spans",
        "need": (
            "Independent source with compressor-map/efficiency/flow-capacity shifts and "
            "corrected-speed conditions; training prose alone is insufficient."
        ),
    },
    {
        "topic": "gas_path_leakage_quantification",
        "status": "partial_inspection_only",
        "need": (
            "Seal-wear inspection exists; seek evidence linking a stated leak location to "
            "pressure/temperature/flow symptoms and limits."
        ),
    },
    {
        "topic": "combustion_instability_mechanism",
        "status": "inspection_and_starting_only",
        "need": (
            "Find combustion instability/flame-pattern mechanisms and discriminating "
            "checks, beyond liner damage and nozzle care."
        ),
    },
    {
        "topic": "turbine_performance_degradation",
        "status": "damage_inspection_only",
        "need": (
            "Find efficiency/clearance/erosion performance effects and operating "
            "conditions; structural disk/hub fatigue does not fill this gap."
        ),
    },
    {
        "topic": "controlled_model_specific_limits",
        "status": "not_established",
        "need": (
            "Current applicable approved manuals/ICA or public model-specific instructions; "
            "historical excerpts do not supply current approval."
        ),
    },
    {
        "topic": "real_maintenance_ground_truth",
        "status": "not_available",
        "need": (
            "Confirmed field faults and outcomes are absent; evaluate source knowledge "
            "handling rather than actual diagnosis accuracy."
        ),
    },
]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def load_sources(manifest: Path, repo: Path) -> list[dict[str, Any]]:
    result = []
    seen = set()
    for row in json.loads(manifest.read_text(encoding="utf-8"))["documents"]:
        docid = row["document_id"]
        if docid in seen:
            raise ValueError(f"duplicate source {docid}")
        seen.add(docid)
        data = {}
        for path_key, hash_key in (
            ("raw_path", "raw_sha256"),
            ("processed_path", "canonical_sha256"),
            ("blocks_path", "blocks_sha256"),
        ):
            path = Path(row[path_key])
            data[path_key] = (path if path.is_absolute() else repo / path).read_bytes()
            if digest(data[path_key]) != row[hash_key]:
                raise ValueError(f"{docid}: {hash_key} mismatch")
        if len(data["raw_path"]) != row["raw_bytes"]:
            raise ValueError(f"{docid}: raw byte count mismatch")
        text = data["processed_path"].decode("utf-8")
        blocks = [json.loads(line) for line in data["blocks_path"].decode("utf-8").splitlines()]
        prior = 0
        for index, block in enumerate(blocks):
            start, end = block["canonical_start"], block["canonical_end"]
            if not (prior <= start < end <= len(text) and text[start:end] == block["text"]):
                raise ValueError(f"{docid}: invalid block {index}")
            if (
                block["raw_sha256"] != row["raw_sha256"]
                or block["canonical_sha256"] != row["canonical_sha256"]
                or digest(block["text"].encode()) != block["text_sha256"]
            ):
                raise ValueError(f"{docid}: block {index} fingerprint mismatch")
            prior = end
        result.append({**row, "text": text, "blocks": blocks})
    return result


def validate_ranges(rows: list[dict[str, Any]], sources: list[dict[str, Any]]) -> None:
    by_id = {s["document_id"]: s for s in sources}
    seen = set()
    for row in rows:
        if row["range_id"] in seen or row["scope_label"] not in LABELS:
            raise ValueError("duplicate range or invalid scope")
        seen.add(row["range_id"])
        if row["scope_unit"] != "evidence_span_not_whole_block":
            raise ValueError("scope must apply to a verified passage, not a whole page/document")
        if not row["rationale"].strip() or not row["topic_tags"]:
            raise ValueError("scope needs a source-based rationale and topics")
        e = row["evidence"]
        if row["document_id"] != e["document_id"] or row["block_index"] != e["block_index"]:
            raise ValueError("range and evidence source identities disagree")
        source = by_id[e["document_id"]]
        if not isinstance(e["block_index"], int) or not 0 <= e["block_index"] < len(
            source["blocks"]
        ):
            raise ValueError("invalid evidence block index")
        block = source["blocks"][e["block_index"]]
        start, end = e["canonical_start"], e["canonical_end"]
        if not block["canonical_start"] <= start < end <= block["canonical_end"]:
            raise ValueError("range outside its source block")
        if (
            e["canonical_sha256"] != source["canonical_sha256"]
            or e["raw_sha256"] != source["raw_sha256"]
            or e["source_locator"] != block["source_locator"]
            or digest(source["text"][start:end].encode()) != e["text_sha256"]
        ):
            raise ValueError("range source evidence mismatch")


def union_length(spans: list[tuple[int, int]]) -> int:
    total = 0
    end = -1
    for left, right in sorted(spans):
        total += max(0, right - max(left, end))
        end = max(end, right)
    return total


def candidate_signals(text: str) -> list[str]:
    return [name for name, pattern in SIGNALS.items() if re.search(pattern, text, re.IGNORECASE)]


def build_inventory(
    sources: list[dict[str, Any]], ranges: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    validate_ranges(ranges, sources)
    blocks_out, sources_out = [], []
    for source in sources:
        selected = [r for r in ranges if r["document_id"] == source["document_id"]]
        scope_counts = Counter(r["scope_label"] for r in selected)
        processor = source["artifact_description"]["processor"]
        for index, block in enumerate(source["blocks"]):
            local = [r for r in selected if r["block_index"] == index]
            blocks_out.append(
                {
                    "document_id": source["document_id"],
                    "family_id": source["family_id"],
                    "block_index": index,
                    "source_locator": block["source_locator"],
                    "canonical_start": block["canonical_start"],
                    "canonical_end": block["canonical_end"],
                    "text_sha256": block["text_sha256"],
                    "canonical_sha256": source["canonical_sha256"],
                    "raw_sha256": source["raw_sha256"],
                    "block_scope": "needs_review",
                    "block_scope_reason": (
                        "Only explicitly annotated evidence spans are audited; remaining "
                        "block content is not promoted."
                    ),
                    "audited_range_ids": [r["range_id"] for r in local],
                    "candidate_topics": candidate_signals(block["text"]),
                    "candidate_status": "heuristic_for_review_only",
                    "character_count": len(block["text"]),
                }
            )
        audited_chars = union_length(
            [(r["evidence"]["canonical_start"], r["evidence"]["canonical_end"]) for r in selected]
        )
        sources_out.append(
            {
                key: source[key]
                for key in (
                    "document_id",
                    "logical_document_id",
                    "family_id",
                    "title",
                    "publisher_id",
                    "language",
                    "source_type",
                    "source_url",
                    "raw_path",
                    "processed_path",
                    "blocks_path",
                    "raw_sha256",
                    "canonical_sha256",
                    "blocks_sha256",
                    "rights",
                )
            }
            | {
                "format": processor,
                "structural_blocks": len(source["blocks"]),
                "canonical_characters": len(source["text"]),
                "audited_span_characters": audited_chars,
                "unreviewed_characters": len(source["text"]) - audited_chars,
                "audited_range_count": len(selected),
                "range_scope_counts": dict(scope_counts),
                "source_scope": "mixed_or_not_fully_audited",
                "selection": "core_passages_available"
                if scope_counts["core"]
                else (
                    "related_passages_available"
                    if scope_counts["directly_related"]
                    else "background_or_pending"
                ),
                "format_quality": (
                    "PDF plain-text; only annotated prose verified; other layout requires review"
                    if processor == "pdf"
                    else "HTML structural metadata retained; scope reviewed only at selected spans"
                    if processor == "html"
                    else (
                        "XML table headers restored; numeric table roles still require "
                        "span-level checking"
                    )
                    if processor == "xml"
                    else (
                        "CSV record fields retained; reporter certainty/codes and "
                        "duplicate/amended events require review"
                    )
                ),
                "human_expert_reviewed": False,
            }
        )
    return sources_out, blocks_out


def map_tasks(tasks: list[dict[str, Any]], ranges: list[dict[str, Any]]) -> list[dict[str, Any]]:
    mapped = []
    for task in tasks:
        evidence_matches = []
        for e in task["evidence"]:
            matches = []
            for row in ranges:
                a = row["evidence"]
                if a["document_id"] != e["document_id"]:
                    continue
                if max(a["canonical_start"], e["canonical_start"]) < min(
                    a["canonical_end"], e["canonical_end"]
                ):
                    matches.append(
                        {
                            "range_id": row["range_id"],
                            "scope_label": row["scope_label"],
                            "full_span_contained": a["canonical_start"] <= e["canonical_start"]
                            and e["canonical_end"] <= a["canonical_end"],
                            "quality_status": row["quality_status"],
                        }
                    )
            evidence_matches.append({"document_id": e["document_id"], "matches": matches})
        reusable = bool(evidence_matches) and all(
            any(
                m["full_span_contained"]
                and m["scope_label"] in {"core", "directly_related"}
                and m["quality_status"] == "prose_source_verified"
                for m in item["matches"]
            )
            for item in evidence_matches
        )
        mapped.append(
            {
                "sample_id": task["sample_id"],
                "fact_id": task["fact_id"],
                "old_split": task["split"],
                "evidence_scope_candidates": evidence_matches,
                "recommendation": "reuse_fact_candidate_requires_new_review"
                if reusable
                else "needs_reauthor_or_scope_review",
                "old_annotation_unchanged": True,
            }
        )
    return mapped


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    cli.add_argument(
        "--manifest",
        type=Path,
        default=Path("evaluation/aero_evidence_benchmark_v1/corpus_manifest.json"),
    )
    cli.add_argument("--output", type=Path, default=Path("evaluation/gas_path_scope_v1"))
    args = cli.parse_args()
    repo = args.repo.resolve()
    manifest = args.manifest if args.manifest.is_absolute() else repo / args.manifest
    output = args.output if args.output.is_absolute() else repo / args.output
    ranges_path = output / "range_annotations.jsonl"
    ranges = read_jsonl(ranges_path)
    sources = load_sources(manifest, repo)
    source_inventory, block_inventory = build_inventory(sources, ranges)
    tasks = read_jsonl(manifest.parent / "queries.jsonl")
    mappings = map_tasks(tasks, ranges)
    write_jsonl(output / "source_inventory.jsonl", source_inventory)
    write_jsonl(output / "block_inventory.jsonl", block_inventory)
    write_jsonl(output / "task_mapping.jsonl", mappings)
    summary = {
        "schema_version": 1,
        "audit_id": "gas_path_scope_v1",
        "manifest_sha256": digest(manifest.read_bytes()),
        "range_annotations_sha256": digest(ranges_path.read_bytes()),
        "script_sha256": digest(Path(__file__).read_bytes()),
        "sources": len(sources),
        "logical_documents": len({s["logical_document_id"] for s in sources}),
        "families": len({s["family_id"] for s in sources}),
        "structural_blocks": len(block_inventory),
        "canonical_characters": sum(len(s["text"]) for s in sources),
        "audited_ranges": len(ranges),
        "audited_range_labels": dict(Counter(r["scope_label"] for r in ranges)),
        "audited_span_characters": sum(s["audited_span_characters"] for s in source_inventory),
        "unreviewed_characters": sum(s["unreviewed_characters"] for s in source_inventory),
        "sources_with_core_ranges": [
            s["document_id"]
            for s in source_inventory
            if s["selection"] == "core_passages_available"
        ],
        "format_counts": dict(Counter(s["format"] for s in source_inventory)),
        "audited_topic_counts": dict(Counter(t for r in ranges for t in r["topic_tags"])),
        "old_task_mapping": dict(Counter(m["recommendation"] for m in mappings)),
        "coverage_gaps": COVERAGE_GAPS,
        "limits": [
            "AI source-scope audit, not human or expert confirmation.",
            "Purposive seed spans; no random quality estimate or exhaustive scope coverage.",
            (
                "Heuristic candidate topics never assign "
                "core/directly-related/auxiliary/distractor labels."
            ),
            "Gas turbine and engine document tags do not imply gas-path relevance.",
            "All unreviewed portions remain needs_review; no whole-document purity claim.",
            "Coverage gaps describe audited spans, not proof of absence from the entire corpus.",
            (
                "Current model-specific maintenance limits and actual field-diagnosis truth "
                "remain unavailable."
            ),
            "Old tasks are mapped for review only; existing benchmark and split are unchanged.",
        ],
    }
    (output / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (output / "protocol.json").write_text(
        json.dumps(
            {
                "definitions": DEFINITIONS,
                "scope_unit": "evidence_span_not_whole_block",
                "review_status": "ai_source_scope_audited",
                "human_expert_reviewed": False,
                "publication": (
                    "Source offsets/hashes and audit reasons only; no canonical/full raw "
                    "text redistributed."
                ),
                "source_selection": (
                    "Use core/directly-related verified spans for primary tasks; auxiliary "
                    "and distractors separately; needs_review isolated."
                ),
                "next_review": (
                    "Independent source review, then source-family grouping and new "
                    "task/evidence-pool review before a new frozen benchmark."
                ),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                k: summary[k]
                for k in (
                    "sources",
                    "structural_blocks",
                    "audited_ranges",
                    "audited_range_labels",
                    "old_task_mapping",
                )
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
