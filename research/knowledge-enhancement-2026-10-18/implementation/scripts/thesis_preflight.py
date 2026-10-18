"""Read-only stage checks for thesis execution; never promote candidates to gold."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import build_academic_benchmark as builder


def validate_candidates(root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    candidate_sources = builder.load_sources(root / "corpus_manifest.json")
    if candidate_sources != sources:
        raise ValueError("Blind source catalog differs from the verified authoring corpus")
    tasks = builder.read_jsonl(root / "tasks.jsonl")
    blind = builder.read_jsonl(root / "blind_queries.jsonl")
    ledger = builder.read_jsonl(root / "review_ledger.jsonl")
    if not tasks or len({t["sample_id"] for t in tasks}) != len(tasks):
        raise ValueError("Candidate IDs must be nonempty and unique")
    expected = [builder.blind_view(t) for t in tasks]
    if blind != expected:
        raise ValueError("Blind packet differs from the allowed answer-hidden projection")
    if len(ledger) != len(tasks) or {r["sample_id"] for r in ledger} != {
        t["sample_id"] for t in tasks
    }:
        raise ValueError("Review ledger must contain exactly one entry per task")
    evidence_count = 0
    for task in tasks:
        if task["scope_label"] not in {"core", "directly_related"}:
            raise ValueError("Primary candidate outside gas-path scope")
        if task["human_expert_reviewed"] is not False:
            raise ValueError("Candidate authoring cannot assert expert confirmation")
        if not task["minimal_facts"] or not task["evidence"] or task["split"] != "unassigned":
            raise ValueError("Unreviewed candidates need facts, evidence and unassigned split")
        if (
            sorted({sources[e["document_id"]]["family_id"] for e in task["evidence"]})
            != task["family_ids"]
        ):
            raise ValueError("Family membership differs from source provenance")
        for span in task["evidence"]:
            source = sources[span["document_id"]]
            start, end = span["canonical_start"], span["canonical_end"]
            if not 0 <= start < end <= len(source["text"]):
                raise ValueError("Candidate span outside canonical snapshot")
            text = source["text"][start:end]
            rebuilt = builder.materialize_span(
                {
                    "document_id": span["document_id"],
                    "block_index": span["block_index"],
                    "anchor_text": text,
                },
                sources,
            )
            if rebuilt != span:
                raise ValueError("Candidate evidence fingerprint or locator mismatch")
            evidence_count += 1
    reviews = Counter(r["disposition"] for r in ledger)
    return {
        "tasks": len(tasks),
        "fact_ids": len({t["fact_id"] for t in tasks}),
        "source_families": len({f for t in tasks for f in t["family_ids"]}),
        "scope_counts": dict(Counter(t["scope_label"] for t in tasks)),
        "answerability_counts": dict(Counter(t["answerability"] for t in tasks)),
        "validated_evidence_references": evidence_count,
        "blind_packet_valid": True,
        "review_dispositions": dict(reviews),
        "status": "candidate_not_frozen",
        "statistical_independence_established": False,
    }


def validate_scope(root: Path, sources: dict[str, dict[str, Any]]) -> dict[str, Any]:
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    data = (root / "range_annotations.jsonl").read_bytes()
    if hashlib.sha256(data).hexdigest() != summary["range_annotations_sha256"]:
        raise ValueError("Scope annotation fingerprint mismatch")
    ranges = builder.read_jsonl(root / "range_annotations.jsonl")
    if len({r["range_id"] for r in ranges}) != len(ranges):
        raise ValueError("Duplicate scope range IDs")
    for row in ranges:
        span = row["evidence"]
        source = sources[span["document_id"]]
        start, end = span["canonical_start"], span["canonical_end"]
        if not 0 <= start < end <= len(source["text"]):
            raise ValueError("Scope span outside snapshot")
        anchor = {
            "document_id": span["document_id"],
            "block_index": span["block_index"],
            "anchor_text": source["text"][start:end],
        }
        if builder.materialize_span(anchor, sources) != span:
            raise ValueError("Scope evidence mismatch")
    return {
        "ranges": len(ranges),
        "labels": dict(Counter(r["scope_label"] for r in ranges)),
        "all_evidence_locators_valid": True,
        "human_expert_reviewed": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=Path("evaluation/aero_evidence_benchmark_v1")
    )
    parser.add_argument("--scope", type=Path, default=Path("evaluation/gas_path_scope_v1"))
    parser.add_argument(
        "--candidates", type=Path, default=Path("evaluation/gas_path_benchmark_v2_candidate")
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    # Compute every check before creating a receipt, and never overwrite a prior run.
    frozen = builder.validate_release(args.dataset, args.dataset / "protocol.json")
    sources = builder.load_sources(args.dataset / "corpus_manifest.json")
    scope = validate_scope(args.scope, sources)
    candidates = validate_candidates(args.candidates, sources)
    receipt = {
        "schema_version": "thesis-preflight@1",
        "frozen_v1": frozen,
        "scope": scope,
        "candidates": candidates,
        "api_calls": 0,
        "api_spend_cny": 0,
        "stage_gates": {
            "S0": "protocol_registered",
            "S1": "seed_spans_verified_not_full_quality_audit",
            "S2": "pending_independent_review_and_dependency_split",
            "S3": "not_started_on_new_dataset",
            "S4": "existing_contract_checks_passed_not_full_flow_acceptance",
            "S5": "not_authorized_by_stage_gates",
            "S6": "results_wait_for_real_runs",
        },
        "next_requirements": [
            "independent answer-hidden source reconstruction and adjudication",
            "insufficient-evidence and condition-conflict candidate tasks",
            "source dependency components and dev/test coverage audit",
            "exhaustive extraction subset before recall scoring",
            (
                "live provider model/pricing verification and configured "
                "environment credential before paid pilot"
            ),
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(receipt, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    print(
        json.dumps({"frozen": frozen, "scope": scope, "candidates": candidates}, ensure_ascii=False)
    )


if __name__ == "__main__":
    main()
