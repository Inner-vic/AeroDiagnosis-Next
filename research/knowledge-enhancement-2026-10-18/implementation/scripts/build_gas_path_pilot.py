"""Freeze the independently reconstructed gas-path development pilot, never test gold."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import build_academic_benchmark as builder


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--candidate", type=Path, default=Path("evaluation/gas_path_benchmark_v2_candidate")
    )
    parser.add_argument("--reviews", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("evaluation/gas_path_pilot_v1"))
    parser.add_argument("--work", type=Path, default=Path(".runtime/gas-path-pilot-preparation-v1"))
    args = parser.parse_args()
    if args.output.exists() or args.work.exists():
        raise ValueError("Use new immutable output and preparation directories")
    tasks = builder.read_jsonl(args.candidate / "tasks.jsonl")
    reviews = builder.read_jsonl(args.reviews)
    review_map = {r["sample_id"]: r for r in reviews}
    if len(review_map) != len(reviews) or set(review_map) != {t["sample_id"] for t in tasks}:
        raise ValueError("Exactly one independently reconstructed review is required per task")
    sources = builder.load_sources(args.candidate / "corpus_manifest.json")
    raw, reconciliations, portable_reviews = [], [], []
    for task in tasks:
        review = review_map[task["sample_id"]]
        if review["blind_sha256"] != builder.annotation_hash(builder.blind_view(task)):
            raise ValueError("Review is not bound to this blind question")
        if review["reconstructed_answerability"] != task["answerability"]:
            raise ValueError("Answerability disagreement must be adjudicated separately")
        if review["reviewer"] == task["author"] or review["human_expert_reviewed"] is not False:
            raise ValueError("Invalid independent AI reviewer role")
        if review["human_source_reviewed"] is not False:
            raise ValueError("No real human source review took place in this pilot")
        for evidence in review["evidence"]:
            rebuilt = builder.materialize_span(evidence, sources)
            if any(rebuilt[k] != evidence[k] for k in rebuilt):
                raise ValueError("Independent review source fingerprint mismatch")
        record = {
            k: v
            for k, v in task.items()
            if k
            not in {"evidence", "split", "family_ids", "review_status", "human_expert_reviewed"}
        }
        # The independent reconstruction identifies the essential model mismatch;
        # exact historical procedure IDs are useful context, not mandatory facts.
        if record["sample_id"] == "gp-q019":
            record["minimal_facts"] = [
                {
                    "fact_key": "gp-f019-1",
                    "text": (
                        "A LEAP-1A32 procedure was selected for an aircraft "
                        "fitted with CFM56-5B3/3 engines."
                    ),
                    "allowed_range_ids": ["gp-r028"],
                    "numerical_tolerance": None,
                },
                {
                    "fact_key": "gp-f019-2",
                    "text": (
                        "This is a historical incident finding, "
                        "not a current maintenance instruction."
                    ),
                    "allowed_range_ids": ["gp-r028"],
                    "numerical_tolerance": None,
                },
            ]
        record["reference_answer"] = " ".join(f["text"] for f in record["minimal_facts"])
        record["engine_domain"] = "gas_turbine"
        record["task_type"] = "source_scoped_knowledge"
        record["paired_variant_of"] = None
        record["hard_negatives"] = []
        record["evidence"] = [
            {
                "document_id": e["document_id"],
                "block_index": e["block_index"],
                "anchor_text": sources[e["document_id"]]["text"][
                    e["canonical_start"] : e["canonical_end"]
                ],
            }
            for e in task["evidence"]
        ]
        reason = (
            "Independent source reconstruction agrees on required facts; "
            "author narrow spans retained after exact source recheck."
        )
        if record["sample_id"] == "gp-q019":
            reason += (
                " Historical procedure identifiers retained as optional context, "
                "not required scoring facts."
            )
        reconciliations.append(
            {
                "sample_id": record["sample_id"],
                "adjudicator": "codex:root",
                "decision": "accept",
                "disposition": "agreed_with_source_reconstruction",
                "reason": reason,
                "final_annotation_sha256": builder.annotation_hash(record),
                "human_expert_reviewed": False,
            }
        )
        raw.append(record)
        portable_reviews.append(
            {
                **review,
                "evidence": [
                    {k: v for k, v in e.items() if k != "anchor_text"} for e in review["evidence"]
                ],
            }
        )
    args.output.mkdir(parents=True)
    args.work.mkdir(parents=True)
    families = sorted({f for t in tasks for f in t["family_ids"]})
    manifest = json.loads((args.candidate / "corpus_manifest.json").read_text(encoding="utf-8"))
    protocol = {
        "benchmark_id": "gas-path-development-pilot-v1",
        "protocol_version": "1.0",
        "frozen_before_results": True,
        "evaluation_role": "development_debugging_only_no_test_or_generalization_claim",
        "dev_families": families,
        "source_dependency_groups": [],
        "upstream_snapshot_sha256": manifest["upstream_snapshot_sha256"],
        "shared_corpus": (
            "All 21 frozen source files retained, including background "
            "and distractors; only source text is indexed."
        ),
        "prior_exposure": (
            "Author read old scope audit and sources; "
            "review answers hidden from independent AI reviewer."
        ),
        "review_policy": "AI source reconstruction; not human or expert gold",
        "pilot_only": True,
        "statistical_independence": (
            "25 fact IDs grouped by four source families; not IID observations"
        ),
        "scope_counts": {"core": 20, "directly_related": 5},
        "registered_parameters": {
            "chunk_chars": 800,
            "overlap_chars": 100,
            "parent_chars": 3200,
            "candidate_k": 100,
            "rrf_k": 60,
            "context_tokens": 2048,
            "max_chunks": 10,
        },
        "main_comparison": (
            "Twelve dev configurations: three chunkers times BM25/E5/RRF "
            "plus paired hierarchical parent completion."
        ),
        "rank_qrels_exhaustive": False,
        "graph_extraction_gold": False,
        "missing_tasks": [
            "insufficient_evidence",
            "wrong_premise",
            "condition_conflict",
            "independent_new_source_test",
        ],
        "input_review_sha256": builder.digest(args.reviews.read_bytes()),
        "review_provenance_sha256": builder.digest(args.provenance.read_bytes()),
    }
    builder.write_json(args.output / "protocol.json", protocol)
    builder.write_json(
        args.output / "annotation_provenance.json",
        json.loads(args.provenance.read_text(encoding="utf-8")),
    )
    builder.write_jsonl(args.output / "review_ledger.jsonl", portable_reviews)
    builder.write_jsonl(args.output / "reconciliation.jsonl", reconciliations)
    (args.output / "DATA_CARD.md").write_text(
        """# Gas-path development pilot v1

25 source-scoped English development questions; 20 core and 5 directly related tasks.
Four task source families; all 21 immutable corpus files remain retrieval inputs.
All pilot questions are development-only and currently answerable. No held-out test,
expert gold, complete relevance judgments, extraction-recall gold or field diagnosis labels.
Independent AI reconstruction received only question, language, context and full source catalog.
The author saw old source/scope material. Exact source offsets and SHA-256 remain auditable.
Historical training/incident excerpts do not supply current maintenance authority.

The frozen pilot supports offline debugging and dev-only configuration selection.
It does not satisfy the complete S2 dataset gate or authorize confirmatory S5 reporting.
Full source text remains local; review evidence is published as locator and hashes only.
Raw reconstruction anchors reside in local runtime preparation inputs.
""",
        encoding="utf-8",
    )
    builder.write_jsonl(args.work / "engineering_queries.jsonl", raw)
    builder.write_jsonl(args.work / "engineering_reviews.jsonl", reviews)
    builder.write_jsonl(args.work / "investigation_reviews.jsonl", [])
    builder.write_jsonl(args.work / "reconciliation.jsonl", reconciliations)
    result = builder.build(
        args.candidate / "corpus_manifest.json",
        args.work,
        args.output / "protocol.json",
        args.output,
        review_dir=args.work,
    )
    print(
        json.dumps(
            {
                "build": result,
                "validation": builder.validate_release(args.output, args.output / "protocol.json"),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
