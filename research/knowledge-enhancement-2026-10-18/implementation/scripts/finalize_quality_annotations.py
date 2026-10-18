"""Reconcile source-reviewed v3 labels and bind the protocol before predictions."""

from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path

from author_quality_questions import locate_quote
from evaluate_quality_holdout import read, rows, save

from aerodiagnosis.domain.atomic_annotation import AtomicAnnotation


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    if (dataset / "freeze.json").exists():
        raise ValueError("Cannot change frozen questions")
    contract = read(dataset / "development-code-contract.json")
    if any(
        hashlib.sha256((root / f).read_bytes()).hexdigest() != sha
        for f, sha in contract["code_sha256_bytes"].items()
    ):
        raise ValueError("Algorithms changed after independent question review began")
    original = dataset / "queries-before-independent-review.jsonl"
    if not original.exists():
        original.write_bytes((dataset / "queries.jsonl").read_bytes())
    qs = rows(original)
    reconstruction = root / ".runtime/holdout-v3/answer-hidden/reconstruction.jsonl"
    recon = rows(reconstruction)
    review_path = root / ".runtime/holdout-v3/label-reconciliation.json"
    review = read(review_path)
    decisions = review["items"]
    by = {r["sample_id"]: r for r in decisions}
    if len(recon) != 120 or len(decisions) != 120 or set(by) != {q["sample_id"] for q in qs}:
        raise ValueError("Complete hidden reconstruction and visible-label review required")
    prepared = root / ".runtime/holdout-v3/prepared-v5"
    sources = {
        r["original_source_id"]: read(prepared / r["parsed_path"])["canonical_content"]
        for r in read(prepared / "catalog.json")["documents"]
    }
    changed_questions, changed_annotations, resolutions = [], [], []
    for q in qs:
        decision = by[q["sample_id"]]
        if decision["accepted"] is not True:
            raise ValueError("Unresolved independent review: " + q["sample_id"])
        if decision.get("query") and decision["query"] != q["query"]:
            q["query"] = decision["query"]
            changed_questions.append(q["sample_id"])
        if decision.get("annotation"):
            annotation = decision["annotation"]
            if annotation != {
                k: q[k] for k in ["question_slots", "required_facts", "supplementary_facts"]
            }:
                changed_annotations.append(q["sample_id"])
            source = sources[q["document_id"]]
            for kind in ["required_facts", "supplementary_facts"]:
                annotation.setdefault(kind, [])
                for i, f in enumerate(annotation[kind]):
                    start, quote, norm = locate_quote(source, f["source_quote"])
                    f.update(
                        source_quote=quote,
                        source_start=start,
                        source_end=start + len(quote),
                        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
                    )
                    f.setdefault("fact_key", q["sample_id"] + "-" + kind + "-" + str(i + 1))
                    resolutions.append(
                        {
                            "sample_id": q["sample_id"],
                            "fact_key": f["fact_key"],
                            "normalization": norm,
                        }
                    )
            validated = AtomicAnnotation.model_validate(annotation)
            validated.verify_source(source)
            q.update(validated.model_dump(mode="json"))
        if q["document_id"] in {"aaib-g-cgtc", "aaib-g-leap"}:
            q["correlation_group"] = "aaib-engine-icing"
        q.update(
            reference_answer=" ".join(f["text"] for f in q["required_facts"]),
            minimal_facts=q["required_facts"],
            review_status="accepted_after_hidden_source_reconstruction_and_label_reconciliation",
        )
    import json

    (dataset / "queries.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in qs), encoding="utf-8"
    )
    (dataset / "generation_queries.jsonl").write_text(
        "".join(
            json.dumps(
                {k: q[k] for k in ["sample_id", "query", "query_language"]}, ensure_ascii=False
            )
            + "\n"
            for q in qs
        ),
        encoding="utf-8",
    )
    sha = hashlib.sha256((dataset / "queries.jsonl").read_bytes()).hexdigest()
    details = {
        "before_predictions": True,
        "human_expert_reviewed": False,
        "source_reconstruction_sha256": hashlib.sha256(reconstruction.read_bytes()).hexdigest(),
        "label_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "question_revisions": changed_questions,
        "annotation_revisions": changed_annotations,
        "source_resolutions": resolutions,
        "final_queries_sha256": sha,
    }
    save(dataset / "annotation-reconciliation.json", details)
    save(dataset / "question-reconciliation-draft.json", details)
    save(
        dataset / "final-acceptance.json",
        {**details, "accepted_sample_ids": list(by), "unresolved": 0},
    )
    save(
        dataset / "reconciliation.json",
        {
            **details,
            "queries_reviewed": 120,
            "ai_source_accepted": 120,
            "system_predictions_seen": False,
            "source_text_layer": 90,
            "summary_auxiliary_layer": 30,
            "required_fact_count": sum(len(q["required_facts"]) for q in qs),
        },
    )
    helper_names = [
        "prepare_quality_scoring.py",
        "finalize_quality_annotations.py",
        "prepare_quality_review.py",
        "summarize_quality_comparison.py",
        "validate_quality_annotations.py",
        "adopt_quality_review.py",
    ]
    protocol = {
        "status": "ready_for_freeze",
        "questions": 120,
        "primary_source_text": 90,
        "summary_auxiliary": 30,
        "primary_sources": 18,
        "auxiliary_sources": 6,
        "required_atomic_facts": sum(len(q["required_facts"]) for q in qs),
        "task_types": dict(Counter(q["task_type"] for q in qs)),
        "arms": ["text_only", "text_plus_graph"],
        "top_k": 5,
        "evidence_token_budget": 2048,
        "budget_counts": "whole evidence catalogue including selectors and source metadata",
        "ranking_policy": "grounded_source_rank@4",
        "embedding": "multilingual-e5-small pinned ONNX CPU",
        "citation_policy": "source-span-and-uncertainty@2",
        "coverage": "independent verifier plus index guards",
        "bounded_source_only_gate_repair": 1,
        "bounded_source_only_coverage_repair": 1,
        "primary_judge": (
            "all 240 answers reviewed by anonymous independent nonexpert AI with original source"
        ),
        "paid_pro_audit": "first question per source; 24 pairs; auxiliary grading cross-check only",
        "generation_model": "deepseek-flash",
        "temperature": 0,
        "case_route": False,
        "source_scope": "explicit document reference in query; same scope in both arms",
        "bootstrap_repetitions": 10000,
        "bootstrap_seed": 20261014,
        "interval_method": "paired whole-source cluster bootstrap",
        "conservative_correlation_groups": (
            "NASA six abstracts together; AAIB two icing sources together"
        ),
        "test_tuning_prohibited": True,
        "cumulative_cap_cny": 35,
        "ledger_reset": False,
        "algorithm_contract_sha256": hashlib.sha256(
            (dataset / "development-code-contract.json").read_bytes()
        ).hexdigest(),
        "analysis_helpers_sha256": {
            name: hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest()
            for name in helper_names
        },
        "limitations": [
            "curated 18 primary sources; 90 questions are not 90 independent samples",
            "30 NASA publisher abstracts separate from full-source-text analysis",
            "new project sources, not guaranteed absent from model pretraining",
            "source scope selected before questions; figures/formulas not fully reconstructed",
            "AI nonexpert source review, not real maintenance diagnostic accuracy",
            "single generation per arm; model calls can vary even for identical contexts",
            "related NASA and icing sources receive conservative grouped sensitivity",
            "graph extends source candidates; no causal path composition",
        ],
    }
    save(dataset / "protocol.json", protocol)
    print(
        "Final independent labels",
        len(qs),
        "question revisions",
        len(changed_questions),
        "annotation revisions",
        len(changed_annotations),
        "facts",
        protocol["required_atomic_facts"],
    )


if __name__ == "__main__":
    main()
