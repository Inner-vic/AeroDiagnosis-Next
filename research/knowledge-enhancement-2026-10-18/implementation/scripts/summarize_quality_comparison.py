"""All-task anonymous source verdicts with frozen citation/coverage gates and paired CIs."""

from __future__ import annotations

import hashlib
from pathlib import Path

from evaluate_quality_holdout import citation_gate, read, rows, save
from summarize_holdout_comparison import paired_summary


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    out = root / ".runtime/experiments/holdout-v3-final-20261014"
    qs = {q["sample_id"]: q for q in rows(dataset / "queries.jsonl")}
    reviews_path = out / "anonymous-source-adjudication.jsonl"
    reviews = rows(reviews_path)
    if len(reviews) != 120 or {r["sample_id"] for r in reviews} != set(qs):
        raise ValueError("All 120 unique paired independent reviews required")
    protocol = read(dataset / "protocol.json")
    for name, sha in protocol["analysis_helpers_sha256"].items():
        if hashlib.sha256((root / "scripts" / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Analysis helper changed after freeze: " + name)
    mapping = read(out / "source-review-arm-mapping.json")
    final = []
    for r in sorted(reviews, key=lambda r: r["sample_id"]):
        q = qs[r["sample_id"]]
        a = read(out / "answers" / (r["sample_id"] + ".json"))
        scores = {}
        for letter, name in mapping[r["sample_id"]].items():
            score = dict(r[letter])
            if any(
                type(score.get(k)) is not bool
                for k in ["correct", "unsupported_claim", "uncertainty_strengthened"]
            ):
                raise ValueError("Explicit source review booleans required")
            checks = score["required_fact_checks"]
            if (
                len(checks) != len(q["required_facts"])
                or {c["fact_key"] for c in checks} != {f["fact_key"] for f in q["required_facts"]}
                or any(type(c["covered"]) is not bool for c in checks)
            ):
                raise ValueError("Every frozen necessary fact must be reviewed once")
            score["strict_correct"] = bool(
                score["correct"]
                and all(c["covered"] for c in checks)
                and not score["unsupported_claim"]
                and not score["uncertainty_strengthened"]
                and citation_gate(a["arms"][name])
            )
            scores[name] = score
        final.append(
            {
                **a,
                "judge": scores,
                "source_layer": q["source_layer"],
                "source_cluster": q["statistical_cluster"],
                "correlation_group": q["correlation_group"],
                "task_type": q["task_type"],
            }
        )
    groups = {
        "primary_source_text": [r for r in final if r["source_layer"] == "source_text"],
        "summary_auxiliary": [r for r in final if r["source_layer"] != "source_text"],
        "all_descriptive": final,
    }
    if len(groups["primary_source_text"]) != 90 or len(groups["summary_auxiliary"]) != 30:
        raise ValueError("Frozen source subsets changed")
    result = {
        "basis": "all 240 independent anonymous AI source reviews plus frozen program gates",
        "human_expert_reviewed": False,
        "snapshot_sha256": read(out / "generation-complete.json")["snapshot_sha256"],
        "independent_review_sha256": hashlib.sha256(reviews_path.read_bytes()).hexdigest(),
        "subsets": {
            k: {unit: paired_summary(v, unit) for unit in ["source_cluster", "correlation_group"]}
            for k, v in groups.items()
        },
        "task_types_descriptive": {
            k: paired_summary([r for r in final if r["task_type"] == k], "source_cluster")
            for k in sorted({r["task_type"] for r in final})
        },
        "discordant_ids": [
            r["id"]
            for r in final
            if r["judge"]["text"]["strict_correct"] != r["judge"]["text_graph"]["strict_correct"]
        ],
        "same_packed_context_pairs": sum(
            r["arms"]["text"]["span_catalog"] == r["arms"]["text_graph"]["span_catalog"]
            for r in final
        ),
        "diagnostics": {
            name: {
                "semantic_correct_before_gates": sum(r["judge"][name]["correct"] for r in final),
                "span_gate_failures": sum(
                    not r["arms"][name]["answer"]["gate_valid"] for r in final
                ),
                "coverage_gate_failures": sum(not r["arms"][name]["coverage_valid"] for r in final),
                "exact_quote_failures": sum(
                    not all(r["arms"][name]["citation_exact_checks"]) for r in final
                ),
                "unsupported_claim": sum(r["judge"][name]["unsupported_claim"] for r in final),
                "uncertainty_strengthened": sum(
                    r["judge"][name]["uncertainty_strengthened"] for r in final
                ),
                "required_facts_covered": sum(
                    c["covered"] for r in final for c in r["judge"][name]["required_fact_checks"]
                ),
                "required_facts_total": sum(
                    len(r["judge"][name]["required_fact_checks"]) for r in final
                ),
                "source_only_repair_calls": sum(r["arms"][name]["repair_calls"] for r in final),
            }
            for name in ["text", "text_graph"]
        },
        "limitations": protocol["limitations"],
    }
    target = out / "source-adjudicated"
    target.mkdir(exist_ok=True)
    for r in final:
        save(target / (r["id"] + ".json"), r)
    save(out / "source-adjudicated-summary.json", result)
    print({k: v["source_cluster"] for k, v in result["subsets"].items()})


if __name__ == "__main__":
    main()
