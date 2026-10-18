"""Join independent anonymous source reviews with frozen program gates, without reruns."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from summarize_holdout_comparison import paired_summary


def read(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    root = Path.cwd()
    out = root / ".runtime/experiments/holdout-v2-final-20261014"
    dataset = root / "evaluation/gas_path_holdout_v2_candidate"
    review_path = out / "anonymous-source-adjudication.jsonl"
    reviews = [json.loads(line) for line in review_path.read_text(encoding="utf-8").splitlines()]
    questions = {
        q["sample_id"]: q
        for q in map(
            json.loads, (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    if len(reviews) != 120 or {r["sample_id"] for r in reviews} != set(questions):
        raise ValueError("All 120 unique independent reviews required")
    notes = read(out / "anonymous-source-review-notes.json")
    mapping = read(out / "source-review-arm-mapping.json")
    rows = []
    for review in sorted(reviews, key=lambda r: r["sample_id"]):
        q = questions[review["sample_id"]]
        raw = read(out / "graded" / (review["sample_id"] + ".json"))
        scores = {}
        for letter in ["X", "Y"]:
            name = mapping[review["sample_id"]][letter]
            score = dict(review[letter])
            for key in ["correct", "unsupported_claim", "uncertainty_strengthened"]:
                if type(score.get(key)) is not bool:
                    raise ValueError("Explicit review booleans required")
            checks = score.get("required_fact_checks", [])
            if (
                len(checks) != len(q["required_facts"])
                or {f["fact_key"] for f in checks} != {f["fact_key"] for f in q["required_facts"]}
                or any(type(f.get("covered")) is not bool for f in checks)
            ):
                raise ValueError("Every frozen atomic fact must be reviewed exactly once")
            arm = raw["arms"][name]
            citation_ok = all(arm["citation_exact_checks"]) and bool(
                arm["citation_exact_checks"] or arm["answer"]["insufficient_evidence"]
            )
            score["strict_correct"] = bool(
                score["correct"]
                and all(f["covered"] for f in checks)
                and not score["unsupported_claim"]
                and not score["uncertainty_strengthened"]
                and arm["answer"]["gate_valid"]
                and citation_ok
            )
            scores[name] = score
        rows.append({**raw, "raw_judge": raw["judge"], "judge": scores})
    groups = {
        "primary_source_text": [r for r in rows if r["source_layer"] == "source_text"],
        "summary_auxiliary": [r for r in rows if r["source_layer"] != "source_text"],
        "all_descriptive": rows,
    }
    if len(groups["primary_source_text"]) != 95 or len(groups["summary_auxiliary"]) != 25:
        raise ValueError("Frozen analysis subsets changed")
    result = {
        "basis": "independent anonymous AI source review plus unchanged frozen program gates",
        "human_expert_reviewed": False,
        "independent_review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "review_notes": notes,
        "subsets": {
            k: {unit: paired_summary(v, unit) for unit in ["source_cluster", "correlation_group"]}
            for k, v in groups.items()
        },
        "discordant_ids": [
            r["id"]
            for r in rows
            if r["judge"]["text"]["strict_correct"] != r["judge"]["text_graph"]["strict_correct"]
        ],
        "raw_strict_decisions_changed": {
            name: [
                r["id"]
                for r in rows
                if r["judge"][name]["strict_correct"] != r["raw_judge"][name]["strict_correct"]
            ]
            for name in ["text", "text_graph"]
        },
        "semantic_review": {
            name: {
                "correct_without_program_gate": sum(r["judge"][name]["correct"] for r in rows),
                "unsupported_claim": sum(r["judge"][name]["unsupported_claim"] for r in rows),
                "uncertainty_strengthened": sum(
                    r["judge"][name]["uncertainty_strengthened"] for r in rows
                ),
                "covered_required_facts": sum(
                    f["covered"] for r in rows for f in r["judge"][name]["required_fact_checks"]
                ),
            }
            for name in ["text", "text_graph"]
        },
    }
    target = out / "source-adjudicated"
    target.mkdir(exist_ok=True)
    for row in rows:
        (target / (row["id"] + ".json")).write_text(
            json.dumps(row, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    (out / "source-adjudicated-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                k: {
                    n: v
                    for n, v in vals["source_cluster"].items()
                    if n
                    in [
                        "questions",
                        "text_correct",
                        "graph_correct",
                        "graph_minus_text",
                        "paired_cluster_bootstrap_95_percentile_interval",
                    ]
                }
                for k, vals in result["subsets"].items()
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
