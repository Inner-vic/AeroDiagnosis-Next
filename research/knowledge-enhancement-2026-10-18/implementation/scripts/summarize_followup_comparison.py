"""Predeclared source-cluster paired effect, atomic coverage and uncertainty diagnostics."""

from __future__ import annotations

import json
from pathlib import Path

from summarize_holdout_comparison import paired_summary


def main():
    dataset = Path("evaluation/gas_path_holdout_v2_candidate")
    out = Path(".runtime/experiments/holdout-v2-final-20261014")
    protocol = json.loads((dataset / "protocol.json").read_text(encoding="utf-8"))
    rows = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "graded").glob("*.json"))
    ]
    if len(rows) != protocol["questions"] or len({r["id"] for r in rows}) != len(rows):
        raise ValueError("All paired outcomes required")
    groups = {
        "primary_source_text": [r for r in rows if r["source_layer"] == "source_text"],
        "summary_auxiliary": [r for r in rows if r["source_layer"] != "source_text"],
        "all_descriptive": rows,
    }
    result = {
        "basis": "raw blinded Pro judge and program gates; independent source adjudication pending",
        "subsets": {
            name: {key: paired_summary(sub, key) for key in ["source_cluster", "correlation_group"]}
            for name, sub in groups.items()
        },
        "diagnostics": {
            arm: {
                "gate_failures": sum(not r["arms"][arm]["answer"]["gate_valid"] for r in rows),
                "gate_repair_calls": sum(r["arms"][arm]["repair_calls"] for r in rows),
                "uncertainty_strengthened": sum(
                    r["judge"][arm]["uncertainty_strengthened"] for r in rows
                ),
                "unsupported_claim": sum(r["judge"][arm]["unsupported_claim"] for r in rows),
                "required_facts_covered": sum(
                    c["covered"] for r in rows for c in r["judge"][arm]["required_fact_checks"]
                ),
                "required_facts_total": sum(
                    len(r["judge"][arm]["required_fact_checks"]) for r in rows
                ),
                "exact_quote_failures": sum(
                    not all(r["arms"][arm]["citation_exact_checks"]) for r in rows
                ),
            }
            for arm in ["text", "text_graph"]
        },
        "discordant_ids": [
            r["id"]
            for r in rows
            if r["judge"]["text"]["strict_correct"] != r["judge"]["text_graph"]["strict_correct"]
        ],
        "same_packed_context_pairs": sum(
            r["arms"]["text"]["span_catalog"] == r["arms"]["text_graph"]["span_catalog"]
            for r in rows
        ),
    }
    (out / "statistical-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
