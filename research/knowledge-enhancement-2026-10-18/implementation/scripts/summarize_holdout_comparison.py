"""Paired binary results with whole-source bootstrap and no test-driven tuning."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def paired_summary(
    rows: list[dict[str, Any]], cluster_key: str, *, repeats: int = 10000, seed: int = 20261014
) -> dict[str, Any]:
    if not rows:
        raise ValueError("Empty analysis subset")
    ids = [r["id"] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("Duplicate sample IDs")
    clusters = sorted({r[cluster_key] for r in rows})
    if any(not isinstance(c, str) or not c for c in clusters):
        raise ValueError("Missing cluster identity")
    totals = []
    paired = Counter()
    for group in clusters:
        subset = [r for r in rows if r[cluster_key] == group]
        text = [r["judge"]["text"]["strict_correct"] for r in subset]
        graph = [r["judge"]["text_graph"]["strict_correct"] for r in subset]
        if any(not isinstance(x, bool) for x in text + graph):
            raise ValueError("Binary outcomes must be explicit booleans")
        paired.update(zip(text, graph, strict=True))
        totals.append([len(subset), sum(text), sum(graph)])
    counts = np.array(totals, dtype=float)
    observed = counts.sum(axis=0)
    rng = np.random.default_rng(seed)
    # Resample whole source groups, retaining both arms and all their questions.
    samples = counts[rng.integers(0, len(clusters), size=(repeats, len(clusters)))].sum(axis=1)
    deltas = (samples[:, 2] - samples[:, 1]) / samples[:, 0]
    lo, hi = np.quantile(deltas, [0.025, 0.975])
    return {
        "questions": int(observed[0]),
        "clusters": len(clusters),
        "cluster_key": cluster_key,
        "cluster_sizes": {g: int(c[0]) for g, c in zip(clusters, counts, strict=True)},
        "text_correct": int(observed[1]),
        "graph_correct": int(observed[2]),
        "text_rate": observed[1] / observed[0],
        "graph_rate": observed[2] / observed[0],
        "graph_minus_text": (observed[2] - observed[1]) / observed[0],
        "paired_cluster_bootstrap_95_percentile_interval": [float(lo), float(hi)],
        "equal_source_weight_mean_difference": float(
            np.mean((counts[:, 2] - counts[:, 1]) / counts[:, 0])
        ),
        "graph_only_correct": paired[False, True],
        "text_only_correct": paired[True, False],
        "both_correct": paired[True, True],
        "both_incorrect": paired[False, False],
        "repetitions": repeats,
        "seed": seed,
        "interpretation": (
            "conditional on curated sources; AI-adjudicated source knowledge, "
            "not operational diagnostic accuracy"
        ),
    }


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument(
        "--experiment", type=Path, default=Path(".runtime/experiments/holdout-v1-paired-20261014")
    )
    args = cli.parse_args()
    out = args.experiment
    complete = json.loads((out / "comparison-complete.json").read_text(encoding="utf-8"))
    data = [
        json.loads(p.read_text(encoding="utf-8")) for p in sorted((out / "graded").glob("*.json"))
    ]
    if len(data) != complete["questions"] or len(data) != 120:
        raise ValueError("All 120 paired outcomes required; never drop failures")
    if any(r["snapshot_sha256"] != complete["snapshot_sha256"] for r in data):
        raise ValueError("Mixed snapshots")
    subsets = {
        "primary_source_text": [r for r in data if r["source_layer"] == "source_text"],
        "summary_auxiliary": [r for r in data if r["source_layer"] != "source_text"],
        "all_descriptive": data,
    }
    if len(subsets["primary_source_text"]) != 97 or len(subsets["summary_auxiliary"]) != 23:
        raise ValueError("Predeclared source-layer split changed")
    result = {
        "snapshot_sha256": complete["snapshot_sha256"],
        "basis": "raw blinded AI judge; inspect disagreements separately",
        "analysis_code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "subsets": {
            name: {key: paired_summary(sub, key) for key in ["source_cluster", "correlation_group"]}
            for name, sub in subsets.items()
        },
        "task_types_descriptive": {
            k: paired_summary([r for r in data if r["task_type"] == k], "source_cluster")
            for k in sorted({r["task_type"] for r in data})
        },
        "discordant_ids": [
            r["id"]
            for r in data
            if r["judge"]["text"]["strict_correct"] != r["judge"]["text_graph"]["strict_correct"]
        ],
        "limitations": [
            "No representative source sampling or expert review",
            "Same-provider generation and judging",
            "Single generation per arm",
            "Explicit source scope; does not measure cross-document source discovery",
            (
                "Cluster intervals can be degenerate when no discordant pair is observed; "
                "this does not prove equivalence"
            ),
        ],
    }
    (out / "statistical-summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result["subsets"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
