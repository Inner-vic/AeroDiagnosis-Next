"""Score canonical source evidence; selected spans are not exhaustive passage qrels.

Input: one JSONL row per query with sample_id, abstained and ranked chunks.
Each chunk has document_id, canonical_start/end and canonical_sha256.
Graph predictions use sample_id and label; no answer-string matching is performed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def rows(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def verify_dataset_fingerprints(dataset: Path) -> str:
    """Fail closed on edited labels; this check needs no restricted source full text."""
    release_path = dataset / "release.json"
    release_bytes = release_path.read_bytes()
    release = json.loads(release_bytes)
    if (
        not {"queries.jsonl", "graph_claims.jsonl", "corpus_manifest.json"}
        <= release["files"].keys()
    ):
        raise ValueError("Release is missing core fingerprints")
    for name, expected in release["files"].items():
        if Path(name).name != name:
            raise ValueError("Release paths must be local filenames")
        if hashlib.sha256((dataset / name).read_bytes()).hexdigest() != expected:
            raise ValueError(f"Dataset fingerprint mismatch: {name}")
    return hashlib.sha256(release_bytes).hexdigest()


def covered_fraction(span: dict[str, Any], chunks: list[dict[str, Any]]) -> float:
    start, end = span["canonical_start"], span["canonical_end"]
    intervals = sorted(
        (max(start, c["canonical_start"]), min(end, c["canonical_end"]))
        for c in chunks
        if c["document_id"] == span["document_id"]
        and c["canonical_sha256"] == span["canonical_sha256"]
        and c["canonical_end"] > start
        and c["canonical_start"] < end
    )
    covered, cursor = 0, start
    for left, right in intervals:
        if right > max(left, cursor):
            covered += right - max(left, cursor)
        cursor = max(cursor, right)
    return float(covered / (end - start))


def evidence_metrics(
    query: dict[str, Any], chunks: list[dict[str, Any]], k: int
) -> dict[str, float]:
    if query["answerability"] != "answerable":
        raise ValueError("Insufficient-evidence queries have no recall denominator")
    coverage = [covered_fraction(span, chunks[:k]) for span in query["evidence"]]
    complete = [v >= 1 - 1e-12 for v in coverage]
    any_mode = query["evidence_requirement"] == "any"
    docs = {s["document_id"] for s in query["evidence"]}
    return {
        f"required_evidence_hit_at_{k}": float(any(complete) if any_mode else all(complete)),
        f"evidence_coverage_at_{k}": max(coverage) if any_mode else sum(coverage) / len(coverage),
        f"source_hit_at_{k}": float(any(c["document_id"] in docs for c in chunks[:k])),
    }


def macro_f1(truth: list[str], predictions: list[str], labels: tuple[str, ...]) -> float:
    values = []
    for label in labels:
        tp = sum(a == label and b == label for a, b in zip(truth, predictions, strict=True))
        fp = sum(a != label and b == label for a, b in zip(truth, predictions, strict=True))
        fn = sum(a == label and b != label for a, b in zip(truth, predictions, strict=True))
        values.append(2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
    return sum(values) / len(values)


def evaluate(
    dataset: Path,
    predictions_path: Path,
    split: str,
    *,
    translated: bool = False,
    variant: str = "native",
    graph_predictions: Path | None = None,
    graph_variant: str = "native",
) -> dict[str, Any]:
    release_sha256 = verify_dataset_fingerprints(dataset)
    if translated:
        variant = "translation"
    if variant not in {"native", "translation", "scenario_paraphrase"}:
        raise ValueError("Unknown evaluation variant")

    def variant_type(q: dict[str, Any]) -> str:
        return str(
            q.get("query_variant_type", "translation" if q.get("paired_variant_of") else "native")
        )

    queries = [
        q
        for q in rows(dataset / "queries.jsonl")
        if q["split"] == split and variant_type(q) == variant
    ]
    if not queries:
        raise ValueError("Empty evaluation stratum")
    predictions = rows(predictions_path)
    index = {p["sample_id"]: p for p in predictions}
    if len(index) != len(predictions):
        raise ValueError("Duplicate prediction id")
    if set(index) != {q["sample_id"] for q in queries}:
        raise ValueError("Predictions must exactly match selected query ids")
    source_meta = json.loads((dataset / "corpus_manifest.json").read_text(encoding="utf-8"))
    sources = {s["document_id"]: s for s in source_meta["documents"]}
    scored = []
    truth, decisions = [], []
    for query in queries:
        prediction = index[query["sample_id"]]
        if not isinstance(prediction.get("abstained"), bool):
            raise ValueError("Abstained must be an explicit boolean")
        chunks = prediction.get("chunks", [])
        for c in chunks:
            source = sources.get(c["document_id"])
            if not source or c["canonical_sha256"] != source["canonical_sha256"]:
                raise ValueError("Prediction source fingerprint mismatch")
            if (
                type(c["canonical_start"]) is not int
                or type(c["canonical_end"]) is not int
                or not 0
                <= c["canonical_start"]
                < c["canonical_end"]
                <= source["canonical_character_count"]
            ):
                raise ValueError("Invalid prediction source interval")
        truth.append(query["answerability"] == "insufficient_evidence")
        decisions.append(prediction["abstained"])
        if query["answerability"] == "answerable":
            metric = {
                "sample_id": query["sample_id"],
                "fact_id": query["fact_id"],
                "statistical_cluster": query["statistical_cluster"],
                "task_type": query["task_type"],
                "query_language": query["query_language"],
            }
            for k in (5, 10):
                metric.update(evidence_metrics(query, chunks, k))
            scored.append(metric)
    tp = sum(a and b for a, b in zip(truth, decisions, strict=True))
    fp = sum(not a and b for a, b in zip(truth, decisions, strict=True))
    fn = sum(a and not b for a, b in zip(truth, decisions, strict=True))
    keys = [
        f"{name}_at_{k}"
        for k in (5, 10)
        for name in ("required_evidence_hit", "evidence_coverage", "source_hit")
    ]
    # Multiple native questions of the same fact are one statistical unit.
    facts: dict[str, list[dict[str, Any]]] = {}
    for row in scored:
        facts.setdefault(row["fact_id"], []).append(row)
    means = {
        key: sum(sum(r[key] for r in group) / len(group) for group in facts.values()) / len(facts)
        if facts
        else None
        for key in keys
    }
    result: dict[str, Any] = {
        "split": split,
        "dataset_release_sha256": release_sha256,
        "scorer_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "query_variant": variant,
        "translated_stress_only": variant == "translation",
        "query_count": len(queries),
        "answerable_query_count": len(scored),
        "answerable_fact_count": len(facts),
        "metrics": means,
        "abstention_precision": tp / (tp + fp) if tp + fp else None,
        "abstention_recall": tp / (tp + fn) if tp + fn else None,
        "abstention_confusion": {"tp": tp, "fp": fp, "fn": fn},
        "per_query": scored,
        "task_type_counts": dict(Counter(q["task_type"] for q in queries)),
        "limitations": [
            "Known-evidence coverage; alternate passage judgments are incomplete",
            "No semantic answer-quality or expert diagnostic-accuracy claim",
            "Use paired family-cluster uncertainty before reporting superiority",
        ],
    }
    if graph_predictions:
        if graph_variant not in {"native", "negation_stress", "all"}:
            raise ValueError("Invalid graph variant")
        claims = [
            s
            for s in rows(dataset / "graph_claims.jsonl")
            if s["split"] == split
            and (graph_variant == "all" or s.get("graph_variant_type", "native") == graph_variant)
        ]
        if not claims:
            raise ValueError("Empty graph evaluation stratum")
        predicted = rows(graph_predictions)
        claim_index = {p["sample_id"]: p["label"] for p in predicted}
        if len(claim_index) != len(predicted) or set(claim_index) != {
            c["sample_id"] for c in claims
        }:
            raise ValueError("Graph predictions must exactly match claim ids")
        if any(v not in {"supported", "unsupported"} for v in claim_index.values()):
            raise ValueError("Invalid graph verdict")
        truth_labels = [c["label"] for c in claims]
        predicted_labels = [claim_index[c["sample_id"]] for c in claims]
        result["qualified_claim_macro_f1"] = (
            macro_f1(truth_labels, predicted_labels, ("supported", "unsupported"))
            if len(set(truth_labels)) == 2
            else None
        )
        result["qualified_claim_accuracy"] = sum(
            a == b for a, b in zip(truth_labels, predicted_labels, strict=True)
        ) / len(claims)
        result["graph_variant"] = graph_variant
        result["graph_claim_count"] = len(claims)
        result["graph_claim_group_count"] = len(
            {c.get("claim_group_id", c["sample_id"]) for c in claims}
        )
        result["graph_polarity_label_counts"] = dict(
            Counter(f"{c['polarity']}:{c['label']}" for c in claims)
        )
        result["graph_polarity_label_confusion"] = dict(
            Counter(f"{c['polarity']}:{c['label']}:{claim_index[c['sample_id']]}" for c in claims)
        )
        result["graph_task"] = "qualified claim verification, not exhaustive extraction"
        if len(set(truth_labels)) != 2:
            result["graph_metric_note"] = (
                "Single-class stratum: report accuracy; binary macro F1 is not estimable"
            )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--split", choices=("dev", "test"), required=True)
    parser.add_argument("--translated", action="store_true", help="Alias for --variant translation")
    parser.add_argument(
        "--variant", choices=("native", "translation", "scenario_paraphrase"), default="native"
    )
    parser.add_argument("--graph-predictions", type=Path)
    parser.add_argument(
        "--graph-variant", choices=("native", "negation_stress", "all"), default="native"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(
        args.dataset,
        args.predictions,
        args.split,
        translated=args.translated,
        variant=args.variant,
        graph_predictions=args.graph_predictions,
        graph_variant=args.graph_variant,
    )
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "per_query"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
