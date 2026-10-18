"""Validate atomic source bindings and stored source coverage, never retrieve questions."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Exact Chinese source prefix.
import json
from pathlib import Path

from aerodiagnosis.domain.atomic_annotation import AtomicAnnotation


def main():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    prepared = root / ".runtime/holdout-v3/prepared-v5"
    catalog = json.loads((prepared / "catalog.json").read_text(encoding="utf-8"))
    sources = {
        row["original_source_id"]: json.loads(
            (prepared / row["parsed_path"]).read_text(encoding="utf-8")
        )
        for row in catalog["documents"]
    }
    qs = [
        json.loads(line)
        for line in (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    results = []
    if len({q["query"] for q in qs}) != len(qs):
        raise ValueError("Duplicate final questions")
    for row in qs:
        if not row["query"].startswith("依据资料「" + row["document_id"] + "」："):
            raise ValueError("Question source scope changed: " + row["sample_id"])
        annotation = AtomicAnnotation.model_validate(
            {k: row[k] for k in ["question_slots", "required_facts", "supplementary_facts"]}
        )
        source = sources[row["document_id"]]
        annotation.verify_source(source["canonical_content"])
        covered = bytearray(len(source["canonical_content"]))
        for chunk in source["chunks"]:
            lo, hi = [chunk["locator"]["coordinates"][k] for k in ["start", "end"]]
            covered[lo:hi] = bytes([1]) * (hi - lo)
        checks = []
        for fact in annotation.required_facts:
            missing = [
                i
                for i in range(fact.source_start, fact.source_end)
                if not source["canonical_content"][i].isspace() and not covered[i]
            ]
            checks.append(
                {
                    "fact_key": fact.fact_key,
                    "exact_source_valid": True,
                    "source_characters_covered": not missing,
                    "uncovered_source_characters": len(missing),
                }
            )
        results.append({"sample_id": row["sample_id"], "checks": checks})
    if len(qs) != 120 or len({q["sample_id"] for q in qs}) != 120:
        raise ValueError("120 unique tasks required")
    output = {
        "status": "source_binding_coverage_only_not_retrieval_recall",
        "questions": len(qs),
        "required_atomic_facts": sum(len(q["required_facts"]) for q in qs),
        "exact_source_bindings_valid": True,
        "results": results,
        "uncovered_facts": [
            c["fact_key"]
            for r in results
            for c in r["checks"]
            if not c["source_characters_covered"]
        ],
    }
    (dataset / "source-binding-validation.json").write_text(
        json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print({k: v for k, v in output.items() if k != "results"})
    if output["uncovered_facts"]:
        raise ValueError("Annotation refers to text outside stored source chunks")


if __name__ == "__main__":
    main()
