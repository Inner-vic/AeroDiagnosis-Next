"""Prepare anonymous source grading and predeclared 24-pair Pro audit."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from evaluate_quality_holdout import (
    JUDGE_INSTRUCTION,
    StageClient,
    judge_facts,
    judge_view,
    read,
    rows,
    save,
)

from aerodiagnosis.config import RuntimeSettings


def prepare():
    root = Path.cwd()
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    out = root / ".runtime/experiments/holdout-v3-final-20261014"
    complete = read(out / "generation-complete.json")
    qs = rows(dataset / "queries.jsonl")
    if complete["questions"] != 120 or len(list((out / "answers").glob("*.json"))) != 120:
        raise ValueError("All paired generations must precede scoring")
    groups, mapping = {}, {}
    prepared = root / ".runtime/holdout-v3/prepared-v5"
    sources = {
        r["original_source_id"]: read(prepared / r["parsed_path"])["canonical_content"]
        for r in read(prepared / "catalog.json")["documents"]
    }
    packet = out / "anonymous-source-review"
    packet.mkdir(exist_ok=True)
    for q in qs:
        a = read(out / "answers" / (q["sample_id"] + ".json"))
        if a["snapshot_sha256"] != complete["snapshot_sha256"]:
            raise ValueError("Mixed prediction snapshots")
        order = ["text", "text_graph"]
        if hashlib.sha256((q["sample_id"] + "quality-v3-anonymous").encode()).digest()[0] % 2:
            order.reverse()
        mapping[q["sample_id"]] = dict(zip(["X", "Y"], order, strict=True))
        record = {
            "sample_id": q["sample_id"],
            "question": q["query"],
            "required_facts": judge_facts(q),
            "question_slots": q["question_slots"],
            **{
                letter: judge_view(a["arms"][name])
                for letter, name in mapping[q["sample_id"]].items()
            },
        }
        groups.setdefault(q["document_id"], []).append(record)
    for doc, items in groups.items():
        save(
            packet / (doc + ".json"),
            {
                "source_id": doc,
                "source_original": sources[doc],
                "questions": items,
                "instructions": JUDGE_INSTRUCTION,
                "expert_reviewed": False,
                "method_labels_hidden": True,
            },
        )
    save(out / "source-review-arm-mapping.json", mapping)
    return out, groups, sources, mapping


def main():
    import argparse

    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--pro-audit", action="store_true")
    args = cli.parse_args()
    out, groups, sources, _mapping = prepare()
    print("Anonymous packets", len(groups), "pairs", sum(map(len, groups.values())), flush=True)
    if not args.pro_audit:
        return
    settings = RuntimeSettings.from_env(Path(".runtime/holdout-v3/profile"))
    client = StageClient(settings, "generation_evaluation", model="deepseek-v4-pro")
    target = out / "pro-audit"
    target.mkdir(exist_ok=True)
    selected = [items[0] for items in groups.values()]
    save(
        target / "predeclared-selection.json",
        {
            "rule": "first question per source",
            "sample_ids": [q["sample_id"] for q in selected],
            "pairs": 24,
            "primary_judge": False,
        },
    )

    def audit(q):
        p = target / (q["sample_id"] + ".json")
        if p.exists():
            return
        doc = next(doc for doc, items in groups.items() if items[0]["sample_id"] == q["sample_id"])
        reply = client.complete(
            task="quality-source-blind-pro-audit-v3",
            instruction=JUDGE_INSTRUCTION,
            payload={**q, "source_original": sources[doc]},
            max_tokens=2200,
        )
        save(
            p,
            {
                "sample_id": q["sample_id"],
                "anonymous_judgment": reply.value,
                "receipt": reply.receipt,
                "primary_judge": False,
            },
        )
        print("Pro audit", q["sample_id"], flush=True)

    with ThreadPoolExecutor(max_workers=3) as pool:
        list(pool.map(audit, selected))
    save(
        target / "complete.json",
        {"pairs": 24, "budget": {k: v for k, v in client.summary().items() if k != "receipts"}},
    )


if __name__ == "__main__":
    main()
