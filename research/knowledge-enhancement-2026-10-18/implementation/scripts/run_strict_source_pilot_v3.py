"""Isolated source-first pilot; reuse frozen v2 code without changing it."""

from __future__ import annotations

import argparse
import importlib
import json
import shutil
from pathlib import Path

PRIMARY = Path("D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime")
ROOT = Path("evaluation/strict_source_pilot_v3_candidate")
OUT = Path(".runtime/strict-source-pilot-v3")
EXP = Path(".runtime/experiments/strict-source-pilot-v3-20261016")
REVIEW = PRIMARY / "strict-eligibility-v3/pilot-source-review.json"
SEMANTICS = PRIMARY / "strict-eligibility-v3/pilot-semantic-review"
QUESTIONS = PRIMARY / "strict-pilot-v3-blind-review/review.json"


def configure():
    import prepare_graph_supplement_v2 as prep

    prep.ROOT, prep.OUT, prep.REVIEW = ROOT, OUT, REVIEW
    prep.configure()
    return prep


def init():
    prep = configure()
    ROOT.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    assert not (ROOT / "strategy-freeze.json").exists()
    previous = Path("evaluation/graph_supplement_v2_holdout_candidate/strategy-freeze.json")
    strategy = prep.read(previous)
    strategy.update(
        schema="source-first-strict-pilot@3",
        source_first=True,
        interpretation=(
            "Exploratory one-family feasibility pilot, "
            "never confirmatory or general performance proof."
        ),
        source_eligibility_protocol=prep.read(
            PRIMARY / "strict-eligibility-v3/screening-protocol.json"
        ),
        old_strategy_sha256=prep.digest(previous),
        corpus_base=(
            "Previous strict v1 corpus/index, exactly as v2 comparison; "
            "new source only added to this unchanged base. "
            "Prior v2 holdout corpus is not incorporated."
        ),
    )
    strategy["new_sources"].update(candidate_cap=12, selected_cap=1)
    prep.save(ROOT / "strategy-freeze.json", strategy)
    for source, dest in [
        (
            PRIMARY / "strict-eligibility-v3/challenge-review.json",
            ROOT / "source-independent-challenge.json",
        ),
        (
            PRIMARY / "strict-eligibility-v3/screening-protocol.json",
            ROOT / "source-eligibility-protocol.json",
        ),
        (PRIMARY / "strict-eligibility-v3-a/candidates.json", ROOT / "source-screening-a.json"),
        (
            PRIMARY / "strict-eligibility-v3/root-extra-source-review.json",
            ROOT / "source-screening-extra.json",
        ),
        (
            Path(".runtime/graph-supplement-v2-holdout/pricing-check.json"),
            OUT / "pricing-check.json",
        ),
    ]:
        shutil.copyfile(source, dest)
    prep.prepare()


def preflight():
    prep = configure()
    data = prep.read(ROOT / "source-catalog.json")
    batches = []
    for c in data["chunks"]:
        if not batches or sum(len(x["content"]) for x in batches[-1]) + len(c["content"]) > 3200:
            batches.append([])
        batches[-1].append(c)
    # Conservative bytes-as-input-tokens plus framing, same peak rates as shared ledger.
    # No automatic paid format retry in this pilot. Each subsequent dispatch is ledger guarded.
    translation = (
        sum(
            (
                len(
                    json.dumps(
                        {"items": [{"key": str(i), "text": c["content"]} for i, c in enumerate(b)]},
                        ensure_ascii=False,
                    ).encode()
                )
                + 6000
            )
            * 2
            + 4096 * 8
            for b in batches
        )
        / 1e6
    )
    graph = (
        sum(len(c["content"].encode()) for c in data["chunks"]) + 22000
    ) * 2 / 1e6 + 7000 * 8 / 1e6
    author = (
        sum(len(c["content"].encode()) for c in data["chunks"]) + 25000
    ) * 2 / 1e6 + 8192 * 8 / 1e6
    budget = {k: v for k, v in prep.mature.client().summary().items() if k != "receipts"}
    result = dict(
        translation_batches=len(batches),
        source_characters=sum(len(d["content"]) for d in data["documents"]),
        translation_reservation_bound_cny=translation,
        graph_reservation_bound_cny=graph,
        author_reservation_bound_cny=author,
        conservative_sequential_cost_without_repair_cny=translation + graph + author,
        budget=budget,
        automatic_format_retries=False,
        stop_on_error_or_budget_guard=True,
    )
    prep.save(OUT / "budget-preflight.json", result)
    print(json.dumps(result, ensure_ascii=False))


def run(stage):
    prep = configure()
    if stage in {"translate", "extract", "author"}:
        # Preserve shared credential loading, cache and accounting, but disable retry.
        prep.mature.BoundedFormatClient.complete = prep.mature.ResearchClient.complete
    if stage == "init":
        init()
    elif stage == "preflight":
        preflight()
    elif stage == "translate":
        prep.strict.translate(0)
    elif stage == "extract":
        prep.strict.extract(0)
        prep.anchor()
    elif stage == "semantics":
        m = importlib.import_module("adopt_strict_source_pilot_v3_semantics")
        m.ROOT, m.OUT, m.REVIEW = ROOT, OUT, SEMANTICS
        m.adopt()
    elif stage in {"author", "pack"}:
        m = importlib.import_module("author_graph_supplement_v2")
        m.ROOT, m.OUT = ROOT, OUT
        protocol = prep.read(ROOT / "source-only-authoring-protocol.json")
        m.INSTRUCTION += "\n" + protocol["source_scope_instruction_zh"]
        m.run(0) if stage == "author" else m.pack()
    elif stage == "questions":
        m = importlib.import_module("adopt_strict_source_pilot_v3_questions")
        m.ROOT, m.OUT = ROOT, OUT
        m.adopt(True, QUESTIONS)
    elif stage in {"build", "freeze", "evaluate", "audit"}:
        prior = importlib.import_module("evaluate_strict_graph_holdout")
        prior.ROOT = Path("evaluation/strict_graph_holdout_v1_candidate")
        prior.OUT = Path(".runtime/strict-graph-holdout-v1")
        m = importlib.import_module("evaluate_graph_supplement_v2")
        m.ROOT, m.OUT, m.EXP = ROOT, OUT, EXP
        original_comparison = m.comparison

        def bounded_comparison(rows, control):
            if rows and len({r["cluster"] for r in rows}) < 2:
                delta = sum(
                    int(r["complete_all"]) - int(control[r["sample_id"]]["complete_all"])
                    for r in rows
                ) / len(rows)
                return dict(
                    n=len(rows),
                    families=1,
                    difference=delta,
                    p=None,
                    interval=None,
                    wins=sum(
                        r["complete_all"] and not control[r["sample_id"]]["complete_all"]
                        for r in rows
                    ),
                    losses=sum(
                        control[r["sample_id"]]["complete_all"] and not r["complete_all"]
                        for r in rows
                    ),
                    reason="Single source family; p-value and interval unavailable.",
                )
            return original_comparison(rows, control)

        m.comparison = bounded_comparison
        getattr(m, stage)()
    elif stage == "index-audit":
        m = importlib.import_module("audit_graph_supplement_v2_index")
        m.ROOT, m.OUT = ROOT, OUT
        m.main()
    elif stage == "budget":
        print({k: v for k, v in prep.mature.client().summary().items() if k != "receipts"})


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "init",
            "preflight",
            "translate",
            "extract",
            "semantics",
            "author",
            "pack",
            "questions",
            "build",
            "index-audit",
            "freeze",
            "evaluate",
            "audit",
            "budget",
        ],
    )
    run(p.parse_args().stage)
