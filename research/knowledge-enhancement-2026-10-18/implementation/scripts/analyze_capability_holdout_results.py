"""Post-test diagnostics; never changes frozen questions, candidates or policies."""

import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path("evaluation/retrieval_capability_holdout_v1_candidate")
EXP = Path(".runtime/experiments/capability-holdout-v1-20261015")


def read(path):
    return json.loads(path.read_text(encoding="utf8"))


def main():
    questions = {
        q["sample_id"]: q
        for q in map(
            json.loads, (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
        )
    }
    rows = defaultdict(dict)
    with (EXP / "predictions.jsonl").open(encoding="utf8") as f:
        for line in f:
            row = json.loads(line)
            rows[row["method"]][row["sample_id"]] = row
    chunks = {c["chunk_id"]: c for c in read(ROOT / "corpus-final.json")["chunks"]}

    def complete(q, ids):
        covered = set()
        for u in q["support_units"]:
            spans = []
            for k in ids:
                c = chunks[k]
                if c["document_id"] + ":" + c["version_id"] == u["source_key"]:
                    coord = c["metadata"]["locator"]["coordinates"]
                    spans.append((coord["start"], coord["end"]))
            if all(
                any(a <= u["start"] + i < b for a, b in spans)
                for i, ch in enumerate(u["quote"])
                if not ch.isspace()
            ):
                covered.add(u["unit_id"])
        return any(set(s) <= covered for s in q["acceptable_support_sets"])

    families = sorted({q["cluster"] for q in questions.values()})
    paired = read(EXP / "paired-comparisons.json")
    rng = np.random.default_rng(20261015)
    independent = {}
    for name, v in paired.items():
        left, right = name.split("-")
        sums = np.array(
            [
                sum(
                    int(rows[left][k]["complete5"]) - int(rows[right][k]["complete5"])
                    for k, q in questions.items()
                    if q["cluster"] == g
                )
                for g in families
            ]
        )
        counts = np.array([sum(q["cluster"] == g for q in questions.values()) for g in families])
        draw = rng.integers(0, len(families), size=(20000, len(families)))
        ci = np.quantile(sums[draw].sum(1) / counts[draw].sum(1), [0.025, 0.975])
        signs = rng.choice([-1, 1], size=(20000, len(families)))
        p = (1 + int(np.sum(np.abs((signs * sums).sum(1)) >= abs(sums.sum()) - 1e-12))) / 20001
        assert np.allclose(ci, v["cluster_bootstrap_95_interval"], rtol=0, atol=1e-14)
        assert p == v["cluster_signflip_two_sided_p"]
        independent[name] = dict(ci=ci.tolist(), p=p)
    ordered = sorted(independent, key=lambda k: independent[k]["p"])
    last = 0
    for i, k in enumerate(ordered):
        last = max(last, min(1, (len(ordered) - i) * independent[k]["p"]))
        assert last == paired[k]["holm_adjusted_p"]
    burden = {
        m: {
            str(n): {
                "n": sum(q["necessary_original_chunks"] == n for q in questions.values()),
                "complete5": sum(
                    r["complete5"]
                    for k, r in rs.items()
                    if questions[k]["necessary_original_chunks"] == n
                ),
            }
            for n in [1, 2, 3]
        }
        for m, rs in rows.items()
    }
    secondary = {}
    for other in ["BD_matched_B100D50", "BD_matched_B125D25", "BDE_matched", "single_B", "old_BD"]:
        a, b = rows["BDP"], rows[other]
        secondary[other] = dict(
            improved=sum(a[k]["complete5"] and not b[k]["complete5"] for k in questions),
            regressed=sum(b[k]["complete5"] and not a[k]["complete5"] for k in questions),
            same_top10=sum(
                [h["chunk_id"] for h in a[k]["hits"]] == [h["chunk_id"] for h in b[k]["hits"]]
                for k in questions
            ),
        )
    candidate_counts = Counter()
    cases = []
    for k, q in questions.items():
        r = rows["BDP"][k]
        routes = {x["route"]: x["chunk_ids"] for x in r["trace"]["routes"]}
        candidate_counts["questions_with_P_candidates"] += bool(routes["P"])
        candidate_counts["complete_support_in_P_candidate_union"] += complete(q, routes["P"])
        candidate_counts["complete_support_in_BD_candidate_union"] += complete(
            q, routes["B"] + routes["D"]
        )
        candidate_counts["complete_support_in_BDP_candidate_union"] += complete(
            q, [cid for ids in routes.values() for cid in ids]
        )
        regression = rows["old_BD"][k]["complete5"] and not r["complete5"]
        if regression:
            candidate_counts["regressed_vs_old_BD"] += 1
            candidate_counts["regression_support_still_in_BD_candidate_union"] += complete(
                q, routes["B"] + routes["D"]
            )
            candidate_counts["regression_support_still_in_BDP_candidate_union"] += complete(
                q, [cid for ids in routes.values() for cid in ids]
            )
            candidate_counts["regression_recovered_at10"] += r["complete10"]
        if regression or (r["complete5"] and not rows["old_BD"][k]["complete5"]):
            cases.append(
                dict(
                    sample_id=k,
                    kind=q["kind"],
                    query=q["query"],
                    old_BD_complete5=rows["old_BD"][k]["complete5"],
                    BDP_complete5=r["complete5"],
                    BDP_complete10=r["complete10"],
                    support_in_BD_candidates=complete(q, routes["B"] + routes["D"]),
                    support_in_P_candidates=complete(q, routes["P"]),
                    top5=[
                        dict(
                            chunk_id=h["chunk_id"],
                            document_id=chunks[h["chunk_id"]]["document_id"],
                            zh=chunks[h["chunk_id"]]["metadata"].get("display_content_zh", "")[
                                :180
                            ],
                            routes=[name for name, ids in routes.items() if h["chunk_id"] in ids],
                        )
                        for h in r["hits"][:5]
                    ],
                )
            )
    result = dict(
        post_test_exploratory=True,
        statistical_recomputation=independent,
        all_primary_intervals_p_and_holm_independently_reproduced=True,
        by_physical_support_chunks=burden,
        matched_secondary=secondary,
        candidate_union_diagnostics=dict(candidate_counts),
        candidate_union_is_not_budgeted_oracle_or_causal_attribution=True,
        case_details=cases,
    )
    (EXP / "post-test-diagnostics.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf8"
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "case_details"}, ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
