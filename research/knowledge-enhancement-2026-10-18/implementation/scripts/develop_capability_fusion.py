# ruff: noqa: E501 -- Human-readable preregistration prose.
"""Registered cached replay: all channel pairs, weights, allocations and ranking ablations."""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from tokenizers import Tokenizer

from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.fusion_policy import display_terms, fuse_ranked, select_fused

DATA = Path("evaluation/retrieval_capability_v1_candidate")
BASE = Path(".runtime/experiments/capability-dev60-20261015")
OUT = Path(".runtime/experiments/capability-fusion-dev60-20261015")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def configurations():
    rows = []

    def add(family, routes, weights, budgets):
        for mode in ("rrf", "agreement", "mmr90", "no_source_dedup", "legacy_dedup"):
            policy = dict(
                family=family,
                routes=list(routes),
                weights=dict(zip(routes, weights, strict=True)),
                budgets=dict(zip(routes, budgets, strict=True)),
                mode=mode,
            )
            policy["id"] = (
                family
                + "_"
                + hashlib.sha256(json.dumps(policy, sort_keys=True).encode()).hexdigest()[:12]
            )
            rows.append(policy)

    for route in ("B", "D"):
        add("single_" + route, (route,), (1,), (150,))
    for weight, budget in itertools.product(
        (0.125, 0.25, 0.5, 1), ((75, 75), (100, 50), (50, 100))
    ):
        add("BD", ("B", "D"), (1, weight), budget)
    for graph in ("E", "P"):
        for text in ("B", "D"):
            for weight, budget in itertools.product(
                (0.25, 0.5, 1), ((75, 75), (100, 50), (50, 100))
            ):
                add(text + graph, (text, graph), (1, weight), budget)
        for dense, graph_weight, budget in itertools.product(
            (0.125, 0.25, 0.5),
            (0.25, 0.5, 1),
            ((50, 50, 50), (75, 50, 25), (75, 25, 50), (100, 25, 25)),
        ):
            add("BD" + graph, ("B", "D", graph), (1, dense, graph_weight), budget)
    assert len({r["id"] for r in rows}) == len(rows)
    assert all(sum(r["budgets"].values()) == 150 for r in rows)
    return rows


def candidate_cache(corpus):
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    old = [
        json.loads(s) for s in (BASE / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    cache = defaultdict(dict)
    for row in old:
        route = {
            "bm25": ("B", "bm25"),
            "dense_zh": ("D", "dense"),
            "triple_entity": ("E", "graph"),
            "triple_path": ("P", "graph"),
        }.get(row["method"])
        if route:
            ids = next(r["chunk_ids"] for r in row["retrieval"]["routes"] if r["name"] == route[1])
            if route[0] in {"E", "P"}:
                assert len(ids) < 50, (
                    "Graph cache truncated: retrieve a new source-only cache first"
                )
            cache[row["sample_id"]][route[0]] = ids
    return chunks, dict(cache)


def label_masks(questions, chunks):
    result = {}
    for q in questions:
        units = q["support_units"]
        bits = {u["unit_id"]: 1 << i for i, u in enumerate(units)}
        masks = {}
        for c in chunks.values():
            key = c.document_id + ":" + c.version_id
            coordinate = c.metadata["locator"]["coordinates"]
            mask = 0
            for u in units:
                if (
                    key == u["source_key"]
                    and coordinate["start"] <= u["start"] < u["end"] <= coordinate["end"]
                ):
                    assert (
                        c.content[u["start"] - coordinate["start"] : u["end"] - coordinate["start"]]
                        == u["quote"]
                    )
                    mask |= bits[u["unit_id"]]
            if mask:
                masks[c.chunk_id] = mask
        # No gold unit crosses stable chunks; otherwise use canonical position union scorer.
        for u in units:
            assert sum(bool(mask & bits[u["unit_id"]]) for mask in masks.values()) == 1
        result[q["sample_id"]] = (
            masks,
            [sum(bits[key] for key in s) for s in q["acceptable_support_sets"]],
        )
    return result


def score_hits(ids, labels):
    masks, bundles = labels
    covered = 0
    first = None
    complete5 = False
    coverage5 = 0.0
    for i, key in enumerate(ids, 1):
        covered |= masks.get(key, 0)
        complete = any(covered & b == b for b in bundles)
        if complete and first is None:
            first = i
        if i == 5:
            complete5 = complete
            coverage5 = max((covered & b).bit_count() / b.bit_count() for b in bundles)
    if len(ids) < 5:
        complete5 = any(covered & b == b for b in bundles)
        coverage5 = max((covered & b).bit_count() / b.bit_count() for b in bundles)
    return dict(
        complete5=int(complete5),
        complete10=int(any(covered & b == b for b in bundles)),
        mrr10=1 / first if first else 0.0,
        coverage5=coverage5,
    )


def policy_complexity(p):
    return (
        len(p["routes"]),
        int("P" in p["routes"]),
        {"rrf": 0, "agreement": 1, "mmr90": 2, "no_source_dedup": 4, "legacy_dedup": 3}[p["mode"]],
        p["weights"].get("D", 0),
        sum(p["weights"].get(k, 0) for k in ("E", "P")),
        p["id"],
    )


def choose(policies, results, questions, allowed=None):
    selected_ids = {q["sample_id"] for q in questions}
    baseline = next(
        p
        for p in policies
        if p["family"] == "BD"
        and p["mode"] == "rrf"
        and p["weights"]["D"] == 0.25
        and p["budgets"] == {"B": 75, "D": 75}
    )
    baseline_rows = [r for r in results[baseline["id"]] if r["sample_id"] in selected_ids]

    def subgroup(rows, kinds):
        return sum(r["complete5"] for r in rows if r["kind"].startswith(kinds))

    minimum_simple = subgroup(baseline_rows, ("S1", "S2", "S3")) - 1
    minimum_limits = subgroup(baseline_rows, ("S6",))
    options = []
    unrestricted = []
    for p in policies:
        if allowed is not None and not allowed(p):
            continue
        rows = [r for r in results[p["id"]] if r["sample_id"] in selected_ids]
        key = (
            -sum(r["complete5"] for r in rows),
            -sum(r["complete10"] for r in rows),
            -sum(r["mrr10"] for r in rows),
            -sum(r["coverage5"] for r in rows),
            *policy_complexity(p),
        )
        unrestricted.append((key, p))
        # Guardrails apply to training data only in source-held-out selection.
        if subgroup(rows, ("S1", "S2", "S3")) < minimum_simple:
            continue
        if subgroup(rows, ("S6",)) < minimum_limits:
            continue
        options.append((key, p))
    # A failed family stays that family; never relabel the dual baseline as a single.
    return min(options or unrestricted, key=lambda x: x[0])[1], bool(options)


def aggregate(rows):
    grouped = {}
    for kind in ["ALL", *sorted({r["kind"] for r in rows}), "STRICT_MULTIHOP"]:
        part = [
            r
            for r in rows
            if kind == "ALL"
            or r["kind"] == kind
            or (kind == "STRICT_MULTIHOP" and r["strict_directed_chain"])
        ]
        grouped[kind] = dict(
            n=len(part),
            complete5=sum(r["complete5"] for r in part),
            complete10=sum(r["complete10"] for r in part),
            mrr10=sum(r["mrr10"] for r in part) / len(part),
            coverage5=sum(r["coverage5"] for r in part) / len(part),
        )
    return grouped


def main():
    if OUT.exists():
        raise ValueError("Fresh experiment directory required")
    for name, digest in read(DATA / "development-freeze.json")["file_sha256"].items():
        assert sha(DATA / name) == digest, name
    assert sha(BASE / "predictions.jsonl") == read(BASE / "completion.json")["predictions_sha256"]
    policies = configurations()
    OUT.mkdir(parents=True)
    frozen_files = [
        Path(__file__),
        Path("src/aerodiagnosis/retrieval/fusion_policy.py"),
        Path("src/aerodiagnosis/retrieval/manual.py"),
        DATA / "questions-final.jsonl",
        DATA / "corpus.json",
        BASE / "predictions.jsonl",
    ]
    protocol = dict(
        created_at_utc=datetime.now(UTC).isoformat(),
        status="registered_before_new_grid_scores",
        development_only=True,
        api_calls=0,
        manual_reviewed_graph_upper_bound=True,
        policies=policies,
        candidate_budget=150,
        context_tokens=4096,
        top_k=[5, 10],
        pool_k=50,
        max_per_document=5,
        rrf_k=60,
        cache_replay="Exact ranked prefixes; graph lists proven complete below50. No latency or new raw recall claim.",
        selection="maximize complete5,complete10,completeMRR10,unitcoverage5; tie simpler; train-only simple tasks lose<=1, S6 no loss vs fixed dual; then independent source test",
        source_cross_validation="10-fold leave-one-source-cluster-out: select with9clusters, score held-out6questions; never send kind/gold to ranking",
        file_sha256={p.as_posix(): sha(p) for p in frozen_files},
    )
    save(OUT / "protocol.json", protocol)
    corpus = read(DATA / "corpus.json")
    chunks, cache = candidate_cache(corpus)
    counter = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    counter.no_padding()
    counter.no_truncation()
    costs = {
        k: len(counter.encode(str(c.metadata.get("display_content_zh", c.content))).ids)
        for k, c in chunks.items()
    }
    terms = {key: display_terms(c) for key, c in chunks.items()}
    questions = [
        json.loads(s)
        for s in (DATA / "questions-final.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    labels = label_masks(questions, chunks)
    results = {p["id"]: [] for p in policies}
    # This replayed bitmask scorer is audited against canonical quote-position metrics later.
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for qi, q in enumerate(questions):
            raw = {
                name: [VectorMatch(chunks[key], 0) for key in keys]
                for name, keys in cache[q["sample_id"]].items()
            }
            fuse_cache = {}
            for p in policies:
                signature = json.dumps([p["routes"], p["weights"], p["budgets"]], sort_keys=True)
                if signature not in fuse_cache:
                    routes = {name: raw[name][: p["budgets"][name]] for name in p["routes"]}
                    fuse_cache[signature] = fuse_ranked(routes, p["weights"])
                fused = fuse_cache[signature]
                final, trace = select_fused(
                    fused, token_count=lambda c: costs[c.chunk_id], mode=p["mode"], terms=terms
                )
                ids = [c.chunk.chunk_id for c, _ in final]
                # Gold is consulted only AFTER candidates and final ranking are fixed.
                scores = score_hits(ids, labels[q["sample_id"]])
                row = dict(
                    sample_id=q["sample_id"],
                    cluster=q["cluster"],
                    kind=q["kind"],
                    strict_directed_chain=q["strict_directed_chain"],
                    policy_id=p["id"],
                    **scores,
                    trace=trace,
                    actual_candidates=sum(
                        min(p["budgets"][name], len(raw[name])) for name in p["routes"]
                    ),
                    hits=[
                        dict(
                            chunk_id=c.chunk.chunk_id,
                            rrf=c.score,
                            selection_score=score,
                            route_ranks=dict(c.route_ranks),
                        )
                        for c, score in final
                    ],
                )
                results[p["id"]].append(row)
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
            stream.flush()
            print(f"scored {qi + 1}/60 questions x {len(policies)} policies", flush=True)
    save(
        OUT / "all-policy-summary.json",
        [dict(policy=p, metrics=aggregate(results[p["id"]])) for p in policies],
    )
    families = {
        family: (lambda p, family=family: p["family"] == family)
        for family in ("BD", "BE", "DE", "BP", "DP", "BDE", "BDP")
    }
    families.update(
        overall=lambda p: True,
        single=lambda p: len(p["routes"]) == 1,
        graph_pair=lambda p: len(p["routes"]) == 2 and any(g in p["routes"] for g in ("E", "P")),
        triple=lambda p: len(p["routes"]) == 3,
    )
    selected = {}
    for family, condition in families.items():
        policy, passed = choose(policies, results, questions, condition)
        selected[family] = dict(
            policy=policy, guardrails_passed=passed, metrics=aggregate(results[policy["id"]])
        )
    cv = {name: [] for name in ("overall", "single", "BD", "graph_pair", "triple")}
    folds = []
    for held in sorted({q["cluster"] for q in questions}):
        train = [q for q in questions if q["cluster"] != held]
        fold = dict(held_out_cluster=held, train_clusters=9, test_questions=6, selections={})
        for family in cv:
            policy, passed = choose(policies, results, train, families[family])
            fold["selections"][family] = dict(policy_id=policy["id"], guardrails_passed=passed)
            cv[family].extend(r for r in results[policy["id"]] if r["cluster"] == held)
        folds.append(fold)
    save(
        OUT / "selection.json",
        dict(
            selected=selected,
            source_held_out_metrics={name: aggregate(rows) for name, rows in cv.items()},
            folds=folds,
            caveat="Shared source-curated graph and10seenclusters; source cross-validation is development robustness, not independent validation",
        ),
    )
    save(OUT / "source-held-out-predictions.json", cv)
    save(
        OUT / "completion.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            policies=len(policies),
            prediction_rows=60 * len(policies),
            api_calls=0,
            predictions_sha256=sha(OUT / "predictions.jsonl"),
            selection_sha256=sha(OUT / "selection.json"),
        ),
    )
    print("selection complete", flush=True)


if __name__ == "__main__":
    main()
