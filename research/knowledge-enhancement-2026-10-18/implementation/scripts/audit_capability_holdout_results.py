"""Independently recompute final policies, canonical coverage and paired point estimates."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from audit_capability_fusion import independent_select
from evaluate_capability_development import source_texts
from prepare_capability_holdout import OUT, ROOT, read, save
from tokenizers import Tokenizer

from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder
from aerodiagnosis.ports import VectorChunk

EXP = Path(".runtime/experiments/capability-holdout-v1-20261015")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    frozen = read(ROOT / "evaluation-freeze.json")
    for p, h in frozen["hashes"].items():
        assert digest(Path(p)) == h, p
    assert digest(OUT / "index.sqlite") == frozen["index_sha256"]
    completion = read(EXP / "completion.json")
    assert digest(EXP / "predictions.jsonl") == completion["predictions_sha256"]
    corpus = read(ROOT / "corpus-final.json")
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    sources = source_texts(
        corpus,
        [
            *read(Path("evaluation/retrieval_capability_v1_candidate/sources.json")),
            *data["documents"],
        ],
    )
    questions = {
        q["sample_id"]: q
        for q in map(
            json.loads, (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
        )
    }
    policies = {
        p["id"]: p
        for p in [
            *read(ROOT / "fusion-selection-freeze.json")["controls"],
            *read(ROOT / "runtime-control-addendum.json")["controls"],
        ]
    }
    counter = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    counter.no_padding()
    counter.no_truncation()
    costs = {
        k: len(counter.encode(c.metadata.get("display_content_zh", c.content)).ids)
        for k, c in chunks.items()
    }
    terms = {
        k: set(HashingEmbedder._tokens(c.metadata.get("display_content_zh", c.content)))
        for k, c in chunks.items()
    }
    rows = [
        json.loads(s) for s in (EXP / "predictions.jsonl").read_text(encoding="utf8").splitlines()
    ]
    assert len(rows) == len(questions) * len(policies)
    seen = set()
    max_candidates = max_tokens = 0
    by_method = defaultdict(list)
    for row in rows:
        identity = (row["method"], row["sample_id"])
        assert identity not in seen
        seen.add(identity)
        q = questions[row["sample_id"]]
        policy = policies[row["method"]]
        trace = row["trace"]
        scores = defaultdict(float)
        ranks = defaultdict(dict)
        denom = sum(policy["weights"].values()) / 61
        assert sum(r["requested"] for r in trace["routes"]) == 150
        for route in trace["routes"]:
            name = route["route"]
            assert route["requested"] == policy["budgets"][name]
            assert route["accepted"] == route["returned"] == len(route["chunk_ids"])
            assert route["returned"] <= route["requested"]
            for rank, k in enumerate(route["chunk_ids"], 1):
                scores[k] += policy["weights"][name] / (60 + rank) / denom
                ranks[k][name] = rank
        pool = [
            (k, scores[k], ranks[k]) for k in sorted(scores, key=lambda k: (-scores[k], k))[:50]
        ]
        assert [k for k, _, _ in pool] == trace["post_fusion_pool"]
        chosen, selection = independent_select(pool, chunks, costs, terms, policy["mode"])
        ids = [k for k, _ in chosen]
        assert ids == [h["chunk_id"] for h in row["hits"]]
        for k, v in selection.items():
            assert trace[k] == v
        for hit, (_, score) in zip(row["hits"], chosen, strict=True):
            assert abs(hit["score"] - score) < 1e-12
        for hit in row["hits"]:
            assert sources[hit["source_key"]][hit["start"] : hit["end"]] == hit["content"]
        units = {u["unit_id"]: u for u in q["support_units"]}
        for u in units.values():
            assert sources[u["source_key"]][u["start"] : u["end"]] == u["quote"]

        def cover(k, units=units, row=row, q=q):
            present = set()
            for uid, u in units.items():
                spans = [
                    (h["start"], h["end"])
                    for h in row["hits"][:k]
                    if h["source_key"] == u["source_key"]
                ]
                if all(
                    any(a <= i < b for a, b in spans)
                    for i in range(u["start"], u["end"])
                    if not u["quote"][i - u["start"]].isspace()
                ):
                    present.add(uid)
            return any(set(s) <= present for s in q["acceptable_support_sets"]), max(
                len(set(s) & present) / len(s) for s in q["acceptable_support_sets"]
            )

        for k in [5, 10]:
            assert cover(k)[0] == row["complete" + str(k)]
        assert abs(cover(5)[1] - row["coverage5"]) < 1e-12
        first = next((i for i in range(1, len(ids) + 1) if cover(i)[0]), None)
        assert abs(row["mrr10"] - (1 / first if first else 0)) < 1e-12
        assert (
            trace["context_tokens"] <= 4096
            and max(Counter(chunks[k].document_id for k in ids).values(), default=0) <= 5
        )
        actual = sum(r["returned"] for r in trace["routes"])
        assert actual == row["actual_candidates"] <= 150
        max_candidates = max(max_candidates, actual)
        max_tokens = max(max_tokens, trace["context_tokens"])
        by_method[row["method"]].append(row)
    summary = read(EXP / "summary.json")
    for method, values in by_method.items():
        assert summary[method]["ALL"]["n"] == len(questions)
        assert summary[method]["ALL"]["complete5"] == sum(r["complete5"] for r in values)
    comparisons = read(EXP / "paired-comparisons.json")
    for name, value in comparisons.items():
        a, b = name.split("-")
        left = {r["sample_id"]: r for r in by_method[a]}
        right = {r["sample_id"]: r for r in by_method[b]}
        difference = sum(
            int(left[k]["complete5"]) - int(right[k]["complete5"]) for k in questions
        ) / len(questions)
        assert abs(difference - value["difference_at5"]) < 1e-12
    save(
        EXP / "audit.json",
        {
            "rows_verified": len(rows),
            "questions": len(questions),
            "methods": len(policies),
            "all_rankings_independently_recomputed": True,
            "all_evidence_metrics_independently_recomputed": True,
            "all_freeze_hashes_valid": True,
            "maximum_actual_candidates": max_candidates,
            "maximum_context_tokens": max_tokens,
            "index_unchanged": True,
            "api_calls": 0,
        },
    )
    print("Independent final audit passed", len(rows), flush=True)


if __name__ == "__main__":
    main()
