"""Independent numeric, selection and canonical-evidence audit of the fusion grid."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from tokenizers import Tokenizer

from capability_metrics import evidence_metrics
from develop_capability_fusion import BASE, DATA, OUT, candidate_cache, read, save, sha
from evaluate_capability_development import source_texts
from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder


def redundant(a, b, mode):
    if (a.document_id, a.version_id) != (b.document_id, b.version_id):
        return False
    if a.metadata.get("document_metadata") != b.metadata.get("document_metadata"):
        return False
    ac = a.metadata["locator"]["coordinates"]
    bc = b.metadata["locator"]["coordinates"]
    if mode != "legacy_dedup":
        keys = ("qualifiers", "applicability", "role", "engine_model", "polarity", "certainty")
        if any(a.metadata.get(k) != b.metadata.get(k) for k in keys):
            return False
        if ac.get("section_path") != bc.get("section_path"):
            return False
    if a.content == b.content:
        return True
    if a.metadata["locator"].get("kind") != b.metadata["locator"].get("kind") or ac.get(
        "page"
    ) != bc.get("page"):
        return False
    lo, hi = ac["start"], ac["end"]
    start, end = bc["start"], bc["end"]
    return (
        start <= lo < hi <= end
        and len(b.content) == end - start
        and b.content[lo - start : hi - start] == a.content
    )


def independent_select(pool, chunks, costs, terms, mode):
    remaining = list(pool)
    chosen = []
    counts = Counter()
    tokens = duplicates = capped = 0
    while remaining and len(chosen) < 10:
        values = {}
        for key, rrf, ranks in remaining:
            if mode == "agreement":
                score = rrf + 0.04 * max(0, len(ranks) - 1)
            elif mode == "mmr90":
                similarity = max(
                    (
                        len(terms[key] & terms[k]) / max(1, len(terms[key] | terms[k]))
                        for k, _ in chosen
                    ),
                    default=0,
                )
                score = 0.9 * rrf - 0.1 * similarity
            else:
                score = rrf
            values[key] = score
        key, rrf, ranks = min(remaining, key=lambda row: (-values[row[0]], -row[1], row[0]))
        remaining.remove((key, rrf, ranks))
        c = chunks[key]
        if counts[c.document_id] >= 5:
            capped += 1
            continue
        if mode != "no_source_dedup" and any(redundant(c, chunks[old], mode) for old, _ in chosen):
            duplicates += 1
            continue
        if tokens + costs[key] > 4096:
            break
        chosen.append((key, values[key]))
        tokens += costs[key]
        counts[c.document_id] += 1
    return chosen, dict(
        context_tokens=tokens, source_duplicates_skipped=duplicates, document_cap_skipped=capped
    )


def main():
    protocol = read(OUT / "protocol.json")
    completion = read(OUT / "completion.json")
    for name, digest in protocol["file_sha256"].items():
        assert sha(Path(name)) == digest, name
    assert sha(OUT / "predictions.jsonl") == completion["predictions_sha256"]
    assert sha(OUT / "selection.json") == completion["selection_sha256"]
    assert read(OUT / "graph-cache-parity.json")["all_equal"]
    corpus = read(DATA / "corpus.json")
    chunks, raw = candidate_cache(corpus)
    sources = source_texts(corpus, read(DATA / "sources.json"))
    questions = {
        q["sample_id"]: q
        for q in map(
            json.loads, (DATA / "questions-final.jsonl").read_text(encoding="utf-8").splitlines()
        )
    }
    policies = {p["id"]: p for p in protocol["policies"]}
    counter = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    counter.no_padding()
    counter.no_truncation()
    costs = {
        key: len(counter.encode(str(c.metadata.get("display_content_zh", c.content))).ids)
        for key, c in chunks.items()
    }
    terms = {
        key: set(HashingEmbedder._tokens(str(c.metadata.get("display_content_zh", c.content))))
        for key, c in chunks.items()
    }
    metric_cache = {}
    fused_cache = {}
    count = 0
    seen = set()
    max_tokens = 0
    old_parity = 0
    old = {
        r["sample_id"]: r
        for r in map(
            json.loads, (BASE / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if r["method"] == "dual"
    }
    for line in (OUT / "predictions.jsonl").read_text(encoding="utf-8").splitlines():
        row = json.loads(line)
        q = questions[row["sample_id"]]
        p = policies[row["policy_id"]]
        assert (row["sample_id"], p["id"]) not in seen
        seen.add((row["sample_id"], p["id"]))
        assert sum(p["budgets"].values()) == 150
        signature = (row["sample_id"], json.dumps([p["weights"], p["budgets"]], sort_keys=True))
        if signature not in fused_cache:
            scores = defaultdict(float)
            ranks = defaultdict(dict)
            denominator = sum(p["weights"].values()) / 61
            actual = 0
            for name, weight in p["weights"].items():
                ids = raw[q["sample_id"]][name][: p["budgets"][name]]
                actual += len(ids)
                for rank, key in enumerate(ids, 1):
                    scores[key] += weight / (60 + rank) / denominator
                    ranks[key][name] = rank
            pool = [
                (key, scores[key], ranks[key])
                for key in sorted(scores, key=lambda key: (-scores[key], key))[:50]
            ]
            fused_cache[signature] = (pool, actual)
        pool, actual = fused_cache[signature]
        assert actual == row["actual_candidates"] and actual <= 150
        chosen, trace = independent_select(pool, chunks, costs, terms, p["mode"])
        ids = [key for key, _ in chosen]
        assert ids == [h["chunk_id"] for h in row["hits"]]
        assert trace == row["trace"]
        scores = {key: value for key, value, _ in pool}
        for hit, (_, selection_score) in zip(row["hits"], chosen, strict=True):
            assert abs(hit["rrf"] - scores[hit["chunk_id"]]) < 1e-12
            assert abs(hit["selection_score"] - selection_score) < 1e-12
        max_tokens = max(max_tokens, trace["context_tokens"])
        metric_key = (row["sample_id"], tuple(ids))
        if metric_key not in metric_cache:
            hits = []
            for key in ids:
                c = chunks[key]
                coordinates = c.metadata["locator"]["coordinates"]
                hits.append(
                    dict(
                        source_key=c.document_id + ":" + c.version_id,
                        start=coordinates["start"],
                        end=coordinates["end"],
                        content=c.content,
                    )
                )
            metric_cache[metric_key] = evidence_metrics(
                sources=sources,
                support_units=q["support_units"],
                acceptable_support_sets=q["acceptable_support_sets"],
                hits=hits,
            )
        metric = metric_cache[metric_key]
        m = metric["retrieval_evidence_metrics"]
        assert row["complete5"] == m["5"]["complete_necessary_evidence"]
        assert row["complete10"] == m["10"]["complete_necessary_evidence"]
        assert abs(row["coverage5"] - m["5"]["best_sufficient_set_unit_coverage"]) < 1e-12
        assert abs(row["mrr10"] - metric["complete_support_reciprocal_rank"]) < 1e-12
        if (
            p["family"] == "BD"
            and p["mode"] == "legacy_dedup"
            and p["weights"] == {"B": 1, "D": 0.25}
            and p["budgets"] == {"B": 75, "D": 75}
        ):
            assert ids == [h["chunk_id"] for h in old[q["sample_id"]]["retrieval"]["hits"]]
            old_parity += 1
        count += 1
        if count % 6100 == 0:
            print(f"audited {count}/36600", flush=True)
    assert count == 60 * len(policies) == 36600 and old_parity == 60
    cv = read(OUT / "source-held-out-predictions.json")
    selection = read(OUT / "selection.json")
    for family, rows in cv.items():
        assert len(rows) == 60 and len({r["sample_id"] for r in rows}) == 60
        for fold in selection["folds"]:
            matching = [r for r in rows if r["cluster"] == fold["held_out_cluster"]]
            assert len(matching) == 6 and all(
                r["policy_id"] == fold["selections"][family]["policy_id"] for r in matching
            )
    save(
        OUT / "audit.json",
        dict(
            predictions_checked=count,
            policies=len(policies),
            questions=60,
            canonical_metric_rankings=len(metric_cache),
            weighted_rrf_and_final_sort_independently_recomputed=True,
            all_budget_and_source_dedup_checks_passed=True,
            original_default_dual_top10_identical=old_parity,
            max_context_tokens=max_tokens,
            graph_top150_cache_parity=120,
            source_held_out_fold_partitions_checked=True,
            all_frozen_input_hashes_valid=True,
            api_calls=0,
        ),
    )
    print("All36600 fusion rows verified", flush=True)


if __name__ == "__main__":
    main()
