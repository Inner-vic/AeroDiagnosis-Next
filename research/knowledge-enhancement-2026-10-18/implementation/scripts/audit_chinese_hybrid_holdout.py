"""Independently recompute saved v5 evidence metrics and index provenance; no retrieval/API."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, required=True)
    cli.add_argument("--index", type=Path, required=True)
    cli.add_argument("--experiment", type=Path, required=True)
    cli.add_argument("--recompute-embeddings", action="store_true")
    args = cli.parse_args()
    ds, index, exp = args.dataset, args.index, args.experiment
    frozen = read(ds / "freeze.json")
    for name, value in frozen["file_sha256"].items():
        assert sha(ds / name) == value, name
    protocol = read(exp / "protocol.json")
    assert sha(exp / "protocol.json") == frozen["protocol_sha256"]
    for name, value in protocol["file_sha256"].items():
        assert sha(exp / "frozen-code" / name) == value, name
    catalog = read(index / "catalog.json")
    assert sha(index / "knowledge.sqlite") == frozen["index_sha256"]
    assert sha(index / "chinese.sqlite") == frozen["aligned_index_sha256"]
    assert sha(index / "catalog.json") == frozen["catalog_sha256"]
    manifest = read(ds / "source_manifest.json")
    sources = {
        "v5-" + r["document_id"]: (ds / r["saved_source_path"]).read_text(encoding="utf-8")
        for r in manifest["sources"]
    }
    questions = {
        q["sample_id"]: q
        for q in map(json.loads, (ds / "queries.jsonl").read_text(encoding="utf-8").splitlines())
    }
    rows = list(
        map(
            json.loads,
            (exp / "predictions/rankings.jsonl").read_text(encoding="utf-8").splitlines(),
        )
    )
    summary = read(exp / "predictions/summary.json")
    methods = set(summary["methods"])
    assert len(rows) == len(questions) * 5 == 600 and len(methods) == 5
    assert len({(r["sample_id"], r["method"]) for r in rows}) == 600
    with sqlite3.connect(index / "knowledge.sqlite") as db:
        chunks = {
            key: (doc, version, text, json.loads(meta), embedding)
            for key, doc, version, text, meta, embedding in db.execute(
                "SELECT chunk_id,document_id,version_id,content,metadata_json,embedding_json "
                "FROM vector_chunks"
            )
        }
    active = set(catalog["active_versions"])
    assert sum(c[1] in active for c in chunks.values()) == catalog["active_index_chunks"]
    method_rows = defaultdict(list)
    wide = defaultdict(dict)
    route_max = defaultdict(int)
    hits_count = 0
    for row in rows:
        q = questions[row["sample_id"]]
        doc = "v5-" + q["document_id"]
        text = sources[doc]
        facts = q["required_facts"]
        hits = row["hits"]
        assert row["cluster"] == q["statistical_cluster"]
        assert len(hits) <= 10 and len({h["chunk_id"] for h in hits}) == len(hits)
        counts = Counter(h["document_id"] for h in hits)
        assert max(counts.values(), default=0) <= 3
        for rank, h in enumerate(hits, 1):
            hits_count += 1
            c = chunks[h["chunk_id"]]
            assert (h["document_id"], h["version_id"], h["content"]) == c[:3]
            assert h["version_id"] in active and h["rank"] == rank
            co = c[3]["locator"]["coordinates"]
            assert (h["start"], h["end"]) == (co["start"], co["end"])
            if h["document_id"] in sources:
                assert sources[h["document_id"]][h["start"] : h["end"]] == h["content"]
            detail = h["retrieval"]
            assert detail["policy"]["candidate_k"] == 50
            assert detail["effective_max_per_document"] == 3
            for route, count in detail["route_candidates"].items():
                route_max[route] = max(route_max[route], count)
            if row["method"] in {"selected_zh_rrf", "zh_rrf_equal"}:
                weight = 0.25 if row["method"] == "selected_zh_rrf" else 1.0
                score = sum(
                    (weight if part["route"] == "vector" else 1.0) / (60 + part["rank"])
                    for part in detail["contributions"]
                ) / ((1 + weight) / 61)
                assert math.isclose(score, h["score"], abs_tol=1e-12)
        for k in (5, 10):
            flags = []
            fractions = []
            for f in facts:
                assert text[f["source_start"] : f["source_end"]] == f["source_quote"]
                positions = [
                    i for i in range(f["source_start"], f["source_end"]) if not text[i].isspace()
                ]
                covered = sum(
                    any(h["document_id"] == doc and h["start"] <= pos < h["end"] for h in hits[:k])
                    for pos in positions
                )
                assert positions
                flags.append(covered == len(positions))
                fractions.append(covered / len(positions))
            metric = row["metrics"][str(k)]
            assert metric["all_reference_ranges_hit"] == all(flags)
            assert math.isclose(metric["fact_reference_hit_rate"], sum(flags) / len(flags))
            assert math.isclose(
                metric["mean_reference_character_coverage"], sum(fractions) / len(fractions)
            )
        first = next(
            (
                h["rank"]
                for h in hits
                if h["document_id"] == doc
                and any(
                    h["start"] <= f["source_start"] and h["end"] >= f["source_end"] for f in facts
                )
            ),
            None,
        )
        assert row["first_full_fact_rr"] == (1 / first if first else 0)
        redundant = 0
        for pos, hit in enumerate(hits[:5]):
            for prior in hits[:pos]:
                if (hit["document_id"], hit["version_id"]) != (
                    prior["document_id"],
                    prior["version_id"],
                ):
                    continue
                inside = prior["start"] <= hit["start"] < hit["end"] <= prior["end"]
                exact_part = (
                    inside
                    and prior["content"][
                        hit["start"] - prior["start"] : hit["end"] - prior["start"]
                    ]
                    == hit["content"]
                )
                if hit["content"] == prior["content"] or exact_part:
                    redundant += 1
                    break
        assert redundant == row["redundant_source_slots_at_5"]
        method_rows[row["method"]].append(row)
        wide[row["sample_id"]][row["method"]] = int(row["metrics"]["5"]["all_reference_ranges_hit"])
    for name, rs in method_rows.items():
        s = summary["methods"][name]
        for k in (5, 10):
            assert s[f"all_ranges_at_{k}_count"] == sum(
                r["metrics"][str(k)]["all_reference_ranges_hit"] for r in rs
            )
        assert math.isclose(
            s["first_full_fact_mrr_at_10"], sum(r["first_full_fact_rr"] for r in rs) / len(rs)
        )
        assert s["redundant_source_slots_at_5"] == sum(r["redundant_source_slots_at_5"] for r in rs)
        assert math.isclose(
            s["macro_fact_range_coverage_at_5"],
            sum(r["metrics"]["5"]["fact_reference_hit_rate"] for r in rs) / len(rs),
        )
    comparisons = {}
    for baseline in methods - {"selected_zh_rrf"}:
        effects = defaultdict(list)
        for key, value in wide.items():
            effects[questions[key]["statistical_cluster"]].append(
                value["selected_zh_rrf"] - value[baseline]
            )
        groups = sorted(effects)
        assert len(groups) == 30 and {len(effects[g]) for g in groups} == {4}
        means = [sum(effects[g]) / 4 for g in groups]
        rng = random.Random(20261015)
        draws = [sum(means[rng.randrange(30)] for _ in range(30)) / 30 for _ in range(10000)]
        ci = np.percentile(draws, [2.5, 97.5]).tolist()
        check = summary["comparisons"]["selected_zh_rrf-minus-" + baseline]
        assert math.isclose(check["delta"], sum(means) / 30, abs_tol=1e-12)
        assert np.allclose(check["source_cluster_95ci"], ci, atol=1e-12)
        deltas = [value["selected_zh_rrf"] - value[baseline] for value in wide.values()]
        assert (check["wins"], check["losses"], check["ties"]) == (
            sum(d > 0 for d in deltas),
            sum(d < 0 for d in deltas),
            sum(d == 0 for d in deltas),
        )
        comparisons[baseline] = {"delta": sum(means) / 30, "ci": ci}
    base = Path(".runtime/holdout-v4/text-index/knowledge.sqlite")
    with sqlite3.connect(base) as db:
        base_rows = list(db.execute("SELECT chunk_id,embedding_json FROM vector_chunks"))
    assert all(chunks[key][4] == embedding for key, embedding in base_rows)
    embedding_audit = {"base_original_rows_unchanged": len(base_rows), "recomputed": False}
    if args.recompute_embeddings:
        sys.path.insert(0, str((exp / "frozen-code/src").resolve()))
        from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider

        provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
        checked_original = checked_aligned = 0
        with sqlite3.connect(index / "chinese.sqlite") as db:
            aligned_rows = list(
                db.execute("SELECT chunk_id,metadata_json,embedding_json FROM vector_chunks")
            )
        for key, meta, embedding in aligned_rows:
            c = chunks[key]
            if c[1] not in active:
                continue
            metadata = json.loads(meta)
            display = metadata.get("display_content_zh", c[2])
            marker = metadata["vector_embedding_input"]
            assert marker["sha256"] == hashlib.sha256(display.encode()).hexdigest()
            assert marker["provider"] == provider.name
            assert np.allclose(provider.embed(display), json.loads(embedding), rtol=0, atol=0)
            checked_aligned += 1
            assert np.allclose(provider.embed(c[2]), json.loads(c[4]), rtol=0, atol=0)
            checked_original += 1
        embedding_audit.update(
            recomputed=True,
            original_rows_exact=checked_original,
            aligned_rows_exact=checked_aligned,
        )
    result = {
        "status": "passed",
        "rankings": len(rows),
        "hits": hits_count,
        "source_clusters": 30,
        "questions": 120,
        "metric_recomputed_without_original_helper": True,
        "weighted_rrf_scores_recomputed": True,
        "reference_ranges_and_active_scope_verified": True,
        "max_actual_route_candidate_count": dict(route_max),
        "comparisons_recomputed": comparisons,
        "embedding_audit": embedding_audit,
        "audit_script_sha256": sha(Path(__file__)),
        "api_calls": 0,
    }
    (exp / "independent-audit.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
