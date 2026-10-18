"""Actual-store frozen-policy independent retrieval; no generation or paid API calls."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from capability_metrics import evidence_metrics
from evaluate_capability_development import FrozenGraph, source_texts
from frozen_candidate_cache import FrozenCandidateCache
from prepare_capability_holdout import OUT, ROOT, read, save, sha

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import VectorChunk
from aerodiagnosis.retrieval.fusion_runtime import ConfiguredFusionRetriever
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates

EXP = Path(".runtime/experiments/capability-holdout-v1-20261015")


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def build():
    data = read(ROOT / "translated-corpus-final.json")
    base = read(Path("evaluation/retrieval_capability_v1_candidate/corpus.json"))
    path = OUT / "index.sqlite"
    if path.exists():
        raise ValueError("Completed index is immutable; use saved build")
    original = Path(".runtime/holdout-v5/text-index/chinese.sqlite")
    assert digest(original) == "ac1acbaa7e90302c4a527f9580f939cd6ac06584ae8ba6f4d9001b2ba282724f"
    with (
        sqlite3.connect(original.resolve().as_uri() + "?mode=ro", uri=True) as source,
        sqlite3.connect(path) as dest,
    ):
        source.backup(dest)
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    vector = SQLiteChineseVectorStore(path, embedding_provider=provider)
    lexical = SQLiteLexicalStore(path)
    new = [VectorChunk(**c) for c in data["chunks"]]
    vector.upsert(new)
    lexical.upsert(new)
    # Persist embedding metadata in the canonical corpus too, using the same source input contract.
    for c in data["chunks"]:
        c["metadata"]["vector_embedding_input"] = {
            "policy": "chinese_display@1",
            "provider": provider.name,
            "sha256": sha(c["metadata"]["display_content_zh"]),
        }
    corpus = {
        "active_versions": sorted(
            set(base["active_versions"]) | {d["version_id"] for d in data["documents"]}
        ),
        "chunks": [*base["chunks"], *data["chunks"]],
        "original_base_sha256": digest(original),
        "reviewed_index_sha256": digest(path),
    }
    save(ROOT / "corpus-final.json", corpus)
    print(
        "Index",
        len(corpus["chunks"]),
        "chunks",
        len(corpus["active_versions"]),
        "documents",
        flush=True,
    )


def freeze():
    if (EXP / "predictions.jsonl").exists():
        raise ValueError("Predictions already exist")
    paths = [
        ROOT / f
        for f in [
            "fusion-selection-freeze.json",
            "source-freeze.json",
            "graph-freeze.json",
            "source-graph-final.json",
            "translated-corpus-final.json",
            "corpus-final.json",
            "questions-final.jsonl",
            "runtime-control-addendum.json",
            "source-byte-hash-audit.json",
            "execution-amortization-note.json",
            "annotation-errata.json",
        ]
    ]
    paths.extend(
        Path("evaluation/retrieval_capability_v1_candidate") / f
        for f in ["sources.json", "source-graph-final.json"]
    )
    paths.extend(
        ROOT / f
        for f in [
            "question-final-adoption-audit.json",
            "blind-question-independent-review.json",
            "translation-adoption.json",
        ]
    )
    paths.extend(Path("src/aerodiagnosis").rglob("*.py"))
    paths.extend(
        Path("scripts") / f
        for f in [
            "capability_metrics.py",
            "evaluate_capability_holdout.py",
            "evaluate_capability_development.py",
            "prepare_capability_holdout.py",
            "audit_capability_holdout_results.py",
            "audit_capability_fusion.py",
            "frozen_candidate_cache.py",
        ]
    )
    save(
        ROOT / "evaluation-freeze.json",
        {
            "schema": "capability-independent-final-freeze@1",
            "before_predictions": False,
            "original_data_frozen_before_any_prediction": True,
            "one_empty_unscorable_annotation_excluded_under_documented_erratum": True,
            "before_this_cached_replay": True,
            "dataset_revision": "1.1-empty-gold-erratum",
            "initial_freeze_sha256": digest(EXP / "uncached-attempt/evaluation-freeze.json"),
            "hashes": {p.as_posix(): digest(p) for p in paths},
            "index_sha256": digest(OUT / "index.sqlite"),
            "expert_reviewed": False,
            "questions_source_reviewed": True,
        },
    )


def aggregate(rows):
    return {
        "n": len(rows),
        "complete5": sum(r["complete5"] for r in rows),
        "complete10": sum(r["complete10"] for r in rows),
        "mrr10": float(np.mean([r["mrr10"] for r in rows])),
        "coverage5": float(np.mean([r["coverage5"] for r in rows])),
        "median_seconds": float(np.median([r["seconds"] for r in rows])),
        "mean_actual_candidates": float(np.mean([r["actual_candidates"] for r in rows])),
    }


def evaluate():
    f = read(ROOT / "evaluation-freeze.json")
    for p, h in f["hashes"].items():
        assert digest(Path(p)) == h, p
    assert digest(OUT / "index.sqlite") == f["index_sha256"]
    EXP.mkdir(parents=True, exist_ok=True)
    if (EXP / "predictions.jsonl").exists():
        raise ValueError("Run once after frozen inputs")
    corpus = read(ROOT / "corpus-final.json")
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    active = frozenset(corpus["active_versions"])
    verified = [
        *read(Path("evaluation/retrieval_capability_v1_candidate/sources.json")),
        *data["documents"],
    ]
    sources = source_texts(corpus, verified)
    graph = read(ROOT / "source-graph-final.json")
    old = read(Path("evaluation/retrieval_capability_v1_candidate/source-graph-final.json"))
    merged = FrozenGraph(
        {"nodes": [*old["nodes"], *graph["nodes"]], "edges": [*old["edges"], *graph["edges"]]}
    )
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))

    def load(keys):
        return {k: chunks[k] for k in keys if k in chunks}

    stores = {
        "B": SQLiteLexicalStore(OUT / "index.sqlite"),
        "D": SQLiteChineseVectorStore(OUT / "index.sqlite", embedding_provider=provider),
        "E": GraphPathCandidates(merged, load, max_hops=1),
        "P": GraphPathCandidates(merged, load),
    }
    stores = {
        name: FrozenCandidateCache(store, name, EXP / "raw-candidate-cache")
        for name, store in stores.items()
    }
    costs = {
        k: len(provider.counter.encode(str(c.metadata.get("display_content_zh", c.content))).ids)
        for k, c in chunks.items()
    }
    policies = (
        read(ROOT / "fusion-selection-freeze.json")["controls"]
        + read(ROOT / "runtime-control-addendum.json")["controls"]
    )
    retrievers = {
        p["id"]: ConfiguredFusionRetriever(
            stores={k: stores[k] for k in p["weights"]},
            weights=p["weights"],
            budgets=p["budgets"],
            load_chunks=load,
            token_count=lambda c: costs[c.chunk_id],
            mode=p["mode"],
        )
        for p in policies
    }
    uncached = {}
    with (EXP / "uncached-attempt/predictions.jsonl").open(encoding="utf8") as source:
        for line in source:
            try:
                old = json.loads(line)
            except json.JSONDecodeError:
                break
            uncached[(old["sample_id"], old["method"])] = old
    parity_checked = 0
    questions = [
        json.loads(s)
        for s in (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
        if s.strip()
    ]
    for question in questions:
        # Validate every frozen label before any candidate call, including nonempty quotes.
        evidence_metrics(
            sources=sources,
            support_units=question["support_units"],
            acceptable_support_sets=question["acceptable_support_sets"],
            hits=[],
        )
    results = defaultdict(list)
    with (EXP / "predictions.jsonl").open("x", encoding="utf8") as output:
        for qi, q in enumerate(questions):
            for p in policies:
                retriever = retrievers[p["id"]]
                start = time.perf_counter()
                matches = retriever.search(q["query"], top_k=10, active_version_ids=active)
                seconds = time.perf_counter() - start
                hits = []
                for match in matches:
                    c = match.chunk
                    coord = c.metadata["locator"]["coordinates"]
                    hits.append(
                        {
                            "chunk_id": c.chunk_id,
                            "source_key": c.document_id + ":" + c.version_id,
                            "start": coord["start"],
                            "end": coord["end"],
                            "content": c.content,
                            "score": match.score,
                        }
                    )
                m = evidence_metrics(
                    sources=sources,
                    support_units=q["support_units"],
                    acceptable_support_sets=q["acceptable_support_sets"],
                    hits=hits,
                )
                row = {
                    "sample_id": q["sample_id"],
                    "cluster": q["cluster"],
                    "kind": q["kind"],
                    "necessary_original_chunks": q["necessary_original_chunks"],
                    "strict_directed_chain": q["strict_directed_chain"],
                    "evidence_structure": q["evidence_structure"],
                    "method": p["id"],
                    "hits": hits,
                    "trace": retriever.last_trace,
                    "seconds": seconds,
                    "actual_candidates": retriever.last_trace["actual_candidates"],
                    "complete5": m["retrieval_evidence_metrics"]["5"][
                        "complete_necessary_evidence"
                    ],
                    "complete10": m["retrieval_evidence_metrics"]["10"][
                        "complete_necessary_evidence"
                    ],
                    "mrr10": m["complete_support_reciprocal_rank"],
                    "coverage5": m["retrieval_evidence_metrics"]["5"][
                        "best_sufficient_set_unit_coverage"
                    ],
                }
                original = uncached.get((q["sample_id"], p["id"]))
                if original:
                    assert [h["chunk_id"] for h in row["hits"]] == [
                        h["chunk_id"] for h in original["hits"]
                    ], "Cache changed a frozen ranking"
                    for new, old in zip(row["hits"], original["hits"], strict=True):
                        assert abs(new["score"] - old["score"]) < 1e-12
                    for metric in [
                        "complete5",
                        "complete10",
                        "mrr10",
                        "coverage5",
                        "actual_candidates",
                    ]:
                        assert row[metric] == original[metric]
                    assert row["trace"] == original["trace"]
                    parity_checked += 1
                results[p["id"]].append(row)
                output.write(json.dumps(row, ensure_ascii=False) + "\n")
            output.flush()
            print("Evaluated", qi + 1, "/", len(questions), "x", len(policies), flush=True)
    save(
        EXP / "physical-execution.json",
        {
            name: {
                "physical_source_searches": store.physical_calls,
                "raw_candidates_prefetched": store.loaded_candidates,
            }
            for name, store in stores.items()
        },
    )
    assert parity_checked == len(uncached)
    save(
        EXP / "cache-execution-parity.json",
        {
            "uncached_rows_compared": parity_checked,
            "all_rankings_scores_metrics_and_traces_equal": True,
            "all_retained_question_labels_weights_and_scopes_unchanged": True,
            "one_empty_unscorable_annotation_excluded": True,
            "api_calls": 0,
        },
    )
    summary = {
        name: {
            "ALL": aggregate(rows),
            **{
                kind: aggregate([r for r in rows if r["kind"] == kind])
                for kind in sorted({r["kind"] for r in rows})
            },
        }
        for name, rows in results.items()
    }
    save(EXP / "summary.json", summary)
    contrasts = read(ROOT / "fusion-selection-freeze.json")["primary_contrasts"]
    paired = {}
    rng = np.random.default_rng(20261015)
    for contrast in contrasts:
        left, right = contrast.split("-")
        a = {r["sample_id"]: r for r in results[left]}
        b = {r["sample_id"]: r for r in results[right]}
        families = sorted({r["cluster"] for r in a.values()})
        sums = np.array(
            [
                sum(
                    int(r["complete5"]) - int(b[k]["complete5"])
                    for k, r in a.items()
                    if r["cluster"] == g
                )
                for g in families
            ],
            dtype=float,
        )
        counts = np.array([sum(r["cluster"] == g for r in a.values()) for g in families])
        draw = rng.integers(0, len(families), size=(20000, len(families)))
        samples = sums[draw].sum(1) / counts[draw].sum(1)
        obs = sums.sum() / counts.sum()
        signs = rng.choice([-1, 1], size=(20000, len(families)))
        null = (signs * sums).sum(1) / counts.sum()
        pval = (1 + int(np.sum(np.abs(null) >= abs(obs) - 1e-12))) / 20001
        paired[contrast] = {
            "difference_at5": float(obs),
            "cluster_bootstrap_95_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
            "cluster_signflip_two_sided_p": pval,
            "improved": sum(a[k]["complete5"] and not b[k]["complete5"] for k in a),
            "regressed": sum(b[k]["complete5"] and not a[k]["complete5"] for k in a),
            "families": len(families),
        }
    ordered = sorted(paired, key=lambda k: paired[k]["cluster_signflip_two_sided_p"])
    adjusted = 0.0
    for i, k in enumerate(ordered):
        adjusted = max(
            adjusted, min(1.0, (len(ordered) - i) * paired[k]["cluster_signflip_two_sided_p"])
        )
        paired[k]["holm_adjusted_p"] = adjusted
    save(EXP / "paired-comparisons.json", paired)
    save(
        EXP / "completion.json",
        {
            "questions": len(questions),
            "methods": len(policies),
            "predictions": len(questions) * len(policies),
            "families": len({q["cluster"] for q in questions}),
            "predictions_sha256": digest(EXP / "predictions.jsonl"),
            "api_calls": 0,
            "index_sha256_unchanged": digest(OUT / "index.sqlite") == f["index_sha256"],
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "freeze", "evaluate"])
    a = p.parse_args()
    {"build": build, "freeze": freeze, "evaluate": evaluate}[a.stage]()
