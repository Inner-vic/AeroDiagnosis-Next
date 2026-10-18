"""Frozen global text retrieval comparison on new source-bound Chinese questions.

Reference-range metrics measure annotated evidence retrieval, not answer accuracy or
exhaustive semantic relevance. Unjudged alternatives are not negative labels.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import shutil
import sqlite3
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def check_code(root, protocol):
    for name, digest in protocol["file_sha256"].items():
        if sha(root / name) != digest:
            raise ValueError("Frozen implementation changed: " + name)


def paired_cluster_delta(rows, left, right, *, seed=20261015, samples=10000):
    clusters = defaultdict(list)
    for row in rows:
        clusters[row["cluster"]].append(float(row[left]) - float(row[right]))
    groups = sorted(clusters)
    if len(groups) < 2:
        raise ValueError("At least two independent sources required")
    sums = [sum(clusters[g]) for g in groups]
    counts = [len(clusters[g]) for g in groups]
    rng = random.Random(seed)
    draws = []
    for _ in range(samples):
        chosen = [rng.randrange(len(groups)) for _ in groups]
        draws.append(sum(sums[i] for i in chosen) / sum(counts[i] for i in chosen))
    lo, hi = np.percentile(draws, [2.5, 97.5])
    delta = sum(sums) / sum(counts)
    return {
        "delta": delta,
        "source_cluster_95ci": [float(lo), float(hi)],
        "clusters": len(groups),
        "queries": sum(counts),
        "bootstrap_samples": samples,
        "seed": seed,
        "wins": sum(r[left] > r[right] for r in rows),
        "losses": sum(r[left] < r[right] for r in rows),
        "ties": sum(r[left] == r[right] for r in rows),
    }


def validate_labels(dataset):
    label_freeze = dataset / "annotation-freeze.json"
    if label_freeze.exists():
        for name, digest in read(label_freeze)["file_sha256"].items():
            if sha(dataset / name) != digest:
                raise ValueError("Source-only frozen labels changed: " + name)
    manifest = read(dataset / "source_manifest.json")
    sources = {}
    for row in manifest["sources"]:
        p = dataset / row["saved_source_path"]
        if sha(p) != row["saved_source_sha256"]:
            raise ValueError("Source changed")
        sources[row["document_id"]] = p.read_text(encoding="utf-8")
    questions = [
        json.loads(line)
        for line in (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    if len({q["sample_id"] for q in questions}) != len(questions):
        raise ValueError("Repeated sample IDs")
    for q in questions:
        if not q["required_facts"] or not re.search(r"[\u4e00-\u9fff]", q["query"]):
            raise ValueError("Chinese question and necessary facts required")
        if re.search(r"(?i)\ba\d{2}[a-z]\d{4}\b", q["query"]):
            raise ValueError("Source ID leaked in query")
        text = sources[q["document_id"]]
        for f in q["required_facts"]:
            if text[f["source_start"] : f["source_end"]] != f["source_quote"]:
                raise ValueError("Exact source quote mismatch: " + q["sample_id"])
            if hashlib.sha256(text.encode()).hexdigest() != f["source_sha256"]:
                raise ValueError("Label source hash mismatch")
    return manifest, sources, questions


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, required=True)
    cli.add_argument("--index", type=Path, required=True)
    cli.add_argument("--experiment", type=Path, required=True)
    cli.add_argument("--freeze", action="store_true")
    cli.add_argument("--audit-frozen-code-only", action="store_true")
    args = cli.parse_args()
    root, dataset, index, experiment = (
        Path.cwd(),
        args.dataset.resolve(),
        args.index.resolve(),
        args.experiment.resolve(),
    )
    protocol = read(experiment / "protocol.json")
    # Baseline code is immutable even while the live application is repaired.
    code_root = experiment / "frozen-code"
    check_code(code_root, protocol)
    frozen_src = (code_root / "src").resolve()
    loaded = sys.modules.get("aerodiagnosis")
    if loaded is not None and not Path(loaded.__file__).resolve().is_relative_to(frozen_src):
        raise ValueError(
            "Run baseline evaluation in a fresh process, without current application imports"
        )
    sys.path.insert(0, str(frozen_src))
    from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
    from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
    from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
    from aerodiagnosis.retrieval import manual as loaded_manual
    from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
    from aerodiagnosis.retrieval.reranker import E5SemanticReranker, IdentityReranker

    if not Path(loaded_manual.__file__).resolve().is_relative_to(frozen_src):
        raise ValueError("Frozen baseline import resolved outside archive")
    if args.audit_frozen_code_only:
        print(
            json.dumps(
                {
                    "files_verified": len(protocol["file_sha256"]),
                    "loaded_manual": str(Path(loaded_manual.__file__).resolve()),
                    "baseline_policy": asdict(ManualRetrievalPolicy(**protocol["policy"])),
                }
            )
        )
        return
    try:
        from .benchmark_text_retrieval_focus import range_scores
    except ImportError:
        from benchmark_text_retrieval_focus import range_scores
    _manifest, sources, questions = validate_labels(dataset)
    catalog = read(index / "catalog.json")
    if catalog["source_manifest_sha256"] != sha(dataset / "source_manifest.json"):
        raise ValueError("Index source selection differs")
    model = root / ".runtime/models/multilingual-e5-small"
    oldfreeze = read(root / "evaluation/gas_path_holdout_v3_candidate/freeze.json")
    artifacts = oldfreeze["configuration"]["trained_artifact_sha256"]
    for name, digest in artifacts.items():
        if sha(model / name) != digest:
            raise ValueError("Model artifact changed")
    if args.freeze:
        path = dataset / "freeze.json"
        if path.exists():
            raise ValueError("Frozen labels cannot be overwritten")
        hashes = {
            p.relative_to(dataset).as_posix(): sha(p)
            for p in dataset.rglob("*")
            if p.is_file() and p.name != "freeze.json"
        }
        frozen = {
            "created_at": datetime.now(UTC).isoformat(),
            "status": "frozen_before_predictions",
            "file_sha256": hashes,
            "protocol_sha256": sha(experiment / "protocol.json"),
            "index_sha256": sha(index / "knowledge.sqlite"),
            "catalog_sha256": sha(index / "catalog.json"),
            "evaluator_sha256": sha(Path(__file__)),
            "range_metric_sha256": sha(root / "scripts/benchmark_text_retrieval_focus.py"),
            "model_artifacts": artifacts,
            "baseline_source": "protocol-verified frozen-code/src; never the evolving application",
            "sources": len(sources),
            "questions": len(questions),
            "expert_reviewed": False,
        }
        save(path, frozen)
        print(json.dumps({k: frozen[k] for k in ["status", "sources", "questions"]}))
        return
    frozen = read(dataset / "freeze.json")
    for name, digest in frozen["file_sha256"].items():
        if sha(dataset / name) != digest:
            raise ValueError("Frozen input changed")
    for p, key in [
        (index / "knowledge.sqlite", "index_sha256"),
        (index / "catalog.json", "catalog_sha256"),
        (experiment / "protocol.json", "protocol_sha256"),
        (Path(__file__), "evaluator_sha256"),
        (root / "scripts/benchmark_text_retrieval_focus.py", "range_metric_sha256"),
    ]:
        if sha(p) != frozen[key]:
            raise ValueError("Frozen experiment changed")
    out = experiment / "predictions"
    if out.exists():
        raise ValueError("Predictions cannot be overwritten")
    out.mkdir()
    run_database = out / "knowledge.sqlite"
    shutil.copyfile(index / "knowledge.sqlite", run_database)
    provider = LocalE5EmbeddingProvider(model)
    vector, lexical = (
        SQLiteVectorStore(run_database, embedding_provider=provider),
        SQLiteLexicalStore(run_database),
    )
    active = frozenset(catalog["active_versions"])
    methods = [
        ("bm25", "sparse_only", IdentityReranker()),
        ("e5_dense", "vector_only", IdentityReranker()),
        ("old_bm25_e5_rescore", "sparse_only", E5SemanticReranker(provider)),
        ("bm25_e5_rrf", "hybrid_rrf", IdentityReranker()),
        ("bm25_e5_rrf_e5_rescore", "hybrid_rrf", E5SemanticReranker(provider)),
    ]
    retrievers = {
        name: ManualEvidenceRetriever(
            vector_store=vector,
            lexical_store=lexical,
            policy=ManualRetrievalPolicy(strategy=strategy, **protocol["policy"]),
            reranker=reranker,
        )
        for name, strategy, reranker in methods
    }
    # Prewarm passage and query encodings; timings describe explicitly warm local CPU caches.
    started = time.perf_counter()
    with sqlite3.connect(index / "knowledge.sqlite") as db:
        passages = [
            (content, json.loads(metadata))
            for content, metadata in db.execute("SELECT content,metadata_json FROM vector_chunks")
        ]
    for content, metadata in passages:
        provider.embed(str(metadata.get("display_content_zh", content)))
    warmup = time.perf_counter() - started
    source_docs = {"v4-" + key: value for key, value in sources.items()}
    results = []
    with (out / "rankings.jsonl").open("w", encoding="utf-8") as handle:
        for i, q in enumerate(questions):
            provider.embed_query(q["query"])
            order = methods[i % len(methods) :] + methods[: i % len(methods)]
            facts = [{**f, "document_id": "v4-" + q["document_id"]} for f in q["required_facts"]]
            for name, _strategy, _reranker in order:
                start = time.perf_counter()
                matches = retrievers[name].search(q["query"], top_k=10, active_version_ids=active)
                elapsed = time.perf_counter() - start
                hits = []
                for rank, m in enumerate(matches, 1):
                    coord = m.chunk.metadata.get("locator", {}).get("coordinates", {})
                    hits.append(
                        {
                            "rank": rank,
                            "chunk_id": m.chunk.chunk_id,
                            "document_id": m.chunk.document_id,
                            "version_id": m.chunk.version_id,
                            "start": coord.get("start"),
                            "end": coord.get("end"),
                            "score": m.score,
                            "content": m.chunk.content,
                            "retrieval": m.chunk.metadata["retrieval"],
                        }
                    )
                metrics = {}
                for k in [5, 10]:
                    metrics[str(k)] = range_scores(
                        facts, [h for h in hits[:k] if h["document_id"] in source_docs], source_docs
                    )
                first = next(
                    (
                        h["rank"]
                        for h in hits
                        if h["document_id"] == "v4-" + q["document_id"]
                        and any(
                            h["start"] <= f["source_start"] and h["end"] >= f["source_end"]
                            for f in facts
                        )
                    ),
                    None,
                )
                redundant = 0
                for j, h in enumerate(hits[:5]):
                    for prior in hits[:j]:
                        if (h["document_id"], h["version_id"]) != (
                            prior["document_id"],
                            prior["version_id"],
                        ):
                            continue
                        same = h["content"] == prior["content"]
                        contained = (
                            type(h["start"]) is int
                            and type(prior["start"]) is int
                            and prior["start"] <= h["start"] < h["end"] <= prior["end"]
                            and len(prior["content"]) == prior["end"] - prior["start"]
                            and prior["content"][
                                h["start"] - prior["start"] : h["end"] - prior["start"]
                            ]
                            == h["content"]
                        )
                        if same or contained:
                            redundant += 1
                            break
                row = {
                    "sample_id": q["sample_id"],
                    "cluster": q["statistical_cluster"],
                    "method": name,
                    "seconds_warm_cache": elapsed,
                    "redundant_source_slots_at_5": redundant,
                    "metrics": metrics,
                    "first_full_fact_rr": 1 / first if first else 0,
                    "hits": hits,
                }
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                results.append(row)
            print("Evaluated", q["sample_id"], i + 1, "/", len(questions), flush=True)
    summary = {
        "status": "completed_no_test_tuning",
        "sources": len(sources),
        "questions": len(questions),
        "necessary_facts": sum(len(q["required_facts"]) for q in questions),
        "scope": "all active old corpus and every new source; no oracle correct-document filtering",
        "metric_limit": (
            "specified necessary source-range retrieval; "
            "not exhaustive semantic relevance or generated answer accuracy"
        ),
        "expert_reviewed": False,
        "graph_calls": 0,
        "evaluation_api_calls": 0,
        "passage_prewarm_seconds": warmup,
        "catalog": catalog,
        "methods": {},
        "comparisons": {},
    }
    wide = defaultdict(dict)
    for row in results:
        wide[row["sample_id"]].update(
            {
                "cluster": row["cluster"],
                row["method"]: row["metrics"]["5"]["all_reference_ranges_hit"],
            }
        )
    for name, _s, _r in methods:
        rows = [r for r in results if r["method"] == name]
        summary["methods"][name] = {
            "questions": len(rows),
            "redundant_source_slots_at_5": sum(r["redundant_source_slots_at_5"] for r in rows),
            "all_ranges_at_5_count": sum(
                r["metrics"]["5"]["all_reference_ranges_hit"] for r in rows
            ),
            "all_ranges_at_10_count": sum(
                r["metrics"]["10"]["all_reference_ranges_hit"] for r in rows
            ),
            "macro_fact_range_coverage_at_5": sum(
                r["metrics"]["5"]["fact_reference_hit_rate"] for r in rows
            )
            / len(rows),
            "first_full_fact_mrr_at_10": sum(r["first_full_fact_rr"] for r in rows) / len(rows),
            "warm_cache_seconds_median": float(np.median([r["seconds_warm_cache"] for r in rows])),
            "warm_cache_seconds_p95": float(
                np.percentile([r["seconds_warm_cache"] for r in rows], 95)
            ),
            "policy": asdict(retrievers[name].policy),
            "reranker": retrievers[name].reranker_identity,
        }
    for baseline in ["old_bm25_e5_rescore", "bm25", "e5_dense", "bm25_e5_rrf_e5_rescore"]:
        summary["comparisons"]["bm25_e5_rrf-minus-" + baseline] = paired_cluster_delta(
            list(wide.values()), "bm25_e5_rrf", baseline
        )
    save(out / "summary.json", summary)
    print(
        json.dumps(
            {"methods": summary["methods"], "comparisons": summary["comparisons"]},
            ensure_ascii=False,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
