# ruff: noqa: E501
"""One-pass frozen strict-gate independent retrieval and paired source-cluster analysis."""

from __future__ import annotations

import argparse
import json
import sqlite3
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from capability_metrics import evidence_metrics
from evaluate_capability_development import FrozenGraph, source_texts
from frozen_candidate_cache import FrozenCandidateCache
from prepare_strict_graph_holdout import OUT, ROOT, digest, read, save, sha

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import VectorChunk
from aerodiagnosis.retrieval.fusion_runtime import ConfiguredFusionRetriever
from aerodiagnosis.retrieval.gated_graph import GatedGraphRetriever, GraphSupplementPolicy

EXP = Path(".runtime/experiments/strict-graph-holdout-v1-20261016")
OLD = Path("evaluation/retrieval_capability_holdout_v1_candidate")
BASE = Path("evaluation/retrieval_capability_v1_candidate")
BASE_INDEX = Path(".runtime/holdout-capability-v1/index.sqlite")
BASE_SHA = "027d7dc8bed484bef8d6a62d5db3ba9cc6b81b0a6f42f42f7b87fe27a2096228"


def check_strategy():
    for path, h in read(ROOT / "strategy-freeze.json")["frozen_source_hashes"].items():
        assert digest(Path(path)) == h, "Frozen strategy source changed: " + path


def build():
    check_strategy()
    data = read(ROOT / "translated-corpus-final.json")
    base = read(OLD / "corpus-final.json")
    target = OUT / "index.sqlite"
    assert not target.exists(), "Index immutable; use completed build"
    assert digest(BASE_INDEX) == BASE_SHA
    with (
        sqlite3.connect(BASE_INDEX.resolve().as_uri() + "?mode=ro", uri=True) as src,
        sqlite3.connect(target) as dst,
    ):
        src.backup(dst)
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    vector = SQLiteChineseVectorStore(target, embedding_provider=provider)
    lexical = SQLiteLexicalStore(target)
    new = [VectorChunk(**c) for c in data["chunks"]]
    vector.upsert(new)
    lexical.upsert(new)
    for c in data["chunks"]:
        c["metadata"]["vector_embedding_input"] = {
            "policy": "chinese_display@1",
            "provider": provider.name,
            "sha256": sha(c["metadata"]["display_content_zh"]),
        }
    save(
        ROOT / "corpus-final.json",
        {
            "active_versions": sorted(
                set(base["active_versions"]) | {d["version_id"] for d in data["documents"]}
            ),
            "chunks": base["chunks"] + data["chunks"],
            "base_index_sha256": BASE_SHA,
            "base_corpus_sha256": digest(OLD / "corpus-final.json"),
            "new_corpus_translation_sha256": digest(ROOT / "translated-corpus-final.json"),
            "index_sha256": digest(target),
        },
    )
    assert digest(BASE_INDEX) == BASE_SHA
    print("Built index", len(base["chunks"]) + len(new), "chunks", flush=True)


def inputs():
    corpus = read(ROOT / "corpus-final.json")
    data = read(ROOT / "translated-corpus-final.json")
    verified = (
        read(BASE / "sources.json")
        + read(OLD / "translated-corpus-final.json")["documents"]
        + data["documents"]
    )
    return corpus, source_texts(corpus, verified)


def freeze():
    check_strategy()
    assert not (EXP / "predictions.jsonl").exists()
    assert not (ROOT / "evaluation-freeze.json").exists()
    _corpus, sources = inputs()
    questions = [
        json.loads(s)
        for s in (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
    ]
    assert len({q["sample_id"] for q in questions}) == len(questions) and questions
    for q in questions:
        evidence_metrics(
            sources=sources,
            support_units=q["support_units"],
            acceptable_support_sets=q["acceptable_support_sets"],
            hits=[],
        )
    files = [p for p in ROOT.glob("*.json") if p.name != "evaluation-freeze.json"] + [
        ROOT / "questions-final.jsonl"
    ]
    files += [
        BASE / "sources.json",
        BASE / "source-graph-final.json",
        OLD / "source-graph-final.json",
        OLD / "translated-corpus-final.json",
        OLD / "corpus-final.json",
    ]
    files += [Path(p) for p in read(ROOT / "source-byte-hash-audit.json")["files"]]
    files += list(Path("src/aerodiagnosis").rglob("*.py"))
    files += list(Path("scripts").glob("*.py"))
    files += [
        Path("scripts") / s
        for s in [
            "prepare_strict_graph_holdout.py",
            "author_strict_graph_holdout.py",
            "adopt_strict_graph_questions.py",
            "adopt_strict_graph_semantics.py",
            "audit_strict_graph_results.py",
            "repair_strict_translation.py",
            "evaluate_strict_graph_holdout.py",
            "capability_metrics.py",
            "frozen_candidate_cache.py",
            "evaluate_capability_development.py",
        ]
    ]
    save(
        ROOT / "evaluation-freeze.json",
        {
            "created_at_utc": datetime.now(UTC).isoformat(),
            "before_any_prediction": True,
            "hashes": {p.as_posix(): digest(p) for p in files},
            "index_sha256": digest(OUT / "index.sqlite"),
            "base_index_sha256": BASE_SHA,
            "questions": len(questions),
            "source_families": len({q["cluster"] for q in questions}),
            "all_gold_source_labels_validated": True,
            "no_holdout_tuning": True,
            "expert_reviewed": False,
            "actual_store_execution": "SQLite Chinese BM25 and multilingual E5; reviewed read-only graph snapshot; Neo4j server latency not measured",
            "compute": "B75/D75 baseline; graph25 additional eligible request, not matched compute. Shared immutable raw rank-prefix cache physically requests150 per text route; no runtime latency claim",
        },
    )
    print("Frozen", len(questions), "questions", flush=True)


def covered_units(q, hits):
    found = set()
    for u in q["support_units"]:
        required = {
            i for i in range(u["start"], u["end"]) if not u["quote"][i - u["start"]].isspace()
        }
        observed = set()
        for h in hits:
            if h["source_key"] == u["source_key"]:
                observed.update(range(max(u["start"], h["start"]), min(u["end"], h["end"])))
        if required <= observed:
            found.add(u["unit_id"])
    return found


def paired(a, b):
    if not a:
        return {
            "n": 0,
            "families": 0,
            "difference": None,
            "cluster_bootstrap_95_interval": None,
            "cluster_signflip_two_sided_p": None,
        }
    families = sorted({r["cluster"] for r in a})
    mb = {r["sample_id"]: r for r in b}
    sums = np.array(
        [
            sum(
                int(r["complete10"]) - int(mb[r["sample_id"]]["complete10"])
                for r in a
                if r["cluster"] == g
            )
            for g in families
        ]
    )
    counts = np.array([sum(r["cluster"] == g for r in a) for g in families])
    rng = np.random.default_rng(20261016)
    draw = rng.integers(0, len(families), size=(20000, len(families)))
    samples = sums[draw].sum(1) / counts[draw].sum(1)
    obs = sums.sum() / counts.sum()
    null = (rng.choice([-1, 1], size=(20000, len(families))) * sums).sum(1) / counts.sum()
    return {
        "n": len(a),
        "families": len(families),
        "difference": float(obs),
        "cluster_bootstrap_95_interval": np.quantile(samples, [0.025, 0.975]).tolist(),
        "cluster_signflip_two_sided_p": (1 + int(np.sum(np.abs(null) >= abs(obs) - 1e-12))) / 20001,
        "improved": sum(r["complete10"] and not mb[r["sample_id"]]["complete10"] for r in a),
        "regressed": sum(mb[r["sample_id"]]["complete10"] and not r["complete10"] for r in a),
    }


def analyze(rows, questions):
    methods = defaultdict(list)
    for r in rows:
        methods[r["method"]].append(r)
    baseline = {r["sample_id"]: r for r in methods["text_baseline"]}
    summary = {}
    for method, rs in methods.items():
        summary[method] = {}
        for kind in ["ALL", *sorted({r["kind"] for r in rs}), "STRICT_MULTIHOP"]:
            group = [
                r
                for r in rs
                if kind == "ALL"
                or r["kind"] == kind
                or (kind == "STRICT_MULTIHOP" and r["strict_directed_chain"])
            ]
            rejects = Counter()
            for r in group:
                rejects.update(r["trace"].get("gate_rejections", {}))
            summary[method][kind] = {
                "n": len(group),
                "complete5": sum(r["complete5"] for r in group),
                "complete10": sum(r["complete10"] for r in group),
                "rate10": sum(r["complete10"] for r in group) / len(group) if group else None,
                "mrr10": float(np.mean([r["mrr10"] for r in group])) if group else None,
                "coverage10": float(np.mean([r["coverage10"] for r in group])) if group else None,
                "improved10": sum(
                    r["complete10"] and not baseline[r["sample_id"]]["complete10"] for r in group
                ),
                "regressed10": sum(
                    baseline[r["sample_id"]]["complete10"] and not r["complete10"] for r in group
                ),
                "queries_with_added": sum(bool(r["trace"].get("added")) for r in group),
                "trace_reasons": dict(Counter(r["trace"].get("reason", "text") for r in group)),
                "graph_backend_exception_fallbacks": sum(
                    r["trace"].get("reason") == "graph_unavailable_text_fallback" for r in group
                ),
                "queries_lost_gold_units": sum(bool(r["lost_gold_units"]) for r in group),
                "queries_with_novel_gold_units": sum(bool(r["new_gold_units"]) for r in group),
                "gate_rejections": dict(rejects),
            }
    save(EXP / "summary.json", summary)
    comparisons = {}
    for name, rs in methods.items():
        if name == "text_baseline":
            continue
        comparisons[name] = {}
        for stratum in ["ALL", "STRICT_MULTIHOP"]:
            a = [r for r in rs if stratum == "ALL" or r["strict_directed_chain"]]
            comparisons[name][stratum] = paired(a, [baseline[r["sample_id"]] for r in a])
    primary = [comparisons["strict_gate"][k] for k in ["ALL", "STRICT_MULTIHOP"]]
    available = [v for v in primary if v["cluster_signflip_two_sided_p"] is not None]
    adjusted = 0.0
    for i, v in enumerate(sorted(available, key=lambda v: v["cluster_signflip_two_sided_p"])):
        adjusted = max(adjusted, min(1.0, (2 - i) * v["cluster_signflip_two_sided_p"]))
        v["holm_adjusted_p"] = adjusted
    save(EXP / "paired-comparisons.json", comparisons)
    strict = methods["strict_gate"]
    nstrict = sum(q["strict_directed_chain"] for q in questions)
    save(
        EXP / "safety-audit.json",
        {
            "questions": len(questions),
            "strict_multihop_questions": nstrict,
            "top5_identity_order_and_score_parity": all(r["prefix5_identical"] for r in strict),
            "baseline_complete10_regressions": [
                r["sample_id"]
                for r in strict
                if baseline[r["sample_id"]]["complete10"] and not r["complete10"]
            ],
            "lost_reference_units": [
                {
                    "sample_id": r["sample_id"],
                    "lost_units": r["lost_gold_units"],
                    "baseline_complete10": baseline[r["sample_id"]]["complete10"],
                }
                for r in strict
                if r["lost_gold_units"]
            ],
            "displaced_text_tail_queries": sum(
                bool(r["trace"].get("displaced_text_tail")) for r in strict
            ),
            "nonexpert_review_only": True,
            "production_promoted": False,
            "strict_stratum_exploratory": nstrict < 20,
            "no_general_zero_regression_guarantee": True,
        },
    )
    print(json.dumps({k: v["ALL"] for k, v in summary.items()}, ensure_ascii=False), flush=True)


def evaluate():
    check_strategy()
    f = read(ROOT / "evaluation-freeze.json")
    for p, h in f["hashes"].items():
        assert digest(Path(p)) == h, p
    assert digest(OUT / "index.sqlite") == f["index_sha256"]
    assert digest(BASE_INDEX) == BASE_SHA
    assert not (EXP / "predictions.jsonl").exists()
    EXP.mkdir(parents=True, exist_ok=True)
    corpus, sources = inputs()
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    active = frozenset(corpus["active_versions"])
    graphs = [read(p / "source-graph-final.json") for p in [BASE, OLD, ROOT]]
    graph = FrozenGraph(
        {
            "nodes": [n for g in graphs for n in g["nodes"]],
            "edges": [e for g in graphs for e in g["edges"]],
        }
    )
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    costs = {
        k: len(provider.counter.encode(c.metadata.get("display_content_zh", c.content)).ids)
        for k, c in chunks.items()
    }

    def load(keys):
        return {k: chunks[k] for k in keys if k in chunks}

    stores = {
        k: FrozenCandidateCache(s, k, EXP / "raw-candidate-cache")
        for k, s in {
            "B": SQLiteLexicalStore(OUT / "index.sqlite"),
            "D": SQLiteChineseVectorStore(OUT / "index.sqlite", embedding_provider=provider),
        }.items()
    }
    text = ConfiguredFusionRetriever(
        stores=stores,
        weights={"B": 1, "D": 0.25},
        budgets={"B": 75, "D": 75},
        load_chunks=load,
        token_count=lambda c: costs[c.chunk_id],
        mode="legacy_dedup",
    )
    methods = {
        "text_baseline": text,
        **{
            k: GatedGraphRetriever(
                text=text,
                graph=graph,
                load_chunks=load,
                token_count=lambda c: costs[c.chunk_id],
                policy=p,
            )
            for k, p in {
                "strict_gate": GraphSupplementPolicy(),
                "source_gate": GraphSupplementPolicy(gate="source"),
                "no_eviction_gate": GraphSupplementPolicy(protect=10),
            }.items()
        },
    }
    questions = [
        json.loads(s)
        for s in (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
    ]
    for q in questions:
        evidence_metrics(
            sources=sources,
            support_units=q["support_units"],
            acceptable_support_sets=q["acceptable_support_sets"],
            hits=[],
        )
    rows = []
    with (EXP / "predictions.jsonl").open("x", encoding="utf8") as stream:
        for i, q in enumerate(questions):
            basehits = None
            baseunits = None
            for method, retriever in methods.items():
                started = time.perf_counter()
                matches = retriever.search(q["query"], top_k=10, active_version_ids=active)
                seconds = time.perf_counter() - started
                hits = []
                for h in matches:
                    c = h.chunk
                    coord = c.metadata["locator"]["coordinates"]
                    hits.append(
                        {
                            "chunk_id": c.chunk_id,
                            "source_key": c.document_id + ":" + c.version_id,
                            "start": coord["start"],
                            "end": coord["end"],
                            "content": c.content,
                            "score": h.score,
                            "graph_supplement": c.metadata.get("graph_supplement"),
                        }
                    )
                if method == "text_baseline":
                    basehits = hits
                    baseunits = covered_units(q, hits)
                prefix = [(h["chunk_id"], h["content"], h["score"]) for h in hits[:5]] == [
                    (h["chunk_id"], h["content"], h["score"]) for h in basehits[:5]
                ]
                assert (
                    prefix and len(hits) <= 10 and sum(costs[h["chunk_id"]] for h in hits) <= 4096
                )
                assert (
                    max(
                        Counter(chunks[h["chunk_id"]].document_id for h in hits).values(), default=0
                    )
                    <= 5
                )
                metric = evidence_metrics(
                    sources=sources,
                    support_units=q["support_units"],
                    acceptable_support_sets=q["acceptable_support_sets"],
                    hits=hits,
                )
                units = covered_units(q, hits)
                # Independent interval-union implementation checks canonical scorer results.
                assert (
                    any(set(s) <= units for s in q["acceptable_support_sets"])
                    == metric["retrieval_evidence_metrics"]["10"]["complete_necessary_evidence"]
                )
                assert (
                    any(set(s) <= covered_units(q, hits[:5]) for s in q["acceptable_support_sets"])
                    == metric["retrieval_evidence_metrics"]["5"]["complete_necessary_evidence"]
                )
                r = {
                    "sample_id": q["sample_id"],
                    "cluster": q["cluster"],
                    "kind": q["kind"],
                    "strict_directed_chain": q["strict_directed_chain"],
                    "method": method,
                    "hits": hits,
                    "trace": dict(retriever.last_trace),
                    "seconds_amortized_only": seconds,
                    "complete5": metric["retrieval_evidence_metrics"]["5"][
                        "complete_necessary_evidence"
                    ],
                    "complete10": metric["retrieval_evidence_metrics"]["10"][
                        "complete_necessary_evidence"
                    ],
                    "coverage5": metric["retrieval_evidence_metrics"]["5"][
                        "best_sufficient_set_unit_coverage"
                    ],
                    "coverage10": metric["retrieval_evidence_metrics"]["10"][
                        "best_sufficient_set_unit_coverage"
                    ],
                    "mrr10": metric["complete_support_reciprocal_rank"],
                    "prefix5_identical": prefix,
                    "covered_gold_units": sorted(units),
                    "lost_gold_units": sorted(baseunits - units),
                    "new_gold_units": sorted(units - baseunits),
                }
                rows.append(r)
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
            stream.flush()
            print("Evaluated", i + 1, "/", len(questions), "x4", flush=True)
    analyze(rows, questions)
    save(
        EXP / "completion.json",
        {
            "questions": len(questions),
            "methods": len(methods),
            "rows": len(rows),
            "predictions_sha256": digest(EXP / "predictions.jsonl"),
            "index_unchanged": digest(OUT / "index.sqlite") == f["index_sha256"],
            "base_index_unchanged": digest(BASE_INDEX) == BASE_SHA,
            "all_frozen_inputs_unchanged": all(
                digest(Path(p)) == h for p, h in f["hashes"].items()
            ),
            "api_calls": 0,
            "physical_text_source_calls": {k: v.physical_calls for k, v in stores.items()},
        },
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "freeze", "evaluate"])
    a = p.parse_args()
    {"build": build, "freeze": freeze, "evaluate": evaluate}[a.stage]()
