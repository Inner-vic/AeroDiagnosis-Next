"""Zero-API application regression and seen-development dual/graph ablation."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from aerodiagnosis.adapters.persistence import (
    create_graph_store,
    create_vector_store,
)
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.retrieval.graph_sources import GraphSourceCandidates
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
from aerodiagnosis.retrieval.reranker import IdentityReranker


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FrozenGraph:
    backend_name = "neo4j_source_snapshot"

    def __init__(self, nodes, edges):
        self.nodes, self.edges, self.calls = nodes, edges, 0

    def snapshot(self, *, active_version_ids=None, limit=2000):
        self.calls += 1
        return (
            [n for n in self.nodes if n.version_id in active_version_ids],
            [e for e in self.edges if e.version_id in active_version_ids],
        )


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--profile", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    root = Path.cwd()
    out = args.output
    if out.exists():
        raise ValueError("Fresh experiment output required")
    out.mkdir(parents=True)
    settings = RuntimeSettings.from_env(args.profile)
    policy = ManualRetrievalPolicy(
        candidate_k=50,
        rrf_k=60,
        vector_weight=0.25,
        sparse_weight=1,
        graph_weight=0,
        max_per_document=3,
        single_source_unlimited=False,
        expand_context=False,
    )
    # A copy avoids writes from local index adapter initialization to any frozen input.
    v5 = root / ".runtime/experiments/text-hybrid-v5-20261015"
    frozen = read(root / "evaluation/text_hybrid_holdout_v5_candidate/freeze.json")
    v5db = root / ".runtime/holdout-v5/text-index/chinese.sqlite"
    assert sha(v5db) == frozen["aligned_index_sha256"]
    target = out / "v5-regression.sqlite"
    shutil.copyfile(v5db, target)
    configured = settings.model_copy(update={"database_path": target})
    vector = create_vector_store(configured)
    lexical = SQLiteLexicalStore(target)
    retriever = ManualEvidenceRetriever(
        vector_store=vector, lexical_store=lexical, policy=policy, reranker=IdentityReranker()
    )
    questions = [
        json.loads(x)
        for x in (root / "evaluation/text_hybrid_holdout_v5_candidate/queries.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    old = {
        r["sample_id"]: r
        for r in map(
            json.loads, (v5 / "predictions/rankings.jsonl").read_text(encoding="utf-8").splitlines()
        )
        if r["method"] == "selected_zh_rrf"
    }
    active = frozenset(
        read(root / ".runtime/holdout-v5/text-index/catalog.json")["active_versions"]
    )
    for i, q in enumerate(questions):
        result = retriever.search(q["query"], top_k=10, active_version_ids=active)
        assert [m.chunk.chunk_id for m in result] == [
            h["chunk_id"] for h in old[q["sample_id"]]["hits"]
        ]
        assert all(
            abs(m.score - h["score"]) < 1e-12
            for m, h in zip(result, old[q["sample_id"]]["hits"], strict=True)
        )
        if i % 20 == 19:
            print("Regression", i + 1, "/120", flush=True)
    save(
        out / "v5-regression.json",
        {
            "status": "passed",
            "queries": 120,
            "rankings_identical": 120,
            "compared_to_frozen_v5": True,
            "new_independent_test": False,
            "api_calls": 0,
        },
    )
    # Current application corpus, existing Neo4j assertions, and old Chinese development wording.
    graph = create_graph_store(settings)
    with sqlite3.connect(settings.database_path) as db:
        live_active = frozenset(
            r[0]
            for r in db.execute(
                "SELECT active_version_id FROM documents WHERE active_version_id IS NOT NULL"
            )
        )
        documents = {
            d
            for (d,) in db.execute(
                "SELECT document_id FROM documents WHERE active_version_id IS NOT NULL"
            )
        }
    nodes, edges = graph.snapshot(active_version_ids=live_active, limit=2000)
    assert len(nodes) < 2000
    save(
        out / "graph-snapshot.json",
        {
            "nodes": [asdict(n) for n in nodes],
            "edges": [asdict(e) for e in edges],
            "active_versions": sorted(live_active),
        },
    )
    snapshot = FrozenGraph(nodes, edges)
    live_lexical = SQLiteLexicalStore(settings.database_path)
    live_vector = create_vector_store(settings)
    graph_candidates = GraphSourceCandidates(snapshot, live_lexical.get_chunks)
    base = ManualEvidenceRetriever(
        vector_store=live_vector,
        lexical_store=live_lexical,
        policy=policy,
        reranker=IdentityReranker(),
        graph_candidates=graph_candidates,
    )
    triple_policy = ManualRetrievalPolicy(**{**asdict(policy), "graph_weight": 0.25})
    triple = ManualEvidenceRetriever(
        vector_store=live_vector,
        lexical_store=live_lexical,
        policy=triple_policy,
        reranker=IdentityReranker(),
        graph_candidates=graph_candidates,
    )
    qpaths = [
        root / "evaluation/gas_path_pilot_v1/queries.jsonl",
        root / "evaluation/gas_path_benchmark_v2_supplement/tasks.jsonl",
    ]
    qs = [
        json.loads(line)
        for path in qpaths
        for line in path.read_text(encoding="utf-8").splitlines()
    ]
    zhpath = root / ".runtime/experiments/zh-graph-audit-20261014/queries-zh.json"
    zh = {r["id"]: r["query_zh"] for r in read(zhpath)}
    catalogpath = root / ".runtime/fusion/source-catalog/catalog.json"
    derivations = {d["document_id"]: d["derivation"] for d in read(catalogpath)["documents"]}
    available = {
        (d["original_source_id"], p)
        for key, d in derivations.items()
        if key in documents
        for p in d["original_physical_pages"]
    }
    if "fusion-faa-ch10-page45" in documents:
        available.add(("faa-amtp-ch10", 45))

    def pages(chunk):
        coords = chunk.metadata.get("locator", {}).get("coordinates", {})
        ps = coords.get("pages", [])
        d = derivations.get(chunk.document_id)
        if d:
            return {
                (d["original_source_id"], d["derived_to_original_page"].get(str(p), p)) for p in ps
            }
        if chunk.document_id == "fusion-faa-ch10-page45":
            return {("faa-amtp-ch10", 45)}
        return {(chunk.document_id, p) for p in ps}

    protocol = {
        "purpose": "seen-development-only, not independent evidence of gain",
        "query_count": len(qs),
        "arms": {"dual": asdict(policy), "triple": asdict(triple_policy)},
        "graph_nodes": len(nodes),
        "graph_edges": len(edges),
        "graph_snapshot_sha256": sha(out / "graph-snapshot.json"),
        "metric": ("all annotated source pages hit@5/@10; coarse source retrieval, "
                   "not answer accuracy or fact coverage"),
        "primary_eligible": "answerable, has gold pages, all pages available before retrieval",
        "query_file_sha256": {str(p.relative_to(root)): sha(p) for p in qpaths},
        "chinese_queries_sha256": sha(zhpath),
        "source_catalog_sha256": sha(catalogpath),
        "runner_sha256": sha(Path(__file__)),
        "weights_fixed_before_this_ablation": True,
        "no_tuning": True,
        "api_calls": 0,
        "graph_default_enabled": False,
    }
    save(out / "graph-development-protocol.json", protocol)
    rows = []
    counts = {}
    route_hits = Counter()
    wins = losses = ties = 0
    with (out / "graph-development-rankings.jsonl").open("w", encoding="utf-8") as handle:
        for q in qs:
            gold = {
                (e["document_id"], e.get("source_locator", {}).get("page"))
                for e in q.get("evidence", [])
                if e.get("source_locator", {}).get("page") is not None
            }
            eligible = q["answerability"] == "answerable" and bool(gold) and gold <= available
            flags = {}
            for name, r in [("dual", base), ("triple", triple)]:
                start = time.perf_counter()
                result = r.search(zh[q["sample_id"]], top_k=10, active_version_ids=live_active)
                elapsed = time.perf_counter() - start
                hits = []
                for rank, m in enumerate(result, 1):
                    detail = m.chunk.metadata["retrieval"]
                    assert m.chunk.version_id in live_active
                    if name == "dual":
                        assert not any(c["route"] == "graph" for c in detail["contributions"])
                    hits.append(
                        {
                            "rank": rank,
                            "chunk_id": m.chunk.chunk_id,
                            "document_id": m.chunk.document_id,
                            "version_id": m.chunk.version_id,
                            "source_pages": sorted(pages(m.chunk)),
                            "score": m.score,
                            "retrieval": detail,
                        }
                    )
                flags[name] = (
                    all(pair in {p for m in result[:5] for p in pages(m.chunk)} for pair in gold)
                    if eligible
                    else None
                )
                if name == "triple":
                    route_hits["queries_with_graph_candidates"] += bool(
                        hits and hits[0]["retrieval"]["route_candidates"].get("graph")
                    )
                    route_hits["graph_contributed_top5_slots"] += sum(
                        any(c["route"] == "graph" for c in h["retrieval"]["contributions"])
                        for h in hits[:5]
                    )
                row = {
                    "sample_id": q["sample_id"],
                    "task_type": q["task_type"],
                    "arm": name,
                    "eligible": eligible,
                    "gold_pages": sorted(gold),
                    "all_pages_at5": flags[name],
                    "all_pages_at10": all(
                        pair in {p for m in result for p in pages(m.chunk)} for pair in gold
                    )
                    if eligible
                    else None,
                    "seconds": elapsed,
                    "hits": hits,
                }
                rows.append(row)
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            if eligible:
                wins += flags["triple"] > flags["dual"]
                losses += flags["triple"] < flags["dual"]
                ties += flags["triple"] == flags["dual"]
        for name in ("dual", "triple"):
            rs = [r for r in rows if r["arm"] == name and r["eligible"]]
            counts[name] = {
                "eligible": len(rs),
                "all_pages_at5": sum(r["all_pages_at5"] for r in rs),
                "all_pages_at10": sum(r["all_pages_at10"] for r in rs),
            }
    # Verify the snapshot-backed adapter agrees with actual read-only Neo4j for six queries.
    actual = GraphSourceCandidates(graph, live_lexical.get_chunks)
    for q in qs[:6]:
        left = actual.search(zh[q["sample_id"]], top_k=50, active_version_ids=live_active)
        right = graph_candidates.search(
            zh[q["sample_id"]], top_k=50, active_version_ids=live_active
        )
        assert [(m.chunk.chunk_id, m.score) for m in left] == [
            (m.chunk.chunk_id, m.score) for m in right
        ]
    result = {
        "status": "completed_no_tuning",
        "independent": False,
        "queries": len(qs),
        "arms": counts,
        "paired_page_wins": wins,
        "paired_page_losses": losses,
        "paired_page_ties": ties,
        "graph_activity": dict(route_hits),
        "direct_neo4j_parity_queries": 6,
        "api_calls": 0,
        "default_graph_enabled": False,
        "limits": [
            "Seen questions; coarse page labels; no expert gold or independent gain claim",
            "Graph candidates passed exact source checks; relation entailment not thereby proved",
        ],
    }
    save(out / "graph-development-summary.json", result)
    print(json.dumps(result, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
