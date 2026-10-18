"""Seen-v4 development: language alignment and a bounded weighted-RRF grid, no APIs."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path

from benchmark_text_retrieval_focus import range_scores

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.ports import VectorChunk
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
from aerodiagnosis.retrieval.reranker import E5SemanticReranker, IdentityReranker


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    root = Path.cwd()
    out = args.output.resolve()
    if out.exists():
        raise ValueError("Fresh development output required")
    ds = root / "evaluation/text_hybrid_holdout_v4_candidate"
    source_index = root / ".runtime/holdout-v4/text-index"
    frozen = read(ds / "freeze.json")
    for name, digest in frozen["file_sha256"].items():
        assert hashlib.sha256((ds / name).read_bytes()).hexdigest() == digest
    assert (
        hashlib.sha256((source_index / "knowledge.sqlite").read_bytes()).hexdigest()
        == frozen["index_sha256"]
    )
    out.mkdir(parents=True)
    original_path = out / "original.sqlite"
    aligned_path = out / "chinese.sqlite"
    shutil.copyfile(source_index / "knowledge.sqlite", original_path)
    shutil.copyfile(source_index / "knowledge.sqlite", aligned_path)
    catalog = read(source_index / "catalog.json")
    active = frozenset(catalog["active_versions"])
    provider = LocalE5EmbeddingProvider(root / ".runtime/models/multilingual-e5-small")
    original = SQLiteVectorStore(original_path, embedding_provider=provider)
    aligned = SQLiteChineseVectorStore(aligned_path, embedding_provider=provider)
    lexical = SQLiteLexicalStore(original_path)
    methods = [
        ("bm25", original, "sparse_only", 1.0, False),
        ("dense_original", original, "vector_only", 1.0, False),
        ("old_bm25_e5_rescore", original, "sparse_only", 1.0, True),
        ("dense_zh", aligned, "vector_only", 1.0, False),
    ] + [
        (
            f"zh_rrf_w{weight}" + ("_rescore" if rescore else ""),
            aligned,
            "hybrid_rrf",
            weight,
            rescore,
        )
        for weight in (0.25, 0.5, 1.0, 2.0)
        for rescore in (False, True)
    ]
    policy = dict(
        candidate_k=50,
        rrf_k=60,
        max_per_document=3,
        single_source_unlimited=False,
        expand_context=False,
    )
    retrievers = {
        name: ManualEvidenceRetriever(
            vector_store=store,
            lexical_store=lexical,
            policy=ManualRetrievalPolicy(strategy=strategy, vector_weight=weight, **policy),
            reranker=E5SemanticReranker(provider) if rescore else IdentityReranker(),
        )
        for name, store, strategy, weight, rescore in methods
    }
    protocol = {
        "status": "registered_before_development_predictions",
        "scope": "v4 already seen; development only; no independent-gain claim",
        "api_calls": 0,
        "graph_calls": 0,
        "selection": (
            "max all-reference-range coverage@5; then @10; then MRR@10; "
            "tie lower dense weight and simpler method"
        ),
        "index_change": (
            "encode source-verified Chinese display; keep original content and source coordinates"
        ),
        "method_policies": {n: asdict(r.policy) for n, r in retrievers.items()},
        "model": provider.name,
        "frozen_v4_index_sha256": frozen["index_sha256"],
        "source_code_sha256": {
            p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted((root / "src/aerodiagnosis").rglob("*.py"))
        },
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    save(out / "protocol.json", protocol)
    with sqlite3.connect(original_path) as db:
        placeholders = ",".join("?" for _ in active)
        chunks = [
            VectorChunk(k, d, v, text, json.loads(meta))
            for k, d, v, text, meta in db.execute(
                (
                    "SELECT chunk_id,document_id,version_id,content,metadata_json "
                    "FROM vector_chunks WHERE version_id IN ("
                )
                + placeholders
                + ")",
                sorted(active),
            )
        ]
    assert aligned.upsert(chunks) == len(chunks)
    print("Built Chinese aligned vectors", len(chunks), flush=True)
    sources = {
        "v4-" + d["document_id"]: (ds / d["saved_source_path"]).read_text(encoding="utf-8")
        for d in read(ds / "source_manifest.json")["sources"]
    }
    questions = [
        json.loads(s) for s in (ds / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    rows = []
    with (out / "rankings.jsonl").open("w", encoding="utf-8") as file:
        for i, q in enumerate(questions):
            provider.embed_query(q["query"])
            facts = [{**f, "document_id": "v4-" + q["document_id"]} for f in q["required_facts"]]
            order = methods[i % len(methods) :] + methods[: i % len(methods)]
            for name, _store, _strategy, _weight, _rescore in order:
                start = time.perf_counter()
                hits = []
                for rank, match in enumerate(
                    retrievers[name].search(q["query"], top_k=10, active_version_ids=active), 1
                ):
                    c = match.chunk
                    coords = c.metadata.get("locator", {}).get("coordinates", {})
                    hits.append(
                        dict(
                            rank=rank,
                            chunk_id=c.chunk_id,
                            document_id=c.document_id,
                            start=coords.get("start"),
                            end=coords.get("end"),
                            content=c.content,
                            retrieval=c.metadata["retrieval"],
                        )
                    )
                metrics = {
                    str(k): range_scores(
                        facts, [h for h in hits[:k] if h["document_id"] in sources], sources
                    )
                    for k in (5, 10)
                }
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
                row = dict(
                    sample_id=q["sample_id"],
                    cluster=q["statistical_cluster"],
                    method=name,
                    metrics=metrics,
                    first_full_fact_rr=1 / first if first else 0,
                    seconds=time.perf_counter() - start,
                    hits=hits,
                )
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
                rows.append(row)
            if (i + 1) % 10 == 0:
                print("Development evaluated", i + 1, "/", len(questions), flush=True)
    summary = {}
    for name, *_ in methods:
        group = [r for r in rows if r["method"] == name]
        summary[name] = {
            "questions": len(group),
            "all_ranges_at5": sum(r["metrics"]["5"]["all_reference_ranges_hit"] for r in group),
            "all_ranges_at10": sum(r["metrics"]["10"]["all_reference_ranges_hit"] for r in group),
            "mrr_at10": sum(r["first_full_fact_rr"] for r in group) / len(group),
            "existing_corpus_top5_slots": sum(
                not h["document_id"].startswith("v4-") for r in group for h in r["hits"][:5]
            ),
        }
    assert summary["old_bm25_e5_rescore"]["all_ranges_at5"] == 75
    assert summary["bm25"]["all_ranges_at5"] == 71
    assert summary["dense_original"]["all_ranges_at5"] == 1
    save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
