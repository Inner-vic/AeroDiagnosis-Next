"""Zero-API development ablation on already-seen v3 queries, never v4 labels."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Exact legacy Chinese source-prefix syntax.
import argparse
import hashlib
import json
import re
import shutil
import time
from collections import Counter
from dataclasses import asdict
from pathlib import Path

try:
    from .benchmark_text_retrieval_focus import range_scores
except ImportError:
    from benchmark_text_retrieval_focus import range_scores

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
from aerodiagnosis.retrieval.planned_candidates import PlannedCandidateStore
from aerodiagnosis.retrieval.query_planning import build_query_plan, plan_record
from aerodiagnosis.retrieval.reranker import IdentityReranker


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--scope", choices=["global", "known_source"], default="global")
    cli.add_argument("--max-per-document", type=int, choices=range(0, 51), default=3)
    cli.add_argument("--single-source-limit", choices=["adaptive", "legacy"], default="adaptive")
    args = cli.parse_args()
    root = Path.cwd()
    out = args.output.resolve()
    if out.exists():
        raise ValueError("Fresh development output required")
    data = root / "evaluation/gas_path_holdout_v3_candidate"
    freeze = read(data / "freeze.json")
    for name, digest in freeze["file_sha256_bytes"].items():
        if sha(data / name) != digest:
            raise ValueError("Old frozen labels changed")
    model = root / ".runtime/models/multilingual-e5-small"
    for name, digest in freeze["configuration"]["trained_artifact_sha256"].items():
        if sha(model / name) != digest:
            raise ValueError("Model changed")
    experiment = root / ".runtime/experiments/holdout-v3-final-20261014"
    out.mkdir(parents=True)
    db = out / "development.sqlite"
    shutil.copyfile(experiment / "frozen-knowledge.sqlite", db)
    provider = LocalE5EmbeddingProvider(model)
    vector, lexical = SQLiteVectorStore(db, embedding_provider=provider), SQLiteLexicalStore(db)
    snapshot = read(experiment / "snapshot.json")
    versions = {d["document_id"]: d["active_version_id"] for d in snapshot["documents"]}
    all_versions = frozenset(versions.values())
    prepared = root / ".runtime/holdout-v3/prepared-v5"
    sources = {
        d["original_source_id"]: read(prepared / d["parsed_path"])["canonical_content"]
        for d in read(prepared / "catalog.json")["documents"]
    }
    queries = [
        json.loads(line)
        for line in (data / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    methods = [
        ("rrf", False, False),
        ("rrf_terms", True, False),
        ("rrf_split", False, True),
        ("rrf_terms_split", True, True),
    ]
    policy = ManualRetrievalPolicy(
        candidate_k=50,
        rrf_k=60,
        max_per_document=args.max_per_document,
        expand_context=False,
        single_source_unlimited=args.single_source_limit == "adaptive",
    )
    files = [
        Path(__file__),
        root / "src/aerodiagnosis/retrieval/query_planning.py",
        root / "src/aerodiagnosis/retrieval/planned_candidates.py",
        root / "src/aerodiagnosis/retrieval/evidence_budget.py",
        root / "src/aerodiagnosis/retrieval/manual.py",
    ]
    protocol = {
        "registered_before_predictions": True,
        "purpose": "seen-v3-development-only",
        "v4_questions_or_predictions_read": False,
        "api_calls": 0,
        "graph_calls": 0,
        "scope": args.scope,
        "questions": len(queries),
        "source_clusters": len(versions),
        "caveat": (
            "Old source-prefixed questions stripped for global diagnostic may be "
            "underspecified; not independent source-discovery accuracy"
        ),
        "metric": (
            "specified necessary source-range coverage only; "
            "not exhaustive relevance or answer accuracy"
        ),
        "policy": asdict(policy),
        "methods": [{"name": n, "terminology": t, "decomposition": d} for n, t, d in methods],
        "candidate_budget": (
            "50 total requested per route across original and every variant; "
            "repeated recall detected and recorded"
        ),
        "code_sha256": {p.relative_to(root).as_posix(): sha(p) for p in files},
    }
    save(out / "protocol.json", protocol)
    records = []
    with (out / "rankings.jsonl").open("w", encoding="utf-8") as handle:
        for q in queries:
            cleaned = re.fullmatch(r"依据资料「([^」]+)」[：:]\s*(.+)", q["query"])
            if not cleaned or cleaned[1] != q["document_id"]:
                raise ValueError("Unexpected old query")
            query = cleaned[2]
            scope = (
                all_versions
                if args.scope == "global"
                else frozenset([versions["holdout-v3-" + q["document_id"]]])
            )
            for name, terms, split in methods:
                audits = []
                dense = PlannedCandidateStore(
                    vector,
                    terminology=terms,
                    decompose=split,
                    audit_sink=lambda r, sink=audits: sink.append({"route": "vector", **r}),
                )
                sparse = PlannedCandidateStore(
                    lexical,
                    terminology=terms,
                    decompose=split,
                    audit_sink=lambda r, sink=audits: sink.append({"route": "sparse", **r}),
                )
                retriever = ManualEvidenceRetriever(
                    vector_store=dense,
                    lexical_store=sparse,
                    policy=policy,
                    reranker=IdentityReranker(),
                )
                started = time.perf_counter()
                found = retriever.search(query, top_k=10, active_version_ids=scope)
                seconds = time.perf_counter() - started
                requested = {
                    route: sum(
                        a["total_requested_candidates"] for a in audits if a["route"] == route
                    )
                    for route in ["vector", "sparse"]
                }
                hits = []
                for match in found:
                    c = match.chunk
                    coords = c.metadata["locator"]["coordinates"]
                    doc = c.document_id.removeprefix("holdout-v3-")
                    if sources[doc][coords["start"] : coords["end"]] != c.content:
                        raise ValueError("Source range mismatch")
                    hits.append(
                        {
                            "chunk_id": c.chunk_id,
                            "document_id": doc,
                            "start": coords["start"],
                            "end": coords["end"],
                        }
                    )
                facts = [{**f, "document_id": q["document_id"]} for f in q["required_facts"]]
                record = {
                    "sample_id": q["sample_id"],
                    "source_layer": q["source_layer"],
                    "source_cluster": q["statistical_cluster"],
                    "method": name,
                    "plan": plan_record(
                        build_query_plan(query, terminology=terms, decompose=split)
                    ),
                    "requested_candidates": requested,
                    "budget_overrun": any(v > 50 for v in requested.values()),
                    "seconds_descriptive_ordered_cache": seconds,
                    "audits": audits,
                    "hits": hits,
                    **{f"at_{k}": range_scores(facts, hits[:k], sources) for k in [5, 10]},
                }
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                records.append(record)
    summary = {
        "purpose": protocol["purpose"],
        "scope": args.scope,
        "metric": protocol["metric"],
        "api_calls": 0,
        "subsets": {},
    }
    for layer in ["source_text", "summary_auxiliary"]:
        summary["subsets"][layer] = {}
        for name, _terms, _split in methods:
            group = [
                r
                for r in records
                if r["method"] == name
                and (r["source_layer"] == "source_text") == (layer == "source_text")
            ]
            activation = Counter(v["kind"] for r in group for v in r["plan"]["variants"])
            summary["subsets"][layer][name] = {
                "questions": len(group),
                "all_ranges_at_5": sum(r["at_5"]["all_reference_ranges_hit"] for r in group),
                "all_ranges_at_10": sum(r["at_10"]["all_reference_ranges_hit"] for r in group),
                "variant_counts": dict(activation),
                "budget_overruns": sum(r["budget_overrun"] for r in group),
            }
    save(out / "summary.json", summary)
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
