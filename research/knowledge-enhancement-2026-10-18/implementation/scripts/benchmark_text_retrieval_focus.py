"""Offline text-only development diagnostics on previously seen v3 reference ranges.

No API, graph, answer generation, new test claim or exhaustive semantic relevance claim.
"""

from __future__ import annotations

# ruff: noqa: RUF001 -- Exact Chinese source-reference syntax.
import argparse
import hashlib
import json
import re
import shutil
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
from aerodiagnosis.retrieval.reranker import E5SemanticReranker, IdentityReranker


def read(path):
    return json.loads(path.read_text(encoding="utf-8"))


def range_scores(facts, hits, sources):
    """Union exact, non-whitespace source positions; overlapping hits count once."""
    covered = defaultdict(set)
    for hit in hits:
        doc = hit["document_id"]
        lo, hi = hit["start"], hit["end"]
        if doc not in sources or not 0 <= lo < hi <= len(sources[doc]):
            raise ValueError("Retrieved range is outside its source")
        covered[doc].update(range(lo, hi))
    checks = []
    for f in facts:
        text = sources[f["document_id"]]
        if text[f["source_start"] : f["source_end"]] != f["source_quote"]:
            raise ValueError("Frozen annotation extent mismatch")
        required = {i for i in range(f["source_start"], f["source_end"]) if not text[i].isspace()}
        if not required:
            raise ValueError("Reference has no source content")
        checks.append(
            {
                "fact_key": f["fact_key"],
                "full_reference_range_hit": required <= covered[f["document_id"]],
                "reference_character_coverage": len(required & covered[f["document_id"]])
                / len(required),
            }
        )
    return {
        "all_reference_ranges_hit": all(c["full_reference_range_hit"] for c in checks),
        "fact_reference_hit_rate": sum(c["full_reference_range_hit"] for c in checks) / len(checks),
        "mean_reference_character_coverage": sum(c["reference_character_coverage"] for c in checks)
        / len(checks),
        "facts": checks,
    }


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, required=True)
    args = cli.parse_args()
    root = Path.cwd()
    out = args.output.resolve()
    if out.exists():
        raise ValueError("Use a fresh output directory; results are not overwritten")
    dataset = root / "evaluation/gas_path_holdout_v3_candidate"
    experiment = root / ".runtime/experiments/holdout-v3-final-20261014"
    frozen = read(dataset / "freeze.json")
    for name, sha in frozen["file_sha256_bytes"].items():
        if hashlib.sha256((dataset / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Old frozen inputs changed")
    model_path = root / ".runtime/models/multilingual-e5-small"
    for name, sha in frozen["configuration"]["trained_artifact_sha256"].items():
        if hashlib.sha256((model_path / name).read_bytes()).hexdigest() != sha:
            raise ValueError("Embedding model differs from frozen index")
    out.mkdir(parents=True)
    database = out / "development.sqlite"
    shutil.copyfile(experiment / "frozen-knowledge.sqlite", database)
    provider = LocalE5EmbeddingProvider(model_path)
    vector = SQLiteVectorStore(database, embedding_provider=provider)
    lexical = SQLiteLexicalStore(database)
    snapshot = read(experiment / "snapshot.json")
    docs = {d["document_id"]: d["active_version_id"] for d in snapshot["documents"]}
    prepared = root / ".runtime/holdout-v3/prepared-v5"
    sources = {
        d["original_source_id"]: read(prepared / d["parsed_path"])["canonical_content"]
        for d in read(prepared / "catalog.json")["documents"]
    }
    questions = [
        json.loads(s) for s in (dataset / "queries.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    settings = [
        ("bm25", "sparse_only", IdentityReranker()),
        ("e5_dense", "vector_only", IdentityReranker()),
        ("bm25_e5_rrf", "hybrid_rrf", IdentityReranker()),
        ("bm25_e5_rrf_e5_rescore", "hybrid_rrf", E5SemanticReranker(provider)),
    ]
    policies = {
        name: ManualRetrievalPolicy(
            strategy=strategy, candidate_k=50, rrf_k=60, max_per_document=0, expand_context=False
        )
        for name, strategy, _ in settings
    }
    protocol = {
        "status": "registered_before_local_development_run",
        "purpose": "seen-v3-development-only",
        "api_calls": 0,
        "graph_calls": 0,
        "scope": "same explicit source reference in all arms",
        "queries": 120,
        "policies": {k: asdict(v) for k, v in policies.items()},
        "top_k": [5, 10],
        "semantic_relevance_judgments_complete": False,
        "metrics": (
            "specified reference source-range diagnostics; not exhaustive relevance metrics"
        ),
        "parent_expansion": False,
        "end_to_end_token_budget_applied": False,
        "timing": "ordered run with shared warm caches; no fair latency comparison",
        "model": provider.name,
        "snapshot_sha256": frozen["snapshot_sha256"],
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
    }
    (out / "protocol.json").write_text(
        json.dumps(protocol, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    records = []
    for name, _, reranker in settings:
        retriever = ManualEvidenceRetriever(
            vector_store=vector, lexical_store=lexical, policy=policies[name], reranker=reranker
        )
        started = time.perf_counter()
        for q in questions:
            match = re.fullmatch(r"依据资料「([^」]+)」[：:]\s*(.+)", q["query"])
            if not match or match[1] != q["document_id"]:
                raise ValueError("Explicit source scope unavailable")
            version = docs["holdout-v3-" + match[1]]
            found = retriever.search(match[2], top_k=10, active_version_ids=frozenset([version]))
            hits = []
            for m in found:
                c = m.chunk
                if (
                    c.metadata.get("display_translation_source_sha256")
                    != hashlib.sha256(c.content.encode()).hexdigest()
                ):
                    raise ValueError("Display/source range mismatch: " + c.chunk_id)
                co = c.metadata["locator"]["coordinates"]
                doc = c.document_id.removeprefix("holdout-v3-")
                if sources[doc][co["start"] : co["end"]] != c.content:
                    raise ValueError("Returned text does not match original source range")
                hits.append(
                    {
                        "chunk_id": c.chunk_id,
                        "document_id": doc,
                        "start": co["start"],
                        "end": co["end"],
                    }
                )
            facts = [{**f, "document_id": q["document_id"]} for f in q["required_facts"]]
            records.append(
                {
                    "sample_id": q["sample_id"],
                    "source_layer": q["source_layer"],
                    "source_cluster": q["statistical_cluster"],
                    "method": name,
                    "hits": hits,
                    **{f"at_{k}": range_scores(facts, hits[:k], sources) for k in [5, 10]},
                }
            )
        print(
            "Development text retrieval",
            name,
            len(questions),
            "seconds",
            round(time.perf_counter() - started, 2),
            flush=True,
        )
    summary = {"basis": protocol, "subsets": {}}
    for subset in ["source_text", "summary_auxiliary"]:
        summary["subsets"][subset] = {}
        for name, _, _ in settings:
            group = [
                r
                for r in records
                if r["method"] == name
                and (r["source_layer"] == "source_text") == (subset == "source_text")
            ]
            summary["subsets"][subset][name] = {
                "queries": len(group),
                **{
                    f"all_reference_ranges_hit_at_{k}": sum(
                        r[f"at_{k}"]["all_reference_ranges_hit"] for r in group
                    )
                    / len(group)
                    for k in [5, 10]
                },
                **{
                    f"mean_fact_reference_hit_at_{k}": sum(
                        r[f"at_{k}"]["fact_reference_hit_rate"] for r in group
                    )
                    / len(group)
                    for k in [5, 10]
                },
            }
    (out / "rankings.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8"
    )
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary["subsets"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
