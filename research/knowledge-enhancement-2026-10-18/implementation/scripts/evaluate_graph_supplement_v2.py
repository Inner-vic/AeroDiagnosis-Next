# ruff: noqa: E501
"""Frozen independent append-support test, matched text auxiliary control and audits."""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
from capability_metrics import evidence_metrics
from evaluate_capability_development import FrozenGraph, source_texts
from evaluate_strict_graph_holdout import (
    BASE,
    OLD,
    covered_units,
)
from evaluate_strict_graph_holdout import (
    ROOT as PREVIOUS,
)
from evaluate_strict_graph_holdout import (
    inputs as prior_inputs,
)
from frozen_candidate_cache import FrozenCandidateCache
from prepare_graph_supplement_v2 import OUT, ROOT, digest, read, save, sha

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked
from aerodiagnosis.retrieval.fusion_runtime import ConfiguredFusionRetriever
from aerodiagnosis.retrieval.gated_graph import GraphSupplementPolicy
from aerodiagnosis.retrieval.graph_supplement_v2 import (
    DirectedGraphSupport,
    FocusGate,
    append_support,
)

EXP = Path(".runtime/experiments/graph-supplement-v2-holdout-20261016")
BASE_INDEX = Path(".runtime/strict-graph-holdout-v1/index.sqlite")


def strategy_check():
    for p, h in read(ROOT / "strategy-freeze.json")["source_hashes"].items():
        assert digest(Path(p)) == h, p


def build():
    strategy_check()
    data = read(ROOT / "translated-corpus-final.json")
    base = read(PREVIOUS / "corpus-final.json")
    target = OUT / "index.sqlite"
    assert not target.exists()
    before = digest(BASE_INDEX)
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
        dict(
            active_versions=sorted(
                set(base["active_versions"]) | {d["version_id"] for d in data["documents"]}
            ),
            chunks=base["chunks"] + data["chunks"],
            base_index_sha256=before,
            index_sha256=digest(target),
        ),
    )
    assert digest(BASE_INDEX) == before
    print("Index built", len(new), "new chunks", flush=True)


def inputs():
    corpus = read(ROOT / "corpus-final.json")
    _prior, verified = prior_inputs()
    new = read(ROOT / "translated-corpus-final.json")
    sources = source_texts(corpus, new["documents"])
    sources.update(verified)
    return corpus, sources


def freeze():
    strategy_check()
    assert (
        not (EXP / "predictions.jsonl").exists() and not (ROOT / "evaluation-freeze.json").exists()
    )
    corpus, sources = inputs()
    questions = [
        json.loads(s)
        for s in (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
    ]
    assert questions and len({q["sample_id"] for q in questions}) == len(questions)
    for q in questions:
        evidence_metrics(
            sources=sources,
            support_units=q["support_units"],
            acceptable_support_sets=q["acceptable_support_sets"],
            hits=[],
        )
    files = [p for p in ROOT.rglob("*") if p.is_file()]
    files += list(Path("src/aerodiagnosis").rglob("*.py")) + list(Path("scripts").glob("*.py"))
    files += [p / "source-graph-final.json" for p in [BASE, OLD, PREVIOUS]]
    files += [
        Path(r["source_text_file"])
        for r in read(ROOT / "source-selection-after-review.json")["items"]
    ]
    files += list((OUT / "authoring").glob("*.json"))
    files += list((OUT / "extraction").glob("*.json"))
    files += list((OUT / "translation-batches").glob("*.json"))
    files += [
        OUT / name
        for name in [
            "blind-question-review-pack.json",
            "blind-question-private-map.json",
            "index-independent-audit.json",
            "pricing-check.json",
        ]
    ]
    save(
        ROOT / "evaluation-freeze.json",
        dict(
            before_any_prediction=True,
            hashes={p.as_posix(): digest(p) for p in files},
            index_sha256=digest(OUT / "index.sqlite"),
            base_index_sha256=corpus["base_index_sha256"],
            questions=len(questions),
            families=len({q["cluster"] for q in questions}),
            expert_reviewed=False,
            no_test_tuning=True,
            compute="150 text candidates + up to25 graph requests; cache reuse latency not deployment performance; read-only reviewed graph snapshot, not Neo4j latency",
        ),
    )
    print("Frozen", len(questions), flush=True)


def comparison(rs, control):
    if not rs:
        return dict(n=0, families=0, difference=None, p=None, interval=None)
    groups = sorted({r["cluster"] for r in rs})
    sums = np.array(
        [
            sum(
                int(r["complete_all"]) - int(control[r["sample_id"]]["complete_all"])
                for r in rs
                if r["cluster"] == g
            )
            for g in groups
        ]
    )
    counts = np.array([sum(r["cluster"] == g for r in rs) for g in groups])
    rng = np.random.default_rng(20261016)
    draws = rng.integers(0, len(groups), size=(20000, len(groups)))
    delta = sums.sum() / counts.sum()
    samples = sums[draws].sum(1) / counts[draws].sum(1)
    null = (rng.choice([-1, 1], size=(20000, len(groups))) * sums).sum(1) / counts.sum()
    return dict(
        n=len(rs),
        families=len(groups),
        difference=float(delta),
        interval=np.quantile(samples, [0.025, 0.975]).tolist(),
        p=(1 + int(np.sum(np.abs(null) >= abs(delta) - 1e-12))) / 20001,
        wins=sum(r["complete_all"] and not control[r["sample_id"]]["complete_all"] for r in rs),
        losses=sum(control[r["sample_id"]]["complete_all"] and not r["complete_all"] for r in rs),
    )


def analyze(rows):
    summary = {}
    baseline = {r["sample_id"]: r for r in rows if r["method"] == "baseline10"}
    for method in sorted({r["method"] for r in rows}):
        summary[method] = {}
        for kind in ["ALL", *sorted({r["kind"] for r in rows}), "STRICT_MULTIHOP"]:
            rs = [
                r
                for r in rows
                if r["method"] == method
                and (
                    kind == "ALL"
                    or r["kind"] == kind
                    or (kind == "STRICT_MULTIHOP" and r["strict_directed_chain"])
                )
            ]
            summary[method][kind] = dict(
                n=len(rs),
                complete5=sum(r["complete5"] for r in rs),
                complete10=sum(r["complete10"] for r in rs),
                complete_all=sum(r["complete_all"] for r in rs),
                coverage_all=float(np.mean([r["coverage_all"] for r in rs])) if rs else None,
                added_queries=sum(bool(r["trace"].get("added")) for r in rs),
                new_unit_queries=sum(bool(r["new_units"]) for r in rs),
                lost_unit_queries=sum(bool(r["lost_units"]) for r in rs),
                baseline_complete_regressions=sum(
                    baseline[r["sample_id"]]["complete_all"] and not r["complete_all"] for r in rs
                ),
                fallbacks=sum(
                    r["trace"].get("reason") == "graph_unavailable_text_fallback" for r in rs
                ),
                mean_chunks=float(np.mean([len(r["hits"]) for r in rs])) if rs else None,
                mean_tokens=float(
                    np.mean([r["trace"].get("context_tokens", r["tokens"]) for r in rs])
                )
                if rs
                else None,
            )
    save(EXP / "summary.json", summary)
    control = {r["sample_id"]: r for r in rows if r["method"] == "text_append_control"}
    contrasts = {}
    for stratum in ["ALL", "STRICT_MULTIHOP"]:
        rs = [
            r
            for r in rows
            if r["method"] == "append_focus" and (stratum == "ALL" or r["strict_directed_chain"])
        ]
        contrasts[stratum] = comparison(rs, control)
    available = [v for v in contrasts.values() if v["p"] is not None]
    adjusted = 0
    for i, v in enumerate(sorted(available, key=lambda v: v["p"])):
        adjusted = max(adjusted, min(1, (len(available) - i) * v["p"]))
        v["holm_p"] = adjusted
    save(EXP / "paired-comparisons.json", contrasts)
    print(json.dumps({k: v["ALL"] for k, v in summary.items()}, ensure_ascii=False), flush=True)


def evaluate():
    strategy_check()
    frozen = read(ROOT / "evaluation-freeze.json")
    assert all(digest(Path(p)) == h for p, h in frozen["hashes"].items())
    assert digest(OUT / "index.sqlite") == frozen["index_sha256"]
    assert not (EXP / "predictions.jsonl").exists()
    corpus, sources = inputs()
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    active = frozenset(corpus["active_versions"])
    graphs = [read(p / "source-graph-final.json") for p in [BASE, OLD, PREVIOUS, ROOT]]
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

    def load(ids):
        return {k: chunks[k] for k in ids if k in chunks}

    def count(c):
        return costs[c.chunk_id]

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
        token_count=count,
        mode="legacy_dedup",
    )
    methods = {
        "append_without_focus": DirectedGraphSupport(
            text=text, graph=graph, load_chunks=load, token_count=count, focus=False
        ),
        "append_focus": DirectedGraphSupport(
            text=text, graph=graph, load_chunks=load, token_count=count
        ),
    }
    questions = [
        json.loads(s)
        for s in (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
    ]
    rows = []
    EXP.mkdir(parents=True, exist_ok=True)
    with (EXP / "predictions.jsonl").open("x", encoding="utf8") as stream:
        for i, q in enumerate(questions):
            base = text.search(q["query"], top_k=10, active_version_ids=active)
            fused = fuse_ranked({k: v.matches[:75] for k, v in stores.items()}, {"B": 1, "D": 0.25})
            control, trace = append_support(
                baseline=base,
                bundles=[[VectorMatch(c.chunk, c.score)] for c in fused[:50]],
                token_count=count,
            )
            outputs = {
                "baseline10": (base, dict(text.last_trace)),
                "text_append_control": (control, trace),
            }
            for name, retriever in methods.items():
                matches = retriever.search(q["query"], top_k=12, active_version_ids=active)
                outputs[name] = (matches, dict(retriever.last_trace))
            baseunits = None
            for method, (matches, trace) in outputs.items():
                hits = []
                for h in matches:
                    c = h.chunk
                    co = c.metadata["locator"]["coordinates"]
                    hits.append(
                        dict(
                            chunk_id=c.chunk_id,
                            source_key=c.document_id + ":" + c.version_id,
                            start=co["start"],
                            end=co["end"],
                            content=c.content,
                            score=h.score,
                            graph_supplement=c.metadata.get("graph_supplement"),
                        )
                    )
                assert matches[: len(base)] == base
                assert sum(count(h.chunk) for h in matches) <= 4096 and len(matches) <= 12
                assert max(Counter(h.chunk.document_id for h in matches).values(), default=0) <= 7
                m = evidence_metrics(
                    sources=sources,
                    support_units=q["support_units"],
                    acceptable_support_sets=q["acceptable_support_sets"],
                    hits=hits,
                )
                units = covered_units(q, hits)
                if baseunits is None:
                    baseunits = units
                r = dict(
                    sample_id=q["sample_id"],
                    cluster=q["cluster"],
                    kind=q["kind"],
                    strict_directed_chain=q["strict_directed_chain"],
                    method=method,
                    hits=hits,
                    trace=trace,
                    tokens=sum(count(h.chunk) for h in matches),
                    complete5=m["retrieval_evidence_metrics"]["5"]["complete_necessary_evidence"],
                    complete10=m["retrieval_evidence_metrics"]["10"]["complete_necessary_evidence"],
                    coverage10=m["retrieval_evidence_metrics"]["10"][
                        "best_sufficient_set_unit_coverage"
                    ],
                    complete_all=any(set(s) <= units for s in q["acceptable_support_sets"]),
                    coverage_all=max(
                        len(set(s) & units) / len(s) for s in q["acceptable_support_sets"]
                    ),
                    lost_units=sorted(baseunits - units),
                    new_units=sorted(units - baseunits),
                )
                rows.append(r)
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
            stream.flush()
            print("Independent", i + 1, "/", len(questions), flush=True)
    analyze(rows)
    save(
        EXP / "completion.json",
        dict(
            questions=len(questions),
            rows=len(rows),
            api_calls=0,
            predictions_sha256=digest(EXP / "predictions.jsonl"),
            all_frozen_inputs_unchanged=all(
                digest(Path(p)) == h for p, h in frozen["hashes"].items()
            ),
            index_unchanged=digest(OUT / "index.sqlite") == frozen["index_sha256"],
            base_index_unchanged=digest(BASE_INDEX) == frozen["base_index_sha256"],
            production_promoted=False,
        ),
    )


def audit():
    from audit_strict_graph_results import units_covered
    from tokenizers import Tokenizer

    corpus, _ = inputs()
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
    qs = {
        q["sample_id"]: q
        for q in map(
            json.loads, (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
        )
    }
    rs = list(map(json.loads, (EXP / "predictions.jsonl").read_text(encoding="utf8").splitlines()))
    assert len(rs) == 4 * len(qs) and len({(r["sample_id"], r["method"]) for r in rs}) == len(rs)
    baseline = {r["sample_id"]: r for r in rs if r["method"] == "baseline10"}
    token = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    token.no_padding()
    token.no_truncation()
    graphs = [read(p / "source-graph-final.json") for p in [BASE, OLD, PREVIOUS, ROOT]]
    graph = FrozenGraph(
        {
            "nodes": [n for g in graphs for n in g["nodes"]],
            "edges": [e for g in graphs for e in g["edges"]],
        }
    )
    proofs = 0
    for r in rs:
        q = qs[r["sample_id"]]
        hs = r["hits"]
        b = baseline[r["sample_id"]]["hits"]
        assert len(hs) <= 12 and len({h["chunk_id"] for h in hs}) == len(hs)
        assert [(h["chunk_id"], h["score"], h["content"]) for h in hs[: len(b)]] == [
            (h["chunk_id"], h["score"], h["content"]) for h in b
        ]
        assert (
            sum(
                len(
                    token.encode(
                        chunks[h["chunk_id"]].metadata.get("display_content_zh", h["content"])
                    ).ids
                )
                for h in hs
            )
            <= 4096
        )
        assert max(Counter(chunks[h["chunk_id"]].document_id for h in hs).values(), default=0) <= 7
        for h in hs:
            c = chunks[h["chunk_id"]]
            co = c.metadata["locator"]["coordinates"]
            assert (h["source_key"], h["start"], h["end"], h["content"]) == (
                c.document_id + ":" + c.version_id,
                co["start"],
                co["end"],
                c.content,
            )
            if h["graph_supplement"]:
                scope = (c.document_id, c.version_id)
                nodes, edges = graph.snapshot(
                    active_version_ids=frozenset({c.version_id}), limit=2000
                )
                gate = FocusGate(
                    nodes=nodes,
                    edges=edges,
                    chunks=chunks,
                    scope=scope,
                    policy=GraphSupplementPolicy(max_per_document=7),
                )
                if r["method"] == "append_focus":
                    assert gate._path_reason(q["query"], c, h["graph_supplement"]["path"]) is None
                proofs += 1
        for k, field in [(5, "complete5"), (10, "complete10"), (12, "complete_all")]:
            units = units_covered(q, hs[:k])
            assert any(set(s) <= units for s in q["acceptable_support_sets"]) == r[field]
        units = units_covered(q, hs)
        old = units_covered(q, b)
        assert sorted(old - units) == r["lost_units"] and sorted(units - old) == r["new_units"]
        assert not r["lost_units"]
    frozen = read(ROOT / "evaluation-freeze.json")
    assert all(digest(Path(p)) == h for p, h in frozen["hashes"].items())
    assert (
        digest(OUT / "index.sqlite") == frozen["index_sha256"]
        and digest(BASE_INDEX) == frozen["base_index_sha256"]
    )
    save(
        EXP / "independent-output-audit.json",
        dict(
            rows=len(rs),
            path_proofs=proofs,
            all_primary_evidence_preserved=True,
            all_metrics_recomputed=True,
            all_frozen_inputs_unchanged=True,
            all_index_hashes_unchanged=True,
        ),
    )
    print("Audit passed", len(rs), "rows", proofs, "paths", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["build", "freeze", "evaluate", "audit"])
    a = p.parse_args()
    {"build": build, "freeze": freeze, "evaluate": evaluate, "audit": audit}[a.stage]()
