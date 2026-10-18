"""Finite seen113 ablation, replaying immutable raw ranks; no paid requests."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
from capability_metrics import evidence_metrics
from evaluate_capability_development import FrozenGraph
from evaluate_strict_graph_holdout import BASE, OLD, ROOT, covered_units, inputs
from evaluate_strict_graph_holdout import EXP as PREVIOUS
from prepare_capability_holdout import read, save

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked
from aerodiagnosis.retrieval.gated_graph import GatedGraphRetriever
from aerodiagnosis.retrieval.graph_supplement_v2 import DirectedGraphSupport, append_support

EXP = Path(".runtime/experiments/graph-supplement-v2-dev113-20261016")


class FixedText:
    backend_name = "immutable_text_rank_replay"

    def __init__(self, hits):
        self.hits = hits
        self.last_trace = {}

    def search(self, query, *, top_k, active_version_ids):
        return self.hits[:top_k]


def evaluate():
    assert not (EXP / "predictions.jsonl").exists()
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
    def count(c):
        return costs[c.chunk_id]

    def load(ids):
        return {k: chunks[k] for k in ids if k in chunks}
    old = [
        json.loads(s)
        for s in (PREVIOUS / "predictions.jsonl").read_text(encoding="utf8").splitlines()
    ]
    baselines = {r["sample_id"]: r for r in old if r["method"] == "text_baseline"}
    old_strict = {r["sample_id"]: r for r in old if r["method"] == "strict_gate"}
    questions = [
        json.loads(s)
        for s in (ROOT / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
    ]
    rows = []
    EXP.mkdir(parents=True, exist_ok=True)
    save(
        EXP / "input-freeze.json",
        {
            "protocol": read(Path("evaluation/graph_supplement_v2_development/protocol.json")),
            "hashes": {
                p.as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
                for p in [
                    Path(__file__),
                    Path("src/aerodiagnosis/retrieval/graph_supplement_v2.py"),
                    ROOT / "questions-final.jsonl",
                    PREVIOUS / "predictions.jsonl",
                ]
            },
            "api_calls": 0,
        },
    )
    for i, q in enumerate(questions):
        base = [
            VectorMatch(chunks[h["chunk_id"]], h["score"])
            for h in baselines[q["sample_id"]]["hits"]
        ]
        text = FixedText(base)
        methods = {
            "baseline10": text,
            "v1_strict": GatedGraphRetriever(
                text=text, graph=graph, load_chunks=load, token_count=count
            ),
            **{
                name: DirectedGraphSupport(
                    text=text,
                    graph=graph,
                    load_chunks=load,
                    token_count=count,
                    focus=focus,
                    mode=mode,
                    max_per_document=cap,
                )
                for name, focus, mode, cap in [
                    ("quota7_replace", False, "replace", 7),
                    ("focus_replace", True, "replace", 5),
                    ("focus_quota7_replace", True, "replace", 7),
                    ("append_without_focus", False, "append", 7),
                    ("append_focus", True, "append", 7),
                ]
            },
        }
        key = hashlib.sha256(repr((q["query"], tuple(sorted(active)))).encode()).hexdigest()
        routes = {
            route: [
                VectorMatch(chunks[h["chunk_id"]], h["score"])
                for h in read(PREVIOUS / "raw-candidate-cache" / (route + "-" + key + ".json"))[
                    "hits"
                ][:75]
            ]
            for route in ["B", "D"]
        }
        extra_text = [
            VectorMatch(c.chunk, c.score) for c in fuse_ranked(routes, {"B": 1, "D": 0.25})[:50]
        ]
        control, controltrace = append_support(
            baseline=base, bundles=[[h] for h in extra_text], token_count=count
        )
        methods["text_append_control"] = FixedText(control)
        methods["text_append_control"].last_trace = controltrace
        baseunits = covered_units(q, baselines[q["sample_id"]]["hits"])
        for method, retriever in methods.items():
            limit = 12 if method.startswith("append_") or method == "text_append_control" else 10
            matches = retriever.search(q["query"], top_k=limit, active_version_ids=active)
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
            if method == "v1_strict":
                assert [(h["chunk_id"], h["score"]) for h in hits] == [
                    (h["chunk_id"], h["score"]) for h in old_strict[q["sample_id"]]["hits"]
                ]
            prefix = 10 if limit == 12 else 5
            assert matches[:prefix] == base[:prefix]
            assert sum(count(h.chunk) for h in matches) <= 4096
            assert max(Counter(h.chunk.document_id for h in matches).values(), default=0) <= (
                7 if limit == 12 or "quota7" in method else 5
            )
            metric = evidence_metrics(
                sources=sources,
                support_units=q["support_units"],
                acceptable_support_sets=q["acceptable_support_sets"],
                hits=hits,
            )
            units = covered_units(q, hits)
            rows.append(
                dict(
                    sample_id=q["sample_id"],
                    cluster=q["cluster"],
                    kind=q["kind"],
                    strict_directed_chain=q["strict_directed_chain"],
                    method=method,
                    hits=hits,
                    trace=dict(retriever.last_trace),
                    complete5=metric["retrieval_evidence_metrics"]["5"][
                        "complete_necessary_evidence"
                    ],
                    complete10=metric["retrieval_evidence_metrics"]["10"][
                        "complete_necessary_evidence"
                    ],
                    coverage10=metric["retrieval_evidence_metrics"]["10"][
                        "best_sufficient_set_unit_coverage"
                    ],
                    complete_all=any(set(s) <= units for s in q["acceptable_support_sets"]),
                    coverage_all=max(
                        len(set(s) & units) / len(s) for s in q["acceptable_support_sets"]
                    ),
                    lost_units=sorted(baseunits - units),
                    new_units=sorted(units - baseunits),
                )
            )
        print("Dev", i + 1, "/", len(questions), flush=True)
    (EXP / "predictions.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8"
    )
    summary = {}
    for method in methods:
        summary[method] = {}
        for kind in ["ALL", *sorted({q["kind"] for q in questions}), "STRICT_MULTIHOP"]:
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
                lost_queries=sum(bool(r["lost_units"]) for r in rs),
                new_queries=sum(bool(r["new_units"]) for r in rs),
                added_queries=sum(bool(r["trace"].get("added")) for r in rs),
                fallbacks=sum(
                    r["trace"].get("reason") == "graph_unavailable_text_fallback" for r in rs
                ),
            )
    save(EXP / "summary.json", summary)
    save(
        EXP / "completion.json",
        dict(
            questions=len(questions),
            rows=len(rows),
            api_calls=0,
            v1_replayed_exactly=True,
            prior_inputs_unchanged=all(
                hashlib.sha256(Path(p).read_bytes()).hexdigest() == h
                for p, h in read(ROOT / "evaluation-freeze.json")["hashes"].items()
            ),
        ),
    )
    print(json.dumps({k: v["ALL"] for k, v in summary.items()}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    evaluate()
