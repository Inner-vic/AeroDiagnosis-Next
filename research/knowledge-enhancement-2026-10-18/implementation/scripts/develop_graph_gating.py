# ruff: noqa: E501 -- Literal preregistration prose.
"""Fixed small development ablation for source-bound graph supplements; zero API."""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from capability_metrics import evidence_metrics
from develop_capability_fusion import candidate_cache, label_masks, score_hits
from evaluate_capability_development import FrozenGraph, source_texts
from tokenizers import Tokenizer

from aerodiagnosis.ports import VectorMatch
from aerodiagnosis.retrieval.chinese import relevance
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked
from aerodiagnosis.retrieval.fusion_runtime import ConfiguredFusionRetriever
from aerodiagnosis.retrieval.gated_graph import (
    GatedGraphRetriever,
    GraphSupplementPolicy,
    consensus_scope,
    protected_supplement,
)

DATA = Path("evaluation/retrieval_capability_v1_candidate")
OUT = Path(".runtime/experiments/graph-gating-dev60-20261016")
OLD = Path(".runtime/experiments/capability-dev60-20261015")


def READ(p):
    return json.loads(p.read_text(encoding="utf8"))


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


class Cache:
    backend_name = "frozen_query_source_prefix"

    def __init__(self, route, raw, chunks):
        self.route, self.raw, self.chunks = route, raw, chunks

    def search(self, query, *, top_k, active_version_ids):
        return [
            VectorMatch(self.chunks[k], 0)
            for k in self.raw[query][self.route][:top_k]
            if self.chunks[k].version_id in active_version_ids
        ]


def main():
    if OUT.exists():
        raise ValueError("Fresh output directory required")
    for p, h in READ(DATA / "development-freeze.json")["file_sha256"].items():
        assert sha(DATA / p) == h, p
    policies = {
        "source_gate": GraphSupplementPolicy(gate="source"),
        "strict_gate": GraphSupplementPolicy(),
        "uncertainty_gate": GraphSupplementPolicy(gate="uncertainty"),
        "no_eviction_gate": GraphSupplementPolicy(protect=10),
    }
    OUT.mkdir(parents=True)
    frozen = [
        Path(__file__),
        Path("src/aerodiagnosis/retrieval/gated_graph.py"),
        DATA / "questions-final.jsonl",
        DATA / "source-graph-final.json",
        DATA / "corpus.json",
        DATA / "sources.json",
        OLD / "predictions.jsonl",
        Path("src/aerodiagnosis/retrieval/fusion_runtime.py"),
        Path("src/aerodiagnosis/retrieval/fusion_policy.py"),
        Path("src/aerodiagnosis/retrieval/graph_paths.py"),
    ]
    save(
        OUT / "protocol.json",
        dict(
            created_at_utc=datetime.now(UTC).isoformat(),
            development_only=True,
            previous_214_test_not_read=True,
            policies={k: asdict(p) for k, p in policies.items()},
            controls=["text_baseline", "text_pool_control"],
            questions=60,
            api_calls=0,
            parameters_fixed_before_new_scores=True,
            selection="Only strict/uncertainty/no_eviction eligible; require identical protected Top5, zero Top10 question regressions, positive Top10 net vs fixed text baseline; prefer strict then uncertainty then no_eviction. If none qualify retain text default.",
            text_budget="B75/D75 weights1/.25 RRF60 legacy source dedup, same original baseline",
            supplement_budget="additional graph requested25, at most2 source chunks, full output<=10/doc5/token4096; total nominal175 not equal compute150 text",
            confidence="Rule-based provenance/review/intent/uncertainty eligibility, not probability",
            scope="At least2 of original first3 text hits agree on exact document/version; otherwise abstain",
            evaluation_no_strict_chain_claim="60 previously seen questions; manual AI reviewed graph upper bound; no new independent confirmation",
            hashes={p.as_posix(): sha(p) for p in frozen},
        ),
    )
    corpus = READ(DATA / "corpus.json")
    chunks, raw = candidate_cache(corpus)
    questions = [
        json.loads(s)
        for s in (DATA / "questions-final.jsonl").read_text(encoding="utf8").splitlines()
    ]
    query_raw = {q["query"]: raw[q["sample_id"]] for q in questions}
    assert len(query_raw) == 60
    stores = {r: Cache(r, query_raw, chunks) for r in ["B", "D"]}
    counter = Tokenizer.from_file(".runtime/models/multilingual-e5-small/tokenizer.json")
    counter.no_padding()
    counter.no_truncation()
    costs = {
        k: len(counter.encode(c.metadata.get("display_content_zh", c.content)).ids)
        for k, c in chunks.items()
    }

    def load(ids):
        return {k: chunks[k] for k in ids if k in chunks}

    text = ConfiguredFusionRetriever(
        stores=stores,
        weights={"B": 1, "D": 0.25},
        budgets={"B": 75, "D": 75},
        load_chunks=load,
        token_count=lambda c: costs[c.chunk_id],
        mode="legacy_dedup",
    )
    graph = FrozenGraph(READ(DATA / "source-graph-final.json"))
    gated = {
        k: GatedGraphRetriever(
            text=text,
            graph=graph,
            load_chunks=load,
            token_count=lambda c: costs[c.chunk_id],
            policy=p,
        )
        for k, p in policies.items()
    }
    sources = source_texts(corpus, READ(DATA / "sources.json"))
    active = frozenset(corpus["active_versions"])
    labels = label_masks(questions, chunks)
    old_dual = {
        r["sample_id"]: r
        for r in map(
            json.loads, (OLD / "predictions.jsonl").read_text(encoding="utf8").splitlines()
        )
        if r["method"] == "dual"
    }
    rows = []
    with (OUT / "predictions.jsonl").open("x", encoding="utf8") as stream:
        for i, q in enumerate(questions):
            base = text.search(q["query"], top_k=10, active_version_ids=active)
            assert [h.chunk.chunk_id for h in base] == [
                h["chunk_id"] for h in old_dual[q["sample_id"]]["retrieval"]["hits"]
            ]
            result = {"text_baseline": (base, {"added": []})}
            for name, route in gated.items():
                hits = route.search(q["query"], top_k=10, active_version_ids=active)
                assert hits[:5] == base[:5]
                result[name] = (hits, dict(route.last_trace))
            scope = consensus_scope(base)
            original = {h.chunk.chunk_id for h in base}
            pool = fuse_ranked(
                {
                    r: stores[r].search(q["query"], top_k=75, active_version_ids=active)
                    for r in stores
                },
                {"B": 1, "D": 0.25},
            )[:50]
            extra = [
                VectorMatch(c.chunk, c.score)
                for c in pool
                if c.chunk.chunk_id not in original
                and scope
                and (c.chunk.document_id, c.chunk.version_id) == scope
            ]
            extra.sort(
                key=lambda h: (
                    -relevance(
                        q["query"], h.chunk.metadata.get("display_content_zh", h.chunk.content)
                    ),
                    -h.score,
                    h.chunk.chunk_id,
                )
            )
            result["text_pool_control"] = protected_supplement(
                baseline=base,
                bundles=[[h] for h in extra[:25]],
                token_count=lambda c: costs[c.chunk_id],
                policy=GraphSupplementPolicy(),
            )
            for name, (hits, trace) in result.items():
                hs = []
                for h in hits:
                    c = h.chunk
                    co = c.metadata["locator"]["coordinates"]
                    hs.append(
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
                # Only now consult reference labels.
                metric = evidence_metrics(
                    sources=sources,
                    support_units=q["support_units"],
                    acceptable_support_sets=q["acceptable_support_sets"],
                    hits=hs,
                )
                fast = score_hits([h["chunk_id"] for h in hs], labels[q["sample_id"]])
                assert (
                    fast["complete5"]
                    == metric["retrieval_evidence_metrics"]["5"]["complete_necessary_evidence"]
                )
                assert (
                    fast["complete10"]
                    == metric["retrieval_evidence_metrics"]["10"]["complete_necessary_evidence"]
                )
                assert abs(fast["mrr10"] - metric["complete_support_reciprocal_rank"]) < 1e-12
                r = dict(
                    sample_id=q["sample_id"],
                    cluster=q["cluster"],
                    kind=q["kind"],
                    strict_directed_chain=q["strict_directed_chain"],
                    method=name,
                    hits=hs,
                    trace=trace,
                    **fast,
                )
                assert len(hs) <= 10 and sum(costs[h["chunk_id"]] for h in hs) <= 4096
                assert (
                    max(Counter(chunks[h["chunk_id"]].document_id for h in hs).values(), default=0)
                    <= 5
                )
                rows.append(r)
                stream.write(json.dumps(r, ensure_ascii=False) + "\n")
            stream.flush()
            print("Scored", i + 1, "/60 x6", flush=True)
    methods = defaultdict(list)
    for r in rows:
        methods[r["method"]].append(r)
    baseline = {r["sample_id"]: r for r in methods["text_baseline"]}
    summary = {}
    for name, rs in methods.items():
        entry = {}
        for kind in ["ALL", *sorted({r["kind"] for r in rs}), "STRICT_MULTIHOP"]:
            group = [
                r
                for r in rs
                if kind == "ALL"
                or r["kind"] == kind
                or (kind == "STRICT_MULTIHOP" and r["strict_directed_chain"])
            ]
            rejections = Counter(k for r in group for k in [])
            for r in group:
                rejections.update(r["trace"].get("gate_rejections", {}))
            entry[kind] = dict(
                n=len(group),
                complete5=sum(r["complete5"] for r in group),
                complete10=sum(r["complete10"] for r in group),
                mrr10=sum(r["mrr10"] for r in group) / len(group),
                improved10=sum(
                    r["complete10"] and not baseline[r["sample_id"]]["complete10"] for r in group
                ),
                regressed10=sum(
                    baseline[r["sample_id"]]["complete10"] and not r["complete10"] for r in group
                ),
                queries_with_added=sum(bool(r["trace"].get("added")) for r in group),
                queries_with_novel_added=sum(
                    any(
                        k not in {h["chunk_id"] for h in baseline[r["sample_id"]]["hits"]}
                        for k in r["trace"].get("added", [])
                    )
                    for r in group
                ),
                gate_rejections=dict(rejections),
            )
        summary[name] = entry
    eligible = [
        k
        for k in ["strict_gate", "uncertainty_gate", "no_eviction_gate"]
        if summary[k]["ALL"]["regressed10"] == 0 and summary[k]["ALL"]["improved10"] > 0
    ]
    save(OUT / "summary.json", summary)
    save(
        OUT / "selection.json",
        dict(
            selected=eligible[0] if eligible else "text_baseline",
            eligible=eligible,
            production_promotion=False,
            independent_test=False,
            reason="No new independent verification; candidate is opt-in development wrapper only",
        ),
    )
    save(
        OUT / "completion.json",
        dict(
            rows=len(rows),
            questions=60,
            methods=6,
            original_dual_top10_exact_parity=60,
            canonical_and_independent_bitmask_scores_equal=360,
            all_top5_prefixes_identical=True,
            source_inputs_unchanged=all(
                sha(Path(p)) == h for p, h in READ(OUT / "protocol.json")["hashes"].items()
            ),
            predictions_sha256=sha(OUT / "predictions.jsonl"),
            api_calls=0,
        ),
    )
    print("SUMMARY", json.dumps({k: v["ALL"] for k, v in summary.items()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
