# ruff: noqa: E501 -- Chinese research annotations and literal source rows.
"""Matched-budget, zero-API development comparison; graph curation is an upper bound."""

import hashlib
import json
import shutil
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from capability_metrics import evidence_metrics
from evaluate_chinese_hybrid_holdout import paired_cluster_delta

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import GraphEdge, GraphNode, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates
from aerodiagnosis.retrieval.graph_sources import GraphSourceCandidates
from aerodiagnosis.retrieval.manual import _covered_source

DATA = Path("evaluation/retrieval_capability_v1_candidate")
OUT = Path(".runtime/experiments/capability-dev60-20261015")


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def save(p, v):
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2), encoding="utf-8")


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


class FrozenGraph:
    backend_name = "source_frozen_read_only"

    def __init__(self, data):
        self.nodes = [GraphNode(**n) for n in data["nodes"]]
        self.edges = [GraphEdge(**e) for e in data["edges"]]

    def snapshot(self, *, active_version_ids=None, limit=2000):
        return (
            [n for n in self.nodes if n.version_id in active_version_ids],
            [e for e in self.edges if e.version_id in active_version_ids],
        )


class Experiment:
    # Retriever receives only query and fixed source scope; gold type/support/hops stay outside.
    def __init__(self, corpus):
        self.chunks = {c["chunk_id"]: VectorChunk(**c) for c in corpus["chunks"]}
        self.active = frozenset(corpus["active_versions"])
        self.provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
        self.dense = SQLiteChineseVectorStore(
            OUT / "index.sqlite", embedding_provider=self.provider
        )
        self.sparse = SQLiteLexicalStore(OUT / "index.sqlite")

        def load(keys):
            return {k: self.chunks[k] for k in keys if k in self.chunks}

        reviewed = FrozenGraph(read(DATA / "source-graph-final.json"))
        self.paths = GraphPathCandidates(reviewed, load)
        self.entity = GraphPathCandidates(reviewed, load, max_hops=1)
        auto = FrozenGraph(
            read(Path(".runtime/experiments/live-chinese-graph-dev-20261015/graph-snapshot.json"))
        )
        self.automatic = GraphSourceCandidates(auto, load)

    def retrieve(self, query, method):
        trace = []
        routes = []

        def call(name, store, k, weight, q):
            started = time.perf_counter()
            matches = store.search(q, top_k=k, active_version_ids=self.active)
            trace.append(
                dict(
                    route=name,
                    requested=k,
                    returned=len(matches),
                    seconds=time.perf_counter() - started,
                    query=q,
                    graph_trace=dict(getattr(store, "last_trace", {})),
                )
            )
            routes.append((name, weight, matches))

        if method == "bm25":
            call("bm25", self.sparse, 150, 1, query)
        elif method == "dense_zh":
            call("dense", self.dense, 150, 1, query)
        elif method == "dual":
            call("bm25", self.sparse, 75, 1, query)
            call("dense", self.dense, 75, 0.25, query)
        elif method.startswith("triple_"):
            call("bm25", self.sparse, 50, 1, query)
            call("dense", self.dense, 50, 0.25, query)
            store = {
                "triple_entity": self.entity,
                "triple_path": self.paths,
                "triple_automatic": self.automatic,
            }[method]
            call("graph", store, 50, 0.25, query)
        elif method == "iterative_dual":
            expanded = query
            for round_no in range(3):
                call("bm25" + str(round_no), self.sparse, 25, 1 / 3, expanded)
                call("dense" + str(round_no), self.dense, 25, 0.25 / 3, expanded)
                preliminary = self.fuse(routes)[:2]
                # Fixed, source-text-only expansion; same maximum total150 candidates.
                snippets = [
                    str(self.chunks[k].metadata.get("display_content_zh", self.chunks[k].content))[
                        :80
                    ]
                    for k, _, _ in preliminary
                ]
                expanded = query + " " + " ".join(snippets)
        else:
            raise ValueError(method)
        assert sum(t["requested"] for t in trace) == 150
        pool = self.fuse(routes)[:50]
        final = []
        counts = Counter()
        tokens = 0
        selected_chunks = []
        for key, score, contributions in pool:
            chunk = self.chunks[key]
            if counts[chunk.document_id] >= 5:
                continue
            if any(
                _covered_source(VectorMatch(chunk, score), VectorMatch(other, 1))
                for other in selected_chunks
            ):
                continue
            size = len(
                self.provider.counter.encode(
                    str(chunk.metadata.get("display_content_zh", chunk.content))
                ).ids
            )
            if tokens + size > 4096:
                break
            tokens += size
            counts[chunk.document_id] += 1
            selected_chunks.append(chunk)
            final.append(dict(chunk_id=key, score=score, contributions=contributions))
            if len(final) == 10:
                break
        return dict(
            hits=final,
            trace=trace,
            context_tokens=tokens,
            post_fusion_pool=pool,
            routes=[
                dict(name=n, weight=w, chunk_ids=[m.chunk.chunk_id for m in matches])
                for n, w, matches in routes
            ],
        )

    @staticmethod
    def fuse(routes):
        scores = defaultdict(float)
        details = defaultdict(list)
        denominator = sum(weight for _, weight, _ in routes) / 61
        for name, weight, matches in routes:
            seen = set()
            for rank, match in enumerate(matches, 1):
                key = match.chunk.chunk_id
                if key in seen:
                    continue
                seen.add(key)
                value = weight / (60 + rank) / denominator
                scores[key] += value
                details[key].append(dict(route=name, rank=rank, rrf=value))
        return [
            (key, scores[key], details[key])
            for key in sorted(scores, key=lambda k: (-scores[k], k))
        ]


def source_texts(corpus, verified):
    # Gold sources use complete saved report text. Other distractor documents are
    # reconstructed only to validate exact returned ranges, never to invent gold facts.
    groups = defaultdict(list)
    for c in corpus["chunks"]:
        groups[c["document_id"] + ":" + c["version_id"]].append(c)
    result = {}
    for key, chunks in groups.items():
        coords = [c["metadata"]["locator"]["coordinates"] for c in chunks]
        buf = ["\0"] * max(x["end"] for x in coords)
        for c, coordinate in zip(chunks, coords, strict=True):
            lo, hi = coordinate["start"], coordinate["end"]
            assert hi - lo == len(c["content"])
            for pos, char in enumerate(c["content"], lo):
                assert buf[pos] in ("\0", char)
                buf[pos] = char
        result[key] = "".join(buf)
    result.update({s["source_key"]: s["content"] for s in verified})
    return result


def summarize(rows):
    result = {}
    kinds = ["ALL", *sorted({r["kind"] for r in rows}), "STRICT_MULTIHOP"]
    for kind in kinds:
        group = [
            r
            for r in rows
            if kind == "ALL"
            or r["kind"] == kind
            or (kind == "STRICT_MULTIHOP" and r["strict_directed_chain"])
        ]
        result[kind] = {}
        for method in sorted({r["method"] for r in group}):
            subset = [r for r in group if r["method"] == method]
            result[kind][method] = dict(
                n=len(subset),
                complete_at5=sum(r["complete5"] for r in subset),
                complete_at10=sum(r["complete10"] for r in subset),
                rate_at5=float(np.mean([r["complete5"] for r in subset])),
                unit_coverage_at5=float(np.mean([r["coverage5"] for r in subset])),
                complete_mrr10=float(np.mean([r["mrr10"] for r in subset])),
                median_seconds=float(np.median([r["seconds"] for r in subset])),
            )
    paired = []
    for sample in sorted({r["sample_id"] for r in rows}):
        by = {r["method"]: r for r in rows if r["sample_id"] == sample}
        base = by["dual"]
        paired.append(
            dict(
                sample_id=sample,
                cluster=base["cluster"],
                kind=base["kind"],
                strict=base["strict_directed_chain"],
                **{m: float(r["complete5"]) for m, r in by.items()},
            )
        )
    contrasts = {}
    for scope, selected in [
        ("all", paired),
        ("two_three_support", [r for r in paired if r["kind"].startswith(("S4", "S5"))]),
        ("strict_multihop", [r for r in paired if r["strict"]]),
    ]:
        contrasts[scope] = {}
        for left, right in [
            ("triple_path", "dual"),
            ("dual", "bm25"),
            ("dual", "dense_zh"),
            ("triple_path", "triple_entity"),
            ("triple_path", "iterative_dual"),
        ]:
            contrasts[scope][left + "_minus_" + right] = paired_cluster_delta(selected, left, right)
    return dict(
        scope="seen-source development only; graph manually source-curated upper-bound, not automatic improvement",
        api_calls=0,
        per_type=result,
        paired_contrasts=contrasts,
        answer_generation_scored=False,
        limitation="Only2 strict three-support chains; all S6 are answerable limitations, not abstention tests; latency shares model query caches",
    )


def main():
    if OUT.exists():
        raise ValueError("Fresh experiment output required")
    freeze = read(DATA / "development-freeze.json")
    for name, digest in freeze["file_sha256"].items():
        assert sha(DATA / name) == digest, (name, "frozen input changed")
    questions = [
        json.loads(s)
        for s in (DATA / "questions-final.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    corpus = read(DATA / "corpus.json")
    sources = source_texts(corpus, read(DATA / "sources.json"))
    OUT.mkdir(parents=True)
    methods = [
        "bm25",
        "dense_zh",
        "dual",
        "triple_automatic",
        "triple_entity",
        "triple_path",
        "iterative_dual",
    ]
    files = [
        Path(__file__),
        Path("scripts/capability_metrics.py"),
        *Path("src/aerodiagnosis").rglob("*.py"),
    ]
    save(
        OUT / "protocol.json",
        dict(
            created_at_utc=datetime.now(UTC).isoformat(),
            status="registered_before_predictions",
            methods=methods,
            questions=60,
            total_source_candidates_per_method=150,
            common=dict(
                rrf_k=60,
                vector_weight=0.25,
                sparse_weight=1,
                graph_weight=0.25,
                context_tokens=4096,
                max_per_document=5,
                post_fusion_pool=50,
                reranker="identity",
                path_hops=3,
                max_graph_visits=4000,
            ),
            graph_manual_upper_bound=True,
            development_freeze_sha256=sha(DATA / "development-freeze.json"),
            file_sha256={p.as_posix(): sha(p) for p in files},
        ),
    )
    shutil.copyfile(".runtime/holdout-v5/text-index/chinese.sqlite", OUT / "index.sqlite")
    experiment = Experiment(corpus)
    save(OUT / "model.json", dict(identity=experiment.provider.name))
    rows = []
    with (OUT / "predictions.jsonl").open("w", encoding="utf-8") as predictions:
        for i, q in enumerate(questions):
            for method in methods:
                started = time.perf_counter()
                retrieved = experiment.retrieve(q["query"], method)
                hits = []
                for h in retrieved["hits"]:
                    c = experiment.chunks[h["chunk_id"]]
                    coordinate = c.metadata["locator"]["coordinates"]
                    hits.append(
                        dict(
                            source_key=c.document_id + ":" + c.version_id,
                            start=coordinate["start"],
                            end=coordinate["end"],
                            content=c.content,
                        )
                    )
                score = evidence_metrics(
                    sources=sources,
                    support_units=q["support_units"],
                    acceptable_support_sets=q["acceptable_support_sets"],
                    hits=hits,
                )
                metric = score["retrieval_evidence_metrics"]
                row = dict(
                    sample_id=q["sample_id"],
                    cluster=q["cluster"],
                    kind=q["kind"],
                    strict_directed_chain=q["strict_directed_chain"],
                    method=method,
                    complete5=metric["5"]["complete_necessary_evidence"],
                    complete10=metric["10"]["complete_necessary_evidence"],
                    coverage5=metric["5"]["best_sufficient_set_unit_coverage"],
                    mrr10=score["complete_support_reciprocal_rank"],
                    seconds=time.perf_counter() - started,
                    retrieval=retrieved,
                    metrics=score,
                )
                predictions.write(json.dumps(row, ensure_ascii=False) + "\n")
                predictions.flush()
                rows.append(row)
            print(f"completed {i + 1}/60", flush=True)
    save(OUT / "summary.json", summarize(rows))
    save(
        OUT / "completion.json",
        dict(
            completed_at_utc=datetime.now(UTC).isoformat(),
            prediction_rows=len(rows),
            api_calls=0,
            predictions_sha256=sha(OUT / "predictions.jsonl"),
            summary_sha256=sha(OUT / "summary.json"),
        ),
    )


if __name__ == "__main__":
    main()
