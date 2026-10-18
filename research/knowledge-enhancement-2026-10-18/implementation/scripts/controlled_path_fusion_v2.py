# ruff: noqa: E501, RUF001
"""Development on seen v1; strategy-first new controlled scenario verification."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import shutil
from collections import Counter
from dataclasses import replace
from pathlib import Path

import build_controlled_scenario_v2 as builder
import controlled_scenario_benchmark_v1 as old
import numpy as np
from evaluate_capability_development import FrozenGraph

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import GraphEdge, GraphNode, VectorChunk, VectorMatch
from aerodiagnosis.retrieval.atomic_path_fusion import (
    atomic_select,
    identifier_scope,
    verified_bundles,
)
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked, select_fused
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates, _quote_valid, entity_identity

DATA = Path("evaluation/controlled_scenario_v2")
OUT = Path(".runtime/controlled-path-fusion-v2")
DEV = Path("evaluation/controlled_scenario_v1")
REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/controlled-v2-review/review.json"
)
read, save, digest, jsonl, write_jsonl = old.read, old.save, old.digest, old.jsonl, old.write_jsonl
METHODS = [
    "B",
    "D",
    "G",
    "BD",
    "BDG",
    "BD_scope",
    "BDG_scope",
    "BDG_entity_scope",
    "BDG_scope_matched",
    "BDG_gated_scope",
    "BDG_atomic_global",
    "BDG_atomic_scope",
]


def protocol():
    assert not (OUT / "protocol.json").exists()
    save(
        OUT / "protocol.json",
        dict(
            name="自建场景路径整束融合验证v2",
            origin="synthetic_rule_generated",
            template_authoring="AI-assisted templates and programmatic expansion; no per-record external LLM calls",
            dev="All216 seen v1 questions; original72+144 now development, never an independent test again.",
            new_test="12 new mechanism vocabulary templates,3 instances each,6 question types=216. Seed20261019 and QB identifiers; rewrite natural question forms before rankings. Same construction grammar and rule-world abstraction, not real external validation.",
            before_new_test_generation=True,
            methods=METHODS,
            budgets=dict(candidates=150, pool=50, top_k=10, per_document=5, tokens=4096),
            scoped_control="Query-only explicit identifier narrows active versions before every channel; unknown identifier gives empty scope. All scoped methods have the same constraint.",
            atomic="Asserted positive causal source paths2..3 edges; exact authority/quotes/forward continuity, both query endpoints, no uncertain or reverse inference. Preserve first1 or2 text hits, insert one complete path or none, then fill text tail. Does not preserve entire old Top10.",
            dev_choices="Scoped BD dense weights .25/.5/1/2; scoped RRF triple dense and graph .25/.5/1/2; atomic same grid and protected prefix1/2. Select complete@5, then complete-support MRR, lower graph weight, larger protected prefix, lower dense weight. Preserve every trial.",
            primary="atomic_scope minus BD_scope and minus matched BDG_gated_scope, complete@5, ALL and chain3;4 contrasts, exact template sign flips+Holm4 and cluster bootstrap20k seed20261019. Describe all8/10-method metrics regardless of sign.",
            graph="Same explicit causal subset as texts; compiled rule structure upper bound, not automatic extraction. Independent AI nonexpert data review before test ranking.",
            data_policy="Preserve v1. No result-conditioned changes/exclusions. New data changes only wording, scenario IDs/values and new templates; keep all6 types equally weighted and all errors.",
            api_calls=0,
            production_promoted=False,
        ),
    )


def dataset(stage):
    return read((DEV if stage == "dev" else DATA) / "corpus.json")


def questions(stage):
    if stage == "dev":
        return jsonl(DEV / "questions-dev.jsonl") + jsonl(DEV / "questions-test.jsonl")
    return jsonl(DATA / "questions-test.jsonl")


def candidates(stage):
    if stage == "test":
        f = read(OUT / "test-freeze.json")
        assert all(digest(Path(p)) == h for p, h in f["hashes"].items())
    data = dataset(stage)
    index = (
        Path(".runtime/controlled-scenario-v1/index.sqlite")
        if stage == "dev"
        else OUT / "index.sqlite"
    )
    before = digest(index)
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in data["chunks"]}
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    sparse = SQLiteLexicalStore(index)
    dense = SQLiteChineseVectorStore(index, embedding_provider=provider)
    graph = FrozenGraph(read((DEV if stage == "dev" else DATA) / "graph.json"))

    def load(ids):
        return {k: chunks[k] for k in ids if k in chunks}

    path = GraphPathCandidates(graph, load)
    entity = GraphPathCandidates(graph, load, max_hops=1)
    active = frozenset(data["active_versions"])
    for i, q in enumerate(questions(stage)):
        target = OUT / "candidates" / stage / (q["sample_id"].replace(":", "_") + ".json")
        if target.exists():
            assert read(target)["query_sha256"] == old.sha(q["query"])
            continue
        scope = identifier_scope(q["query"], chunks)
        scoped = active if scope is None else scope & active
        routes = {}
        for key, store, versions in [
            ("B", sparse, active),
            ("D", dense, active),
            ("G", path, active),
            ("Bs", sparse, scoped),
            ("Ds", dense, scoped),
            ("Gs", path, scoped),
            ("G1s", entity, scoped),
        ]:
            routes[key] = [
                dict(
                    chunk_id=m.chunk.chunk_id,
                    score=m.score,
                    paths=m.chunk.metadata.get("graph_retrieval", {}).get("paths", []),
                )
                for m in store.search(q["query"], top_k=150, active_version_ids=versions)
            ]
        save(
            target,
            dict(
                sample_id=q["sample_id"],
                query_sha256=old.sha(q["query"]),
                scope=sorted(scoped),
                routes=routes,
            ),
        )
        if (i + 1) % 36 == 0:
            print("Cached", stage, i + 1, flush=True)
    assert digest(index) == before
    save(
        OUT / (stage + "-candidate-freeze.json"),
        dict(
            index_sha256=before,
            hashes={p.as_posix(): digest(p) for p in (OUT / "candidates" / stage).glob("*.json")},
        ),
    )


def environment(stage):
    data = dataset(stage)
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in data["chunks"]}
    g = read((DEV if stage == "dev" else DATA) / "graph.json")
    nodes = {x["node_id"]: GraphNode(**x) for x in g["nodes"]}
    edges = {x["edge_id"]: GraphEdge(**x) for x in g["edges"]}
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    costs = {k: len(provider.counter.encode(c.content).ids) for k, c in chunks.items()}
    return chunks, nodes, edges, costs


_VERIFIED_CACHE = {}


def rank(query, raw, method, weights, protect, env):
    chunks, nodes, edges, costs = env
    scoped = "scope" in method
    atomic = "atomic" in method
    channels = list(weights)
    allocation = 150 // len(channels)
    routes = {}
    for name in channels:
        route = (
            "G1s"
            if name == "G" and method == "BDG_entity_scope"
            else name + ("s" if scoped else "")
        )
        routes[name] = [
            VectorMatch(
                replace(
                    chunks[r["chunk_id"]],
                    metadata={
                        **chunks[r["chunk_id"]].metadata,
                        "graph_retrieval": {"paths": r["paths"]},
                    },
                ),
                r["score"],
            )
            for r in raw["routes"][route][:allocation]
        ]
    pool = fuse_ranked(routes, weights)
    chosen, trace = select_fused(
        pool,
        token_count=lambda c: costs[c.chunk_id],
        mode="legacy_dedup",
        top_k=10,
        pool_k=50,
        max_per_document=5,
        token_budget=4096,
    )
    hits = [VectorMatch(c.chunk, s) for c, s in chosen]
    if atomic or method == "BDG_gated_scope":
        text_pool = fuse_ranked(
            {k: v for k, v in routes.items() if k != "G"},
            {k: v for k, v in weights.items() if k != "G"},
        )
        text_chosen, text_trace = select_fused(
            text_pool,
            token_count=lambda c: costs[c.chunk_id],
            mode="legacy_dedup",
            top_k=10,
            pool_k=50,
            max_per_document=5,
            token_budget=4096,
        )
        text_hits = [VectorMatch(c.chunk, s) for c, s in text_chosen]
        cache_key = (query, tuple(m.chunk.chunk_id for m in routes["G"]))
        if cache_key not in _VERIFIED_CACHE:
            _VERIFIED_CACHE[cache_key] = verified_bundles(
                query, routes["G"], nodes=nodes, edges=edges, chunks=chunks
            )
        bundles, rejected = _VERIFIED_CACHE[cache_key]
        if atomic:
            hits, atomic_trace = atomic_select(
                text_hits,
                bundles,
                chunks=chunks,
                token_count=lambda c: costs[c.chunk_id],
                protect=protect,
            )
            trace.update(
                atomic_trace, text_prefix_ids=[m.chunk.chunk_id for m in text_hits[:protect]]
            )
        elif not bundles:
            hits = text_hits
            trace = dict(text_trace, query_gate_fallback=True)
        trace.update(valid_bundles=len(bundles), path_rejections=rejected)
    return [m.chunk.chunk_id for m in hits], trace


def develop():
    assert not DATA.exists() and not (OUT / "strategy-freeze.json").exists()
    env = environment("dev")
    qs = questions("dev")
    raws = {
        q["sample_id"]: read(OUT / "candidates/dev" / (q["sample_id"].replace(":", "_") + ".json"))
        for q in qs
    }
    grid = [0.25, 0.5, 1.0, 2.0]
    specs = [
        ("B", {"B": 1.0}, 1),
        ("D", {"D": 1.0}, 1),
        ("G", {"G": 1.0}, 1),
        ("BD", {"B": 1.0, "D": 2.0}, 1),
        ("BDG", {"B": 1.0, "D": 2.0, "G": 0.25}, 1),
    ]
    specs += [("BD_scope", {"B": 1.0, "D": d}, 1) for d in grid]
    specs += [
        ("BDG_scope", {"B": 1.0, "D": d, "G": g}, 1) for d, g in itertools.product(grid, grid)
    ]
    specs += [
        ("BDG_atomic_scope", {"B": 1.0, "D": d, "G": g}, p)
        for d, g, p in itertools.product(grid, grid, [1, 2])
    ]
    specs += [("BDG_atomic_global", {"B": 1.0, "D": 2.0, "G": 0.25}, p) for p in [1, 2]]
    all_rows = []
    selected = {}
    for method, weights, protect in specs:
        rows = []
        for q in qs:
            keys, trace = rank(q["query"], raws[q["sample_id"]], method, weights, protect, env)
            rows.append(
                dict(
                    sample_id=q["sample_id"],
                    method=method,
                    weights=weights,
                    protect=protect,
                    kind=q["kind"],
                    hits=keys,
                    trace=trace,
                    metrics=old.metrics(q, keys),
                )
            )
        all_rows += rows
        key = (
            sum(r["metrics"]["at"]["5"]["complete"] for r in rows),
            sum(r["metrics"]["complete_mrr"] for r in rows),
            -weights.get("G", 0),
            protect,
            -weights.get("D", 0),
        )
        if method not in selected or key > tuple(selected[method]["selection_key"]):
            selected[method] = dict(weights=weights, protect=protect, selection_key=key)
    for name in ["BDG_entity_scope", "BDG_scope_matched", "BDG_gated_scope"]:
        selected[name] = dict(selected["BDG_atomic_scope"])
    write_jsonl(OUT / "development-all-trials.jsonl", all_rows)
    # Freeze implementation before constructing any new test source or question.
    files = (
        list(Path("src/aerodiagnosis").rglob("*.py"))
        + list(Path("scripts").glob("*.py"))
        + [OUT / "protocol.json", OUT / "development-all-trials.jsonl", DEV / "dataset-freeze.json"]
    )
    save(
        OUT / "strategy-freeze.json",
        dict(
            before_new_test_generation=True,
            dev_questions=len(qs),
            dev_trials=len(specs),
            development_rows=len(all_rows),
            selected=selected,
            hashes={p.as_posix(): digest(p) for p in files},
        ),
    )
    print(
        "Selected",
        json.dumps(
            {
                m: dict(weights=x["weights"], protect=x["protect"], complete=x["selection_key"][0])
                for m, x in selected.items()
            }
        ),
        flush=True,
    )


def check_strategy():
    f = read(OUT / "strategy-freeze.json")
    assert all(digest(Path(p)) == h for p, h in f["hashes"].items())


def prepare():
    check_strategy()
    builder.construct_initial()
    save(OUT / "initial-builder-data-freeze.json", read(DATA / "dataset-freeze.json"))
    old_parts = jsonl(DATA / "questions-dev.jsonl") + jsonl(DATA / "questions-test.jsonl")
    for name in ["questions-dev.jsonl", "questions-test.jsonl", "protocol.json", "README.md"]:
        dest = OUT / "initial-builder-artifacts" / name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(DATA / name), dest)
    corpus = read(DATA / "corpus.json")
    docs = {d["document_id"]: d for d in corpus["documents"]}
    qs = []
    for q in old_parts:
        doc = docs[q["scenario_id"]]
        bench = doc["bench"]
        i = doc["instance"]
        t = int(doc["template_id"].rsplit("-", 1)[1])
        topic, *states = builder.TEMPLATES[t]
        a, _b, c, d = [bench + "的" + x for x in states]
        forms = {
            "exact": [
                f"{bench}中{topic}采集信号的规定周期为多少毫秒？",
                f"请查询{bench}的{topic}通道采样周期。",
                f"{bench}对{topic}通道设定了多长的采样间隔？",
            ],
            "paraphrase": [
                f"{bench}测量信息忽有忽无，留存的观测日志会有什么变化？",
                f"对于{bench}，监测信号中途断续时，记录完整性会怎样？",
                f"{bench}的观测通道偶尔没有信号，保存记录将出现哪种现象？",
            ],
            "conditions": [
                f"{bench}启用指定试验模式时，允许的转速区间及稳态持续时长各是多少？",
                f"在{bench}进入试验模式之前，转速应处于什么范围，还须保持多久？",
                f"请列出{bench}试验模式的转速适用条件与稳态等待条件。",
            ],
            "chain2": [
                f"依据{bench}设定，从{a}到{c}之间应经过什么？请还原传递过程。",
                f"{bench}发生{a}后，怎样经由中间环节产生{c}？",
                f"请说明{bench}中{a}与{c}之间的连续作用路径。",
            ],
            "chain3": [
                f"{bench}设定下，{a}最终怎样演变到{d}？请补全机理路径。",
                f"请还原{bench}从{a}一路传导至{d}的全部必要环节。",
                f"{bench}中的{a}为何会引起{d}？请沿因果路径逐步解释。",
            ],
            "boundary": [
                f"对{bench}，仅凭{d}就足以断定{a}存在吗？请说明逆推边界。",
                f"{bench}出现{d}，是不是就一定已经发生{a}？请依据设定判断。",
                f"只有{d}这项观察时，能否把{bench}的情况确定为{a}？",
            ],
        }
        q.update(
            query=forms[q["kind"]][i],
            split="test",
            template_id="v2-" + doc["template_id"],
            sample_id=q["sample_id"].replace("control2-", "new2-"),
            prompt_variant=i,
        )
        assert all(x not in q["query"] for x in q["hidden_bridge"])
        qs.append(q)
    for d in corpus["documents"]:
        d["template_id"] = "v2-" + d["template_id"]
    save(DATA / "corpus.json", corpus)
    write_jsonl(DATA / "questions-test.jsonl", qs)
    old_templates = set(x[0] for x in old.TEMPLATES)
    assert not old_templates & {x[0] for x in builder.TEMPLATES}
    assert not {d["bench"] for d in corpus["documents"]} & {
        d["bench"] for d in read(DEV / "corpus.json")["documents"]
    }
    save(
        DATA / "protocol.json",
        {
            **read(OUT / "protocol.json"),
            "strategy_sha256": digest(OUT / "strategy-freeze.json"),
            "test_questions": 216,
            "test_templates": 12,
            "scenarios": 36,
            "source_records": 576,
            "query_variants_per_type": 3,
            "graph_edges": 144,
            "construction_note": "All new templates; familiar deterministic rule grammar. No independent real physical source claims. No data rows changed based on test results.",
        },
    )
    (DATA / "README.md").write_text(
        """# 自建航空发动机场景评测集v2

本轮只作新场景测试：12个新机理词汇模板、36个QB编号试验台、576证据记录和216题，六类各36题。模板和生成代码由AI助手协助编写，固定种子程序化展开；规则为场景设定，不是实际维修数据或专家确认的因果规律。图仅编译因果记录子集，属于正确结构辅助上界。

策略在本轮任何语料或题目构建前由旧216题开发选定并冻结。新机理词汇模板与旧集不交叉，问题有三种自然表述；仍共享规则世界与记录级生成结构，不宣称真实数据外部泛化。两链题共享前缀，按12模板聚类。

所有六类均保留，尤其条件和逆推边界；无按策略输赢筛题。检索粒度为记录，场景容器虽能汇总逻辑，却不是可检索的真实文档摘要，因此不能冒充真实来源严格跨自然段认证。完整协议、代码、初始构建产物和冻结记录均保留。
""",
        encoding="utf8",
    )
    # Canonical source text and graph unchanged; document family metadata only renamed.
    save(
        DATA / "dataset-freeze.json",
        dict(
            before_rankings=True,
            hashes={
                p.as_posix(): digest(p)
                for p in DATA.iterdir()
                if p.is_file() and p.name != "dataset-freeze.json"
            },
            strategy_sha256=digest(OUT / "strategy-freeze.json"),
        ),
    )
    print("New test prepared", len(qs), flush=True)


def build():
    check_strategy()
    assert not (OUT / "index.sqlite").exists()
    data = dataset("test")
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    chunks = [VectorChunk(**c) for c in data["chunks"]]
    SQLiteChineseVectorStore(OUT / "index.sqlite", embedding_provider=provider).upsert(chunks)
    SQLiteLexicalStore(OUT / "index.sqlite").upsert(chunks)
    save(
        OUT / "index-build.json",
        dict(index_sha256=digest(OUT / "index.sqlite"), chunks=len(chunks), provider=provider.name),
    )
    print("Built", len(chunks), flush=True)


def freeze():
    check_strategy()
    review = read(REVIEW)
    assert review["approved"] and not review["predictions_seen"] and not review["expert_reviewed"]
    f = read(DATA / "dataset-freeze.json")
    assert all(digest(Path(p)) == h for p, h in f["hashes"].items())
    shutil.copyfile(REVIEW, OUT / "independent-data-review.json")
    files = [p for p in DATA.rglob("*") if p.is_file()] + [
        OUT / "strategy-freeze.json",
        OUT / "index-build.json",
        OUT / "independent-data-review.json",
    ]
    save(
        OUT / "test-freeze.json",
        dict(
            before_candidates_and_rankings=True,
            hashes={p.as_posix(): digest(p) for p in files},
            index_sha256=digest(OUT / "index.sqlite"),
        ),
    )
    print("Test frozen", flush=True)


def summarize(rows):
    result = {}
    for method in METHODS:
        result[method] = {}
        for kind in ["ALL", "exact", "paraphrase", "conditions", "chain2", "chain3", "boundary"]:
            rs = [r for r in rows if r["method"] == method and (kind == "ALL" or r["kind"] == kind)]
            result[method][kind] = dict(
                n=len(rs),
                at={
                    str(k): {
                        name: float(np.mean([r["metrics"]["at"][str(k)][name] for r in rs]))
                        for name in ["complete", "recall", "precision", "ndcg"]
                    }
                    for k in [1, 3, 5, 10]
                },
                mrr=float(np.mean([r["metrics"]["mrr"] for r in rs])),
                complete_mrr=float(np.mean([r["metrics"]["complete_mrr"] for r in rs])),
                atomic_adopted=sum(bool(r["trace"].get("adopted")) for r in rs),
            )
    return result


def evaluate():
    check_strategy()
    f = read(OUT / "test-freeze.json")
    assert all(digest(Path(p)) == h for p, h in f["hashes"].items())
    c = read(OUT / "test-candidate-freeze.json")
    assert all(digest(Path(p)) == h for p, h in c["hashes"].items())
    assert not (OUT / "test-predictions.jsonl").exists()
    env = environment("test")
    selected = read(OUT / "strategy-freeze.json")["selected"]
    rows = []
    for q in questions("test"):
        raw = read(OUT / "candidates/test" / (q["sample_id"].replace(":", "_") + ".json"))
        for method in METHODS:
            config = selected[method]
            keys, trace = rank(q["query"], raw, method, config["weights"], config["protect"], env)
            rows.append(
                dict(
                    sample_id=q["sample_id"],
                    template_id=q["template_id"],
                    kind=q["kind"],
                    method=method,
                    hits=keys,
                    trace=trace,
                    metrics=old.metrics(q, keys),
                )
            )
    write_jsonl(OUT / "test-predictions.jsonl", rows)
    save(OUT / "summary.json", summarize(rows))
    lookup = {(r["sample_id"], r["method"]): r for r in rows}
    contrasts = {}
    for right, kind in itertools.product(["BD_scope", "BDG_gated_scope"], ["ALL", "chain3"]):
        qs = [q for q in questions("test") if kind == "ALL" or q["kind"] == kind]
        groups = sorted({q["template_id"] for q in qs})
        delta = np.array(
            [
                sum(
                    lookup[q["sample_id"], "BDG_atomic_scope"]["metrics"]["at"]["5"]["complete"]
                    - lookup[q["sample_id"], right]["metrics"]["at"]["5"]["complete"]
                    for q in qs
                    if q["template_id"] == g
                )
                for g in groups
            ]
        )
        counts = np.array([sum(q["template_id"] == g for q in qs) for g in groups])
        rng = np.random.default_rng(20261019)
        draws = rng.integers(0, len(groups), size=(20000, len(groups)))
        boot = delta[draws].sum(1) / counts[draws].sum(1)
        null = np.array(
            [
                sum(sign * d for sign, d in zip(signs, delta, strict=True))
                for signs in itertools.product([-1, 1], repeat=len(groups))
            ]
        )
        p = float(np.mean(np.abs(null) >= abs(delta.sum()) - 1e-12))
        contrasts[right + ":" + kind] = dict(
            n=len(qs),
            templates=len(groups),
            difference=float(delta.sum() / counts.sum()),
            bootstrap95=np.quantile(boot, [0.025, 0.975]).tolist(),
            p_exact_cluster_signflip=p,
            scope="controlled rule-world templates, not empirical engine cases",
        )
    adjusted = 0
    for i, v in enumerate(sorted(contrasts.values(), key=lambda x: x["p_exact_cluster_signflip"])):
        adjusted = max(adjusted, min(1, (4 - i) * v["p_exact_cluster_signflip"]))
        v["holm4_p"] = adjusted
    save(OUT / "comparisons.json", contrasts)
    print(json.dumps({m: v["ALL"] for m, v in read(OUT / "summary.json").items()}), flush=True)


def audit():
    check_strategy()
    qs = {q["sample_id"]: q for q in questions("test")}
    rows = jsonl(OUT / "test-predictions.jsonl")
    env = environment("test")
    chunks, nodes, edges, costs = env
    assert len(rows) == 216 * len(METHODS) and len(
        {(r["sample_id"], r["method"]) for r in rows}
    ) == len(rows)
    atomic = 0
    for r in rows:
        q = qs[r["sample_id"]]
        keys = r["hits"]
        assert len(keys) <= 10 and len(set(keys)) == len(keys)
        assert (
            max(Counter(chunks[k].document_id for k in keys).values(), default=0) <= 5
            and sum(costs[k] for k in keys) <= 4096
        )
        if "scope" in r["method"]:
            scope = identifier_scope(q["query"], chunks)
            assert scope is not None and all(chunks[k].version_id in scope for k in keys)
        rel = set(q["relevant_chunk_ids"])
        for k in [1, 3, 5, 10]:
            got = rel & set(keys[:k])
            v = r["metrics"]["at"][str(k)]
            assert (
                v["complete"] == int(rel <= set(keys[:k]))
                and v["recall"] == len(got) / len(rel)
                and v["precision"] == len(got) / k
            )
            dcg = sum(int(key in rel) / math.log2(i + 2) for i, key in enumerate(keys[:k]))
            ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
            assert abs(dcg / ideal - v["ndcg"]) < 1e-12
        if r["trace"].get("adopted"):
            atomic += 1
            assert set(r["trace"]["path_chunks"]) <= set(keys)
            assert keys[: r["trace"]["protected_prefix"]] == r["trace"]["text_prefix_ids"]
            assert r["trace"]["promoted_path_complete_rank"] <= r["trace"][
                "protected_prefix"
            ] + len(r["trace"]["path_chunks"])
    for node in nodes.values():
        assert entity_identity(node, chunks).startswith("entity:")
    for edge in edges.values():
        assert _quote_valid(chunks[edge.properties["chunk_id"]], edge.properties)
    f = read(OUT / "test-freeze.json")
    assert (
        all(digest(Path(p)) == h for p, h in f["hashes"].items())
        and digest(OUT / "index.sqlite") == f["index_sha256"]
    )
    save(
        OUT / "completion-audit.json",
        dict(
            rows=len(rows),
            questions=len(qs),
            atomic_insertions=atomic,
            source_quotes_verified=True,
            all_metrics_recomputed=True,
            all_frozen_inputs_unchanged=True,
            index_unchanged=True,
            predictions_sha256=digest(OUT / "test-predictions.jsonl"),
            api_calls=0,
            production_promoted=False,
        ),
    )
    print("Audited", len(rows), "atomic", atomic, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "protocol",
            "cache-dev",
            "develop",
            "prepare",
            "build",
            "freeze",
            "cache-test",
            "evaluate",
            "audit",
        ],
    )
    a = p.parse_args()
    if a.stage.startswith("cache-"):
        candidates(a.stage.split("-")[1])
    else:
        globals()[a.stage]()
