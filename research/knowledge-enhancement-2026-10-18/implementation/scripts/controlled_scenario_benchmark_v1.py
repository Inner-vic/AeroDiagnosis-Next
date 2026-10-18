# ruff: noqa: E501, RUF001
"""Reproducible, explicitly constructed scenario benchmark. No provider API calls."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import random
import shutil
from collections import Counter
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from evaluate_capability_development import FrozenGraph

from aerodiagnosis.adapters.e5_embeddings import LocalE5EmbeddingProvider
from aerodiagnosis.adapters.persistence.sqlite_chinese_vector import SQLiteChineseVectorStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.domain import make_chunk_id
from aerodiagnosis.ingestion import ParserRegistry
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.fusion_policy import fuse_ranked, select_fused
from aerodiagnosis.retrieval.graph_paths import GraphPathCandidates

DATA = Path("evaluation/controlled_scenario_v1")
OUT = Path(".runtime/controlled-scenario-v1")
REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/controlled-scenario-review/review.json"
)
SEED = 20261017
TEMPLATES = [
    ("滤网", "入口滤网堵塞", "入口压损增大", "进气裕度降低", "出口压力波动"),
    ("放气", "放气阀卡开", "非计划泄流增加", "核心流量降低", "推力响应变慢"),
    ("导叶", "导向叶片角度偏置", "进气角偏离设定", "级间流量不匹配", "压比调节异常"),
    ("密封", "级间密封磨损", "级间漏气增加", "有效做功流量下降", "效率指示降低"),
    ("喷嘴", "燃油喷嘴局部堵塞", "燃油分配不均", "热端温度分布不均", "温差报警触发"),
    ("冷却", "冷却孔堵塞", "冷却气流减少", "局部壁温升高", "热负荷监控报警"),
    ("润滑", "润滑支路受阻", "供油流量减少", "轴承散热下降", "轴承温度报警"),
    ("传感", "压力测量通道偏置", "反馈压力值偏高", "调节指令偏低", "实际压力响应不足"),
    ("作动", "作动器响应迟滞", "阀位跟随延后", "流量调节延迟", "转速过渡延长"),
    ("涂层", "涂层结合面分离", "热阻分布改变", "表面温度差增大", "热疲劳监控异常"),
    ("结冰", "入口表面积冰", "入口通流面积缩小", "进口流场畸变", "压气机稳定裕度下降"),
    ("排气", "排气通道局部受阻", "背压升高", "膨胀压比减小", "轴功率指示降低"),
]


def read(p):
    return json.loads(p.read_text(encoding="utf8"))


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf8")


def sha(text):
    return hashlib.sha256(text.encode()).hexdigest()


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def jsonl(p):
    return [json.loads(x) for x in p.read_text(encoding="utf8").splitlines()]


def write_jsonl(p, rows):
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf8")


def prepare():
    assert not DATA.exists(), "Immutable dataset: do not regenerate after observing results."
    DATA.mkdir(parents=True)
    rng = random.Random(SEED)
    order = list(range(len(TEMPLATES)))
    rng.shuffle(order)
    dev_templates = set(order[:4])
    protocol = dict(
        title="自建航空发动机场景评测集",
        schema="controlled-scenario@1",
        seed=SEED,
        provenance="synthetic_rule_generated",
        model_generated=False,
        expert_reviewed=False,
        construction_zh="程序化构造的虚拟试验台场景。部件术语来自航空发动机背景，各因果边、阈值和记录均为本实验明示设定，未逐条获得真实试验确认，不是维修记录或事故调查数据。",
        graph_zh="图由同一组场景规则直接编译，所有边有原始证据记录，属于结构正确的规则图上界；不测试真实资料自动抽取能力。",
        evidence_unit_zh="每条规则为独立证据记录，统一按记录级检索；没有可检索的场景总摘要。场景容器可汇总全部记录，但不是检索单位。本实验研究人为控制的证据分散，不冒充真实文档严格跨块来源认证。",
        split_zh="按12个不同机理模板划分：4模板开发、8模板测试，每模板3个参数化实例，每实例6题。同模板实例与共享链前缀题相关，统计按模板聚类。测试为受控模板留出，不是现实维修外部验证。",
        intended_strata=["exact", "paraphrase", "conditions", "chain2", "chain3", "boundary"],
        methods=["B", "D", "G", "BD", "BG", "DG", "BDG", "BDG_entity"],
        budgets=dict(
            total_candidates=150,
            fused_pool=50,
            final_chunks=10,
            per_document=5,
            context_tokens=4096,
        ),
        dev_weight_grid=[0.25, 0.5, 1.0, 2.0],
        selection="development complete@5, then complete reciprocal rank@10, then lower graph/dense weight; no test-based adjustment",
        statistics="Template-cluster bootstrap20000, seed20261017; confirmatory only for this declared controlled population. Primary BDG vs tuned BD complete@5; other contrasts descriptive. No claim that three>two>one must hold.",
        metrics="Finite rule-world relevance: necessary-record recall, precision,nDCG,first-relevant MRR and complete-support MRR, complete evidence@1/3/5/10. Not generated-answer accuracy.",
        leakage="Retriever receives query and common active versions only, never question kind, template ID, supports, hops, answer or split.",
        background_inspiration_only=[
            "https://ntrs.nasa.gov/citations/20040086786",
            "https://aeroenginesafety.tugraz.at/doku.php?id=7:71:712:712",
        ],
        paid_api_calls=0,
        production_changes=False,
    )
    save(DATA / "protocol.json", protocol)
    docs = []
    chunks = []
    nodes = []
    edges = []
    specs = []
    parser = ParserRegistry(text_strategy="hierarchical", parent_budget=3200, child_budget=600)
    for t, (topic, *states) in enumerate(TEMPLATES):
        for instance in range(3):
            bench = f"ZT-{rng.randrange(10000, 99999)}"
            doc = f"controlled-{t:02}-{instance}"
            labels = [bench + "的" + s for s in states]
            period = rng.choice([12, 18, 24, 32, 40, 48])
            lower = rng.randrange(30, 46)
            upper = lower + 20
            side_a = bench + "的监测通道短时掉线"
            side_b = bench + "的观察日志缺帧"
            records = [
                f"本记录只陈述虚拟试验台{bench}的一条设定规则：当{labels[i]}发生时，引起{labels[i + 1]}。本条不陈述其他后续环节。"
                for i in range(3)
            ]
            records += [
                f"虚拟试验台{bench}的监测规则：{side_a}会导致{side_b}；这是独立于主部件链的记录质量故障。",
                f"参数记录：虚拟试验台{bench}的{topic}通道采样周期固定为{period}毫秒。该参数不提供故障因果链。",
                f"适用记录：虚拟试验台{bench}的试验模式仅用于转速指示{lower}%至{upper}%的闭区间；其他转速不按该模式判读。",
                f"触发记录：虚拟试验台{bench}仅在稳态保持不少于{instance + 2}秒后才进入试验模式；转速适用范围由单独的适用记录规定。",
                f"边界记录：本场景明确规定，虚拟试验台{bench}仅观察到{labels[3]}，不足以确定{labels[0]}已经发生；其他未列出的原因仍可能产生相同末端指示。该逆推不是充分条件。",
            ]
            for j in range(8):
                distract = rng.choice(states)
                records.append(
                    f"独立备忘{j + 1}：虚拟试验台{bench}的{topic}显示页面含“{distract}”字样，但本记录仅登记界面刷新批次{rng.randrange(100, 999)}；不能据此建立因果或认定异常已发生。"
                )
            # Preserve record boundaries and shuffle storage order independently of chain order.
            permutation = list(range(len(records)))
            rng.shuffle(permutation)
            content = "\n\n".join(records[i] for i in permutation)
            h = sha(content)
            version = doc + "-" + h[:20]
            source_key = doc + ":" + version
            record_chunks = {}
            offsets = {}
            cursor = 0
            for j in permutation:
                record = records[j]
                parsed = parser.parse("record.txt", record.encode())
                assert len(parsed.chunks) == 1
                loc = parsed.chunks[0].locator.canonical()
                loc["coordinates"].update(
                    start=cursor, end=cursor + len(record), canonical_sha256=h
                )
                key = make_chunk_id(version, loc, record)
                vc = asdict(
                    VectorChunk(
                        key,
                        doc,
                        version,
                        record,
                        dict(
                            locator=loc,
                            display_content_zh=record,
                            display_translation_source_sha256=sha(record),
                            origin="synthetic_rule_generated",
                            document_metadata=dict(
                                source_type="controlled_scenario_record",
                                source_revision=h,
                                language="zh",
                            ),
                        ),
                    )
                )
                chunks.append(vc)
                record_chunks[j] = key
                offsets[j] = cursor
                cursor += len(record) + 2
            docs.append(
                dict(
                    document_id=doc,
                    version_id=version,
                    source_key=source_key,
                    source_sha256=h,
                    content=content,
                    template_id=f"template-{t:02}",
                    instance=instance,
                    bench=bench,
                    origin="synthetic_rule_generated",
                    record_chunks={str(k): v for k, v in record_chunks.items()},
                )
            )
            for j, (left, right) in enumerate([*itertools.pairwise(labels), (side_a, side_b)]):
                key = record_chunks[j]
                quote = records[j]
                proof = dict(
                    chunk_id=key,
                    document_id=doc,
                    canonical_start=offsets[j],
                    canonical_end=offsets[j] + len(quote),
                    canonical_sha256=h,
                    evidence_text=quote,
                )
                ids = []
                for name in [left, right]:
                    nid = doc + ":mention:" + str(j) + ":" + str(len(ids))
                    ids.append(nid)
                    nodes.append(
                        dict(
                            node_id=nid,
                            name=name,
                            kind="Concept",
                            description="Explicit stipulated state in a constructed bench scenario; not an observed real-engine fault.",
                            source_ref="document:" + doc,
                            version_id=version,
                            properties=dict(
                                entity_link={
                                    **proof,
                                    "kind": "Concept",
                                    "role": "场景状态",
                                    "surface": name,
                                    "review_status": "source_verified_nonexpert",
                                },
                                construction_origin="synthetic_rule_generated",
                            ),
                        )
                    )
                edges.append(
                    dict(
                        edge_id=doc + ":rule:" + str(j),
                        source_id=ids[0],
                        target_id=ids[1],
                        relation="CAUSES",
                        source_ref="document:" + doc,
                        version_id=version,
                        confidence=1.0,
                        properties={
                            **proof,
                            "certainty": "established",
                            "polarity": "positive",
                            "validation_status": "necessary_source_checks_passed",
                            "qualifiers": {
                                "condition": "仅在本虚拟试验台规则世界成立，不是经验确定性或现实维修依据"
                            },
                            "construction_origin": "synthetic_rule_generated",
                        },
                    )
                )
            specs.append(
                dict(
                    doc=doc,
                    bench=bench,
                    labels=labels,
                    period=period,
                    lower=lower,
                    upper=upper,
                    hold=instance + 2,
                    topic=topic,
                    records=records,
                    record_chunks=record_chunks,
                    offsets=offsets,
                    split="dev" if t in dev_templates else "test",
                    template_id=f"template-{t:02}",
                )
            )
    corpus = dict(
        documents=docs,
        chunks=chunks,
        active_versions=[d["version_id"] for d in docs],
        origin="synthetic_rule_generated",
    )
    save(DATA / "corpus.json", corpus)
    save(
        DATA / "graph.json",
        dict(
            nodes=nodes,
            edges=edges,
            origin="synthetic_rule_generated",
            graph_construction="compiled from explicit scenario rules, not extracted by a model",
        ),
    )
    save(
        DATA / "graph-freeze.json",
        dict(
            before_questions=True,
            created_at_utc=datetime.now(UTC).isoformat(),
            corpus_sha256=digest(DATA / "corpus.json"),
            graph_sha256=digest(DATA / "graph.json"),
        ),
    )
    grouped = {"dev": [], "test": []}
    bydoc = {d["document_id"]: d for d in docs}
    for s in specs:
        a, b, c, d = s["labels"]
        bench = s["bench"]
        doc = bydoc[s["doc"]]
        questions = [
            (
                "exact",
                f"{bench}的{s['topic']}通道规定的采样周期是多少毫秒？",
                [4],
                f"{s['period']}毫秒。",
            ),
            (
                "paraphrase",
                f"在{bench}，测量信号间歇消失时，保存下来的观测记录会出现什么变化？",
                [3],
                "观察日志缺帧。",
            ),
            (
                "conditions",
                f"{bench}进入规定试验模式，转速指示范围和稳态保持时间分别应满足什么条件？",
                [5, 6],
                f"转速{s['lower']}%至{s['upper']}%（含端点），稳态不少于{s['hold']}秒。",
            ),
            (
                "chain2",
                f"按{bench}的设定规则，{a}如何通过中间环节引起{c}？请给出连续作用链。",
                [0, 1],
                f"{a}→{b}→{c}。",
            ),
            (
                "chain3",
                f"在{bench}设定中，{a}如何最终引起{d}？请补齐所有必要中间环节。",
                [0, 1, 2],
                f"{a}→{b}→{c}→{d}。",
            ),
            (
                "boundary",
                f"在{bench}仅看到{d}，能否据此确定{a}已经发生？请依据场景边界说明。",
                [7],
                "不能。该末端指示不是该初始异常发生的充分条件，其他原因可能产生同一指示。",
            ),
        ]
        for kind, query, record_ids, answer in questions:
            units = [
                dict(
                    unit_id=str(i),
                    source_key=doc["source_key"],
                    start=s["offsets"][j],
                    end=s["offsets"][j] + len(s["records"][j]),
                    quote=s["records"][j],
                    source_sha256=doc["source_sha256"],
                )
                for i, j in enumerate(record_ids)
            ]
            bridges = [b] if kind == "chain2" else [b, c] if kind == "chain3" else []
            assert all(x not in query for x in bridges)
            grouped[s["split"]].append(
                dict(
                    sample_id=s["doc"] + ":" + kind,
                    query=query,
                    kind=kind,
                    template_id=s["template_id"],
                    scenario_id=s["doc"],
                    split=s["split"],
                    reference_answer=answer,
                    support_units=units,
                    acceptable_support_sets=[[u["unit_id"] for u in units]],
                    relevant_chunk_ids=[s["record_chunks"][j] for j in record_ids],
                    semantic_hops=len(record_ids) if kind.startswith("chain") else 0,
                    hidden_bridge=bridges,
                    origin="synthetic_rule_generated",
                )
            )
    for split, qs in grouped.items():
        write_jsonl(DATA / f"questions-{split}.jsonl", qs)
    (DATA / "README.md").write_text(
        """# 自建航空发动机场景评测集

本数据采用程序化场景构造：36个虚拟试验台实例、12个机理模板。所有规则和阈值是明确的实验设定，不是实测维修记录；图谱直接从同一场景规则编译，评价结构正确图谱的检索能力上界。未做专家确认，不代表自动知识抽取或实际诊断准确率。

每条独立规则是一个证据记录，文本与图谱具有相同事实覆盖。完整场景容器可汇总全部规则，但不作为可检索摘要；这是人为控制证据分散的假设，不能等同真实长文档摘要无法作答。

开发4模板/12实例/72题，测试8模板/24实例/144题，按模板留出。每实例六类：精确、改述、条件、两边链、三边链、逆推边界；两链题共享前缀，统计按模板聚类，不能视作144个独立真实案例。所有模板、种子和题目在测试排名前冻结，失败样本和全部方法均保留。

对外名称可用“自建航空发动机场景评测集”，方法部分须保留上述构建方式与使用边界。量化指标全部来自实际运行，未预设多路排序胜负。
""",
        encoding="utf8",
    )
    save(
        DATA / "dataset-freeze.json",
        dict(
            before_any_rankings=True,
            hashes={p.as_posix(): digest(p) for p in DATA.iterdir() if p.is_file()},
            dev_questions=len(grouped["dev"]),
            test_questions=len(grouped["test"]),
            paid_api_calls=0,
        ),
    )
    print("Prepared", len(docs), len(chunks), {k: len(v) for k, v in grouped.items()}, flush=True)


def check_data():
    f = read(DATA / "dataset-freeze.json")
    assert all(digest(Path(p)) == h for p, h in f["hashes"].items())
    data = read(DATA / "corpus.json")
    byid = {c["chunk_id"]: c for c in data["chunks"]}
    docs = {d["source_key"]: d for d in data["documents"]}
    dev = jsonl(DATA / "questions-dev.jsonl")
    test = jsonl(DATA / "questions-test.jsonl")
    assert not {q["template_id"] for q in dev} & {q["template_id"] for q in test}
    for q in dev + test:
        assert len(set(q["relevant_chunk_ids"])) == len(q["support_units"])
        for u, k in zip(q["support_units"], q["relevant_chunk_ids"], strict=True):
            assert (
                docs[u["source_key"]]["content"][u["start"] : u["end"]]
                == u["quote"]
                == byid[k]["content"]
            )
        if q["kind"].startswith("chain"):
            assert len(q["support_units"]) == q["semantic_hops"] and all(
                b not in q["query"] for b in q["hidden_bridge"]
            )
    return data


def build():
    data = check_data()
    assert not (OUT / "index.sqlite").exists()
    OUT.mkdir(parents=True, exist_ok=True)
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    chunks = [VectorChunk(**c) for c in data["chunks"]]
    SQLiteChineseVectorStore(OUT / "index.sqlite", embedding_provider=provider).upsert(chunks)
    SQLiteLexicalStore(OUT / "index.sqlite").upsert(chunks)
    save(
        OUT / "index-build.json",
        dict(
            index_sha256=digest(OUT / "index.sqlite"),
            chunks=len(chunks),
            provider=provider.name,
            api_calls=0,
        ),
    )
    print("Index built", len(chunks), flush=True)


def cache(split):
    if split == "test":
        frozen = read(OUT / "test-freeze.json")
        assert all(digest(Path(p)) == h for p, h in frozen["hashes"].items())
    data = check_data()
    assert digest(OUT / "index.sqlite") == read(OUT / "index-build.json")["index_sha256"]
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in data["chunks"]}
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    stores = {
        "B": SQLiteLexicalStore(OUT / "index.sqlite"),
        "D": SQLiteChineseVectorStore(OUT / "index.sqlite", embedding_provider=provider),
    }
    graph = FrozenGraph(read(DATA / "graph.json"))

    def load(keys):
        return {k: chunks[k] for k in keys if k in chunks}

    stores["G"] = GraphPathCandidates(graph, load)
    stores["G1"] = GraphPathCandidates(graph, load, max_hops=1)
    active = frozenset(data["active_versions"])
    for i, q in enumerate(jsonl(DATA / f"questions-{split}.jsonl")):
        p = OUT / "candidates" / split / (q["sample_id"].replace(":", "_") + ".json")
        if p.exists():
            assert read(p)["query_sha256"] == sha(q["query"])
            continue
        routes = {}
        for name, store in stores.items():
            matches = store.search(q["query"], top_k=150, active_version_ids=active)
            routes[name] = [dict(chunk_id=m.chunk.chunk_id, score=m.score) for m in matches]
        save(
            p,
            dict(
                sample_id=q["sample_id"], query_sha256=sha(q["query"]), routes=routes, api_calls=0
            ),
        )
        if (i + 1) % 12 == 0:
            print("Candidates", split, i + 1, flush=True)
    save(
        OUT / (split + "-candidate-freeze.json"),
        dict(
            hashes={p.as_posix(): digest(p) for p in (OUT / "candidates" / split).glob("*.json")},
            index_sha256=digest(OUT / "index.sqlite"),
        ),
    )


def configs():
    grid = read(DATA / "protocol.json")["dev_weight_grid"]
    values = [("B", {"B": 1.0}), ("D", {"D": 1.0}), ("G", {"G": 1.0})]
    for method, left, right in [("BD", "B", "D"), ("BG", "B", "G"), ("DG", "D", "G")]:
        values += [(method, {left: 1.0, right: w}) for w in grid]
    values += [("BDG", {"B": 1.0, "D": d, "G": g}) for d, g in itertools.product(grid, grid)]
    return values


def metrics(q, keys):
    relevant = set(q["relevant_chunk_ids"])
    result = {}
    for k in [1, 3, 5, 10]:
        got = set(keys[:k]) & relevant
        dcg = sum((1 / math.log2(i + 2)) for i, key in enumerate(keys[:k]) if key in relevant)
        ideal = sum(1 / math.log2(i + 2) for i in range(min(k, len(relevant))))
        result[str(k)] = dict(
            complete=int(relevant <= set(keys[:k])),
            recall=len(got) / len(relevant),
            precision=len(got) / k,
            ndcg=dcg / ideal,
        )
    first = next((i for i, key in enumerate(keys, 1) if key in relevant), None)
    complete = next((i for i in range(1, len(keys) + 1) if relevant <= set(keys[:i])), None)
    return dict(
        at=result, mrr=1 / first if first else 0, complete_mrr=1 / complete if complete else 0
    )


def run_rank(q, method, weights, chunks, costs):
    raw = read(OUT / "candidates" / q["split"] / (q["sample_id"].replace(":", "_") + ".json"))
    assert raw["query_sha256"] == sha(q["query"])
    channels = list(weights)
    allocation = 150 // len(channels)
    routes = {}
    for name in channels:
        cached_name = "G1" if method == "BDG_entity" and name == "G" else name
        routes[name] = [
            VectorMatch(chunks[r["chunk_id"]], r["score"])
            for r in raw["routes"][cached_name][:allocation]
        ]
    assert allocation * len(channels) == 150
    fused = fuse_ranked(routes, weights)
    selected, trace = select_fused(
        fused,
        token_count=lambda c: costs[c.chunk_id],
        mode="legacy_dedup",
        top_k=10,
        pool_k=50,
        max_per_document=5,
        token_budget=4096,
    )
    keys = [c.chunk.chunk_id for c, _ in selected]
    return dict(
        sample_id=q["sample_id"],
        template_id=q["template_id"],
        scenario_id=q["scenario_id"],
        kind=q["kind"],
        method=method,
        weights=weights,
        hits=keys,
        trace=trace,
        requested_candidates=150,
        metrics=metrics(q, keys),
    )


def setup_rank():
    data = check_data()
    chunks = {c["chunk_id"]: VectorChunk(**c) for c in data["chunks"]}
    provider = LocalE5EmbeddingProvider(Path(".runtime/models/multilingual-e5-small"))
    costs = {k: len(provider.counter.encode(c.content).ids) for k, c in chunks.items()}
    return chunks, costs


def develop():
    assert not (OUT / "strategy-freeze.json").exists()
    chunks, costs = setup_rank()
    qs = jsonl(DATA / "questions-dev.jsonl")
    all_rows = []
    choices = {}
    for method, weights in configs():
        rows = [run_rank(q, method, weights, chunks, costs) for q in qs]
        all_rows.extend(rows)
        key = (
            sum(r["metrics"]["at"]["5"]["complete"] for r in rows),
            sum(r["metrics"]["complete_mrr"] for r in rows),
            -weights.get("G", 0),
            -weights.get("D", 0),
        )
        if method not in choices or key > tuple(choices[method]["selection_key"]):
            choices[method] = dict(weights=weights, selection_key=key)
    choices["BDG_entity"] = dict(choices["BDG"])
    write_jsonl(OUT / "development-all-configurations.jsonl", all_rows)
    save(
        OUT / "strategy-freeze.json",
        dict(
            before_test_candidates_and_rankings=True,
            choices=choices,
            development_rows=len(all_rows),
            evaluator_sha256=digest(Path(__file__)),
            data_freeze_sha256=digest(DATA / "dataset-freeze.json"),
            development_predictions_sha256=digest(OUT / "development-all-configurations.jsonl"),
        ),
    )
    print("Selected", json.dumps(choices), flush=True)


def freeze():
    review = read(REVIEW)
    assert review["approved"] and not review["expert_reviewed"] and not review["predictions_seen"]
    assert not (OUT / "test-predictions.jsonl").exists()
    check_data()
    selected = read(OUT / "strategy-freeze.json")
    assert selected["evaluator_sha256"] == digest(Path(__file__))
    assert selected["data_freeze_sha256"] == digest(DATA / "dataset-freeze.json")
    shutil.copyfile(REVIEW, OUT / "independent-data-review.json")
    files = (
        [p for p in DATA.rglob("*") if p.is_file()]
        + list(Path("src/aerodiagnosis").rglob("*.py"))
        + list(Path("scripts").glob("*.py"))
    )
    files += [
        OUT / "strategy-freeze.json",
        OUT / "index-build.json",
        OUT / "independent-data-review.json",
    ]
    save(
        OUT / "test-freeze.json",
        dict(
            before_test_candidates_and_rankings=True,
            hashes={p.as_posix(): digest(p) for p in files},
            index_sha256=digest(OUT / "index.sqlite"),
            test_questions=len(jsonl(DATA / "questions-test.jsonl")),
            api_calls=0,
        ),
    )
    print("Frozen test", flush=True)


def evaluate():
    candidate_freeze = read(OUT / "test-candidate-freeze.json")
    assert all(digest(Path(p)) == h for p, h in candidate_freeze["hashes"].items())
    f = read(OUT / "test-freeze.json")
    assert all(digest(Path(p)) == h for p, h in f["hashes"].items())
    assert not (OUT / "test-predictions.jsonl").exists()
    chunks, costs = setup_rank()
    choices = read(OUT / "strategy-freeze.json")["choices"]
    rows = []
    for q in jsonl(DATA / "questions-test.jsonl"):
        for method, choice in choices.items():
            rows.append(run_rank(q, method, choice["weights"], chunks, costs))
    write_jsonl(OUT / "test-predictions.jsonl", rows)
    summary = {}
    for method in choices:
        summary[method] = {}
        for kind in ["ALL", *read(DATA / "protocol.json")["intended_strata"]]:
            rs = [r for r in rows if r["method"] == method and (kind == "ALL" or r["kind"] == kind)]
            summary[method][kind] = dict(
                n=len(rs),
                mrr=float(np.mean([r["metrics"]["mrr"] for r in rs])),
                complete_mrr=float(np.mean([r["metrics"]["complete_mrr"] for r in rs])),
                at={
                    str(k): {
                        name: float(np.mean([r["metrics"]["at"][str(k)][name] for r in rs]))
                        for name in ["complete", "recall", "precision", "ndcg"]
                    }
                    for k in [1, 3, 5, 10]
                },
            )
    save(OUT / "test-summary.json", summary)
    lookup = {(r["sample_id"], r["method"]): r for r in rows}
    qs = jsonl(DATA / "questions-test.jsonl")
    groups = sorted({q["template_id"] for q in qs})
    delta = np.array(
        [
            sum(
                lookup[q["sample_id"], "BDG"]["metrics"]["at"]["5"]["complete"]
                - lookup[q["sample_id"], "BD"]["metrics"]["at"]["5"]["complete"]
                for q in qs
                if q["template_id"] == g
            )
            for g in groups
        ]
    )
    counts = np.array([sum(q["template_id"] == g for q in qs) for g in groups])
    rng = np.random.default_rng(SEED)
    draw = rng.integers(0, len(groups), size=(20000, len(groups)))
    estimates = delta[draw].sum(1) / counts[draw].sum(1)
    save(
        OUT / "primary-comparison.json",
        dict(
            left="BDG",
            right="BD",
            metric="complete@5",
            template_clusters=len(groups),
            difference=float(delta.sum() / counts.sum()),
            bootstrap95=np.quantile(estimates, [0.025, 0.975]).tolist(),
            seed=SEED,
            samples=20000,
            scope="controlled stipulated-rule population only; small template count; not real maintenance generalization",
        ),
    )
    print(json.dumps({m: s["ALL"]["at"]["5"] for m, s in summary.items()}), flush=True)


def audit():
    candidate_freeze = read(OUT / "test-candidate-freeze.json")
    assert all(digest(Path(p)) == h for p, h in candidate_freeze["hashes"].items())
    check_data()
    qs = {q["sample_id"]: q for q in jsonl(DATA / "questions-test.jsonl")}
    rows = jsonl(OUT / "test-predictions.jsonl")
    chunks, costs = setup_rank()
    assert len(rows) == len(qs) * 8 and len({(r["sample_id"], r["method"]) for r in rows}) == len(
        rows
    )
    for r in rows:
        q = qs[r["sample_id"]]
        keys = r["hits"]
        assert len(keys) <= 10 and len(set(keys)) == len(keys)
        assert sum(costs[k] for k in keys) <= 4096
        assert max(Counter(chunks[k].document_id for k in keys).values(), default=0) <= 5
        rel = set(q["relevant_chunk_ids"])
        for k in [1, 3, 5, 10]:
            covered = rel & set(keys[:k])
            metric = r["metrics"]["at"][str(k)]
            assert metric["complete"] == int(covered == rel) and metric["recall"] == len(
                covered
            ) / len(rel)
            assert metric["precision"] == len(covered) / k
            dcg = sum(int(key in rel) / math.log2(i + 2) for i, key in enumerate(keys[:k]))
            assert (
                abs(
                    metric["ndcg"]
                    - dcg / sum(1 / math.log2(i + 2) for i in range(min(k, len(rel))))
                )
                < 1e-12
            )
    graph = read(DATA / "graph.json")
    from aerodiagnosis.ports import GraphNode
    from aerodiagnosis.retrieval.graph_paths import _quote_valid, entity_identity

    for node in graph["nodes"]:
        assert entity_identity(GraphNode(**node), chunks).startswith("entity:")
    for edge in graph["edges"]:
        assert _quote_valid(chunks[edge["properties"]["chunk_id"]], edge["properties"])
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
            methods=8,
            all_source_quotes_verified=True,
            all_metrics_recomputed=True,
            frozen_inputs_unchanged=True,
            index_unchanged=True,
            predictions_sha256=digest(OUT / "test-predictions.jsonl"),
            paid_api_calls=0,
            production_promoted=False,
            origin="synthetic_rule_generated",
            expert_reviewed=False,
        ),
    )
    print("Audited", len(rows), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "stage",
        choices=[
            "prepare",
            "build",
            "cache-dev",
            "develop",
            "freeze",
            "cache-test",
            "evaluate",
            "audit",
        ],
    )
    a = p.parse_args()
    if a.stage.startswith("cache-"):
        cache(a.stage.split("-")[1])
    else:
        globals()[a.stage]()
