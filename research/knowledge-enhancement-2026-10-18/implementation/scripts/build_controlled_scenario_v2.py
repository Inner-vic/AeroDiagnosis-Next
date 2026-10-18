# ruff: noqa: E501, RUF001
"""Reproducible, explicitly constructed scenario benchmark. No provider API calls."""

from __future__ import annotations

import hashlib
import itertools
import json
import random
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path

from aerodiagnosis.domain import make_chunk_id
from aerodiagnosis.ingestion import ParserRegistry
from aerodiagnosis.ports import VectorChunk

DATA = Path("evaluation/controlled_scenario_v2")
OUT = Path(".runtime/controlled-scenario-v2")
REVIEW = Path(
    "D:/zhaoweihao3/Desktop/Project/AeroDiagnosis/.runtime/controlled-scenario-review/review.json"
)
SEED = 20261019
TEMPLATES = [
    ["滤芯", "燃油滤芯污堵", "滤芯压差升高", "旁通供油比例增加", "燃油洁净度指示变差"],
    ["活门", "计量活门开度不足", "计量截面减小", "供油响应受限", "转速建立延迟"],
    ["管路", "引气管路接头松动", "管路漏气增大", "用气端压力下降", "气源监控告警"],
    ["流道", "压气机静子表面沉积", "静子流道粗糙度升高", "叶栅流动损失增加", "级效率监控下降"],
    ["连杆", "可调静子连杆间隙增大", "静子角度同步性降低", "周向入射分布偏离", "级间匹配告警"],
    ["间隙", "机匣局部变形", "叶尖间隙分布偏斜", "间隙泄漏分布不均", "压比波动监控触发"],
    ["雾化", "喷油器雾化孔磨损", "雾滴尺度分布变宽", "蒸发完成位置后移", "燃烧稳定监控异常"],
    ["滑油", "滑油冷却器散热面污染", "滑油散热能力降低", "供油温度升高", "油温监控告警"],
    ["回油", "回油管路局部阻塞", "回油排出能力降低", "轴承腔油量积聚", "腔压监控异常"],
    ["测温", "热电偶接线接触不良", "温度信号间歇波动", "温控输入不稳定", "温控指令频繁变化"],
    ["调压", "供气调压器阀芯粘滞", "供气调压滞后", "气动执行端压力偏离", "执行端到位延迟"],
    ["防冰", "防冰分配支路受阻", "防冰热气分配减少", "受保护表面温度降低", "表面结冰监控触发"],
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


def construct_initial():
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
            bench = f"QB-{rng.randrange(10000, 99999)}"
            doc = f"control2-{t:02}-{instance}"
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
