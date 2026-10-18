# ruff: noqa: E501, RUF001
"""Source-only authoring and answer-hidden review package for mechanism holdout."""

from __future__ import annotations

import argparse

import author_capability_holdout as mature
import make_capability_blind_review_pack as blind
from prepare_strict_graph_holdout import OUT, ROOT, read, save

INSTRUCTION = """仅依据所给原始调查报告，为中文发动机知识证据检索编写最多6道不同事实的题，不用外部知识、不读取图谱或任何检索结果。报告是历史事件分析，不能当成当前维修放行授权。每来源最多各1题：S1_exact为明确术语/具体部件的单事实；S2_paraphrase为另一事实的自然改述；S3_constraints为适用对象/条件交集；S6_counterevidence为明确排除、不确定边界或反证；S4_two_support为报告明示的至少2条连续有向因果/物理机理边；S5_three_support为至少3条连续有向因果/物理机理边。
多跳题必须真正询问从触发失效如何经中间部件/作用导致最终结果，需要至少2个自然解析chunk内的独立必要事实；至少一个必要中间实体不在题面直接给出。不得把程序清单、时序并列、单边因果重复表达、条件交集或泛泛背景伪装成多跳。不可为了增加块数附带不必要问题、硬拆句或故意选择不完整引文；若同一原始块已给出完整答案，应标出这一替代充分支持集合，这样本题不满足跨块严格链。源不支持就跳过该类型并说明，无义务凑6题。若只存在可能链，保留限定，不声称确定链。所有问题尽量覆盖不同实质事实；S4和S5不能仅同一题加一句。
题面和答案以中文为主，发动机型号/注册号/件号可保留标识；不把英文句子或直接待答中间机制写在题面，不使用内部chunk编号或统一空泛模板。题面给自然可辨认的历史事件/发动机语境。每题返回 {kind,query,reference_answer,supports:[{chunk,quote,requirement_zh}],alternative_support_sets:[[supports索引从0开始]],strict_directed_chain:boolean,evidence_structure:single_fact|condition_intersection|multi_block_synthesis|directed_mechanism_chain,necessity_reason_zh,semantic_hops,chain_links:[{from_zh,to_zh,chunk,quote,certainty}],hidden_bridge_zh:[...] }。quote必须在所指chunk内精确连续复制，不能省略拼接，支持最小必要完整子句且含条件/否定/概率。每个supports是题面所问的必答实质事实，不把背景强列必需。若存在完整等价支持必须列出并给出相应alternative_support_sets，不得隐瞒原文总结段。
返回JSON {questions:[...],skipped:[{kind,reason}]}，输出目标少于5500tokens。"""


def run(limit):
    from collections import defaultdict
    from concurrent.futures import ThreadPoolExecutor

    from prepare_strict_graph_holdout import digest, sha

    freeze = ROOT / "graph-freeze.json"
    assert freeze.exists(), "Source graph must be reviewed/frozen first"
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    grouped = defaultdict(list)
    for doc in data["documents"]:
        grouped[doc["source_family"]].append(doc)
    model = mature.client()

    def one(pair):
        family, docs = pair
        target = OUT / "authoring" / (sha(family)[:16] + ".json")
        if target.exists():
            assert read(target)["graph_freeze_sha256"] == digest(freeze)
            return
        entries = [(d, chunks[k]) for d in docs for k in d["chunk_ids"]]
        reply = model.complete(
            task="strict-holdout-author-v1",
            stage="generation_evaluation",
            max_tokens=8192,
            instruction=INSTRUCTION,
            payload={
                "source_documents": [
                    {
                        "source_identity": d["original_source_id"],
                        "agency": d["agency"],
                        "event_identity": d["event_identity"],
                        "limitations": d["source_limitations"],
                    }
                    for d in docs
                ],
                "source_chunks": [
                    {
                        "key": "k" + str(j),
                        "document": d["original_source_id"],
                        "original": c["content"],
                    }
                    for j, (d, c) in enumerate(entries)
                ],
            },
        )
        values = reply.value["questions"]
        allowed = {
            "S1_exact",
            "S2_paraphrase",
            "S3_constraints",
            "S4_two_support",
            "S5_three_support",
            "S6_counterevidence",
        }
        assert len(values) <= 6 and len({q["kind"] for q in values}) == len(values)
        assert all(q["kind"] in allowed and q["query"].strip() for q in values)
        save(
            target,
            {
                "family": family,
                "chunk_key_map": {"k" + str(j): c["chunk_id"] for j, (d, c) in enumerate(entries)},
                "questions": reply.value["questions"],
                "skipped": reply.value.get("skipped", []),
                "graph_freeze_sha256": digest(freeze),
                "receipt": reply.receipt,
            },
        )
        print(
            "Authored", len(list((OUT / "authoring").glob("*.json"))), "/", len(grouped), flush=True
        )

    pending = [
        (f, ds)
        for f, ds in grouped.items()
        if not (OUT / "authoring" / (sha(f)[:16] + ".json")).exists()
    ]
    if limit:
        pending = pending[:limit]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, len(pending), 2):
            list(pool.map(one, pending[start : start + 2]))


def pack():
    blind.ROOT, blind.OUT = ROOT, OUT
    blind.main()
    data = read(OUT / "blind-question-review-pack.json")
    assert (
        data["reference_answers_hidden"]
        and data["gold_supports_hidden"]
        and data["method_and_prediction_hidden"]
    )
    # Do not expose mechanism-bearing cluster labels to the answer-hidden reviewer.
    from prepare_strict_graph_holdout import sha

    catalog = read(ROOT / "source-catalog.json")
    for family in data["families"]:
        family["source_documents"] = [
            {
                "source_identity": d["original_source_id"],
                "agency": d["agency"],
                "event_identity": d["event_identity"],
            }
            for d in catalog["documents"]
            if d["source_family"] == family["family"]
        ]
        family["family"] = "family-" + sha(family["family"])[:16]
    data["mechanism_cluster_labels_hidden"] = True
    data["review_protocol"] = {
        "expert_reviewed": False,
        "author_labels_hidden": True,
        "graph_hidden": True,
        "instructions_zh": "独立从问题和原文重建最小充分答案、必要支持及替代支持，拒绝/修订不可答或泄露答案题。判断真实因果链，禁止因作者计划类别而强认多跳。返回每题 blind_id,query,query_quality(accept/revise/reject),query_revision(若修订),answer_zh,supports[{chunk_id,quote,requirement_zh}],sufficient_sets[[0,1]],minimum_original_chunks,evidence_structure,strict_directed_chain,semantic_hops,chain_links[{from_zh,to_zh,chunk_id,quote,certainty}],hidden_bridge_zh,reason。严格链必须至少两条相接的原文因果/物理机制边，有题面未给出的中间实体，最小充分集合跨至少2个自然chunk；一块有完整总结就不能算跨块严格链。根因未明不能因结果确定反推根因确定。若不同题同一实质事实，请标重复而拒绝后一题。不要打开作者文件/私有映射/图谱/策略/预测。",
    }
    save(OUT / "blind-question-review-pack.json", data)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["author", "pack"])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    run(a.limit) if a.stage == "author" else pack()
