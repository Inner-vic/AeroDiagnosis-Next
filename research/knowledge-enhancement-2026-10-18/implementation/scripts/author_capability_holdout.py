# ruff: noqa: E501, RUF001
"""Author source-only independent questions AFTER graph freeze, without rankings or graph input."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict

from author_followup_questions import locate_quote
from prepare_capability_holdout import OUT, ROOT, client, read, save, sha

INSTRUCTION = """仅用保存来源编写中文证据检索测试题。源文是历史适航指令，不是当前维修授权。不要调用外部知识，不编造未提供的SB/表图阈值，不混用FAA/EASA生效日和例外。不得为了某检索法获胜而设计措辞；不用图谱或模型的检索排名。
每来源家族最多6道不同事实的题，尽量各一题：S1_exact原文术语精确事实；S2_paraphrase自然语言改述另一事实；S3_constraints型号/触发条件交集；S4_two_support必须联合两个不同稳定chunk的实质事实/定义才能回答；S5_three_support必须联合三个不同chunk的实质事实才能回答；S6_counterevidence原文明确的否定/豁免/可能性边界。
只能根据真正的最低必要证据标注，不用背景硬凑块数。尤其S4/S5：题面必须明确问到各条必要事实，不能只问一个块能回答的问题再把其他块背景设必需。不支持某类就跳过，并给skipped理由；无义务凑齐6题。S5综合三块不等于三跳，严格链须source明示有向机制且桥接实体不在题面全部给出。不要把程序清单/条件交集误写成因果链。
每题返回 {kind,query,reference_answer,supports:[{chunk,quote,requirement_zh}],alternative_support_sets:[[support索引从0开始]],strict_directed_chain:false,evidence_structure:"single_fact|condition_intersection|multi_block_synthesis|directed_mechanism_chain",necessity_reason_zh}。supports的quote必须逐字连续复制某一提供块的原文，不用省略号或拼接，不必引整段但应包含不确定性和条件。只引用答案所需最小完整句/子句。gold_support_eligible=false的块不得作为标注依据。分页拆开的同一句不算两个独立必要事实；若不能在其他完整条款定位则跳过，不能作为S4/S5的证据数优势。若存在完整等价支持，请列出替代supports并在alternative_support_sets中给出完整充分集合。
返回JSON {questions:[...],skipped:[{kind,reason}]}。题面包含可区分发动机型号和部件的自然中文语境，不使用统一泛泛模板，不直接泄露答案。所有问题可回答；S6仅在来源明确否定/例外时出，不从缺词推出无答案。query、reference_answer、requirement_zh必须以自然中文撰写，只有型号/件号/通告编号等标识可保留英文，不许整段英文答案。题面不能预先给出待答事实；S1应问缺失事实槽，不可把适用性整段抄入题面后再问如何表述。避免不同题实质重复同一事实。输出尽量简洁，参考答案只答题面所问的必要事实，不抄整页或整张表，引用最小完整子句，目标总输出少于4500token。"""


def run(limit):
    freeze = ROOT / "graph-freeze.json"
    if not freeze.exists():
        raise ValueError("Source graph must be reviewed and frozen before authoring")
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    docs = data["documents"]
    grouped = defaultdict(list)
    for doc in docs:
        grouped[doc["source_family"]].append(doc)
    limitations = {
        x["document_id"]: x for x in read(ROOT / "supplement-source-review.json")["items"]
    }
    model = client()

    def author_one(pair):
        family, rows = pair
        target = OUT / "authoring" / (sha(family)[:16] + ".json")
        if target.exists():
            return
        entries = [(d, chunks[k]) for d in rows for k in d["chunk_ids"]]
        keymap = {"k" + str(j): c["chunk_id"] for j, (d, c) in enumerate(entries)}
        reply = model.complete(
            task="capability-holdout-author-v1",
            stage="generation_evaluation",
            max_tokens=8192,
            instruction=INSTRUCTION,
            payload={
                "family": family,
                "source_limitations": [
                    limitations[d["original_source_id"]] for d in rows if d["agency"] == "FAA"
                ],
                "source_chunks": [
                    {
                        "key": "k" + str(j),
                        "document": d["document_id"],
                        "agency": d["agency"],
                        "original": c["content"],
                        "gold_support_eligible": not bool(
                            c["metadata"].get("source_annotation_restriction")
                        ),
                        "quality_restriction": c["metadata"].get("source_annotation_restriction"),
                    }
                    for j, (d, c) in enumerate(entries)
                ],
            },
        )
        save(
            target,
            {
                "family": family,
                "chunk_key_map": keymap,
                "questions": reply.value["questions"],
                "skipped": reply.value.get("skipped", []),
                "graph_freeze_sha256": hashlib.sha256(freeze.read_bytes()).hexdigest(),
                "receipt": reply.receipt,
            },
        )
        print(
            "Authored", len(list((OUT / "authoring").glob("*.json"))), "/", len(grouped), flush=True
        )

    from concurrent.futures import ThreadPoolExecutor

    pending = [
        (f, rows)
        for f, rows in grouped.items()
        if not (OUT / "authoring" / (sha(f)[:16] + ".json")).exists()
    ]
    if limit:
        pending = pending[:limit]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for start in range(0, len(pending), 2):
            list(pool.map(author_one, pending[start : start + 2]))


def compile_questions():
    data = read(ROOT / "translated-corpus-final.json")
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    docs = {d["document_id"]: d for d in data["documents"]}
    questions = []
    rejected = []
    skipped = []
    files = list((OUT / "authoring").glob("*.json"))
    assert len(files) == len({d["source_family"] for d in docs.values()})
    for f in sorted(files):
        row = read(f)
        skipped.extend([{"family": row["family"], **v} for v in row["skipped"]])
        seen = set()
        for i, q in enumerate(row["questions"]):
            try:
                if (
                    q["kind"]
                    not in {
                        "S1_exact",
                        "S2_paraphrase",
                        "S3_constraints",
                        "S4_two_support",
                        "S5_three_support",
                        "S6_counterevidence",
                    }
                    or q["kind"] in seen
                ):
                    raise ValueError("Unknown/duplicate kind")
                units = []
                ids = []
                align = []
                for j, u in enumerate(q["supports"]):
                    c = chunks[row["chunk_key_map"][str(u["chunk"])]]
                    d = docs[c["document_id"]]
                    if c["metadata"].get("source_annotation_restriction"):
                        raise ValueError("Gold source has unresolved annotation restriction")
                    lo, quote, mode = locate_quote(c["content"], u["quote"])
                    lo += c["metadata"]["locator"]["coordinates"]["start"]
                    assert d["content"][lo : lo + len(quote)] == quote
                    units.append(
                        {
                            "unit_id": str(j),
                            "source_key": d["source_key"],
                            "start": lo,
                            "end": lo + len(quote),
                            "quote": quote,
                            "source_sha256": d["source_sha256"],
                            "requirement_zh": u["requirement_zh"],
                        }
                    )
                    ids.append(c["chunk_id"])
                    align.append(mode)
                sets = q.get("alternative_support_sets") or [list(range(len(units)))]
                sets = [[str(j) for j in s] for s in sets]
                if any(not s or any(int(j) >= len(units) or int(j) < 0 for j in s) for s in sets):
                    raise ValueError("Invalid sufficient set")
                minimum = min(len({ids[int(j)] for j in s}) for s in sets)
                required = {"S4_two_support": 2, "S5_three_support": 3}.get(q["kind"], 1)
                # Preserve candidate separately; source reviewer must inspect semantic minimality/alternatives before freeze.
                if minimum < required:
                    raise ValueError("Provided supports fail declared minimum stable chunk count")
                question = {
                    "sample_id": "cap1:" + sha(row["family"])[:12] + ":" + q["kind"],
                    "cluster": row["family"],
                    "kind": q["kind"],
                    "query": q["query"],
                    "reference_answer": q["reference_answer"],
                    "answerable": True,
                    "support_units": units,
                    "acceptable_support_sets": sets,
                    "necessary_original_chunks": minimum,
                    "provided_chunk_ids": ids,
                    "strict_directed_chain": q.get("strict_directed_chain", False),
                    "evidence_structure": q["evidence_structure"],
                    "necessity_reason_zh": q["necessity_reason_zh"],
                    "quote_alignment": align,
                    "expert_reviewed": False,
                    "annotation_status": "pending_independent_nonexpert_source_review",
                }
                questions.append(question)
                seen.add(q["kind"])
            except (KeyError, ValueError, TypeError, AssertionError) as e:
                rejected.append(
                    {
                        "family": row["family"],
                        "question_index": i,
                        "reason": str(e)[:180],
                        "original_candidate": q,
                    }
                )
    (ROOT / "questions-candidate.jsonl").write_text(
        "".join(json.dumps(q, ensure_ascii=False) + "\n" for q in questions), encoding="utf8"
    )
    save(
        ROOT / "question-authoring-audit.json",
        {
            "n": len(questions),
            "rejected_before_any_prediction": rejected,
            "skipped_source_unsupported": skipped,
            "source_family_count": len({q["cluster"] for q in questions}),
        },
    )
    print("Compiled", len(questions), "rejected", len(rejected), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["author", "compile"])
    p.add_argument("--limit", type=int, default=0)
    a = p.parse_args()
    run(a.limit) if a.stage == "author" else compile_questions()
