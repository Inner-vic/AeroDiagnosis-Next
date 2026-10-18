"""Source-only second-model verification of corrected and contested major flags."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Chinese audit instructions preserve aviation terminology.
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from adjudicate_holdout_quality import RELATION_ACCEPT, RELATION_REPAIR
from evaluate_chinese_graph_ablation import checked_review
from evaluate_gas_path_holdout import StageClient, digest, read, save

from aerodiagnosis.config import RuntimeSettings


def main():
    root = Path.cwd().resolve()
    if (root / "evaluation/gas_path_holdout_v1_candidate/freeze.json").exists():
        raise ValueError("Only source quality before predictions")
    out = root / ".runtime/experiments/holdout-v1-20261014"
    data = read(out / "snapshot.json")
    audit = read(out / "quality-audit.json")
    client = StageClient(
        RuntimeSettings.from_env(root / ".runtime/holdout-v1/profile"),
        "supplemental_review",
        model="deepseek-v4-pro",
    )
    chunks = {c["chunk_id"]: c for c in data["chunks"]}
    nodes = {n["node_id"]: n for n in data["nodes"]}
    trans = [
        {
            "id": r["id"],
            "original": chunks[r["id"]]["content"],
            "chinese": chunks[r["id"]]["metadata"]["display_content_zh"],
        }
        for r in audit["translations"]
        if r["status"] == "major"
    ]
    edges = [
        {
            "id": e["edge_id"],
            "original": e["properties"]["evidence_text"],
            "source_entity": {k: nodes[e["source_id"]][k] for k in ["name", "kind", "description"]},
            "target_entity": {k: nodes[e["target_id"]][k] for k in ["name", "kind", "description"]},
            "relation": e["relation"],
            "polarity": e["properties"]["polarity"],
            "certainty": e["properties"]["certainty"],
            "qualifiers": e["properties"]["qualifiers"],
        }
        for e in data["edges"]
        if e["edge_id"][:10] in RELATION_ACCEPT | RELATION_REPAIR
    ]
    instruction = (
        "逐项仅依据original检查中文翻译或图谱三元组的技术语义。source_entity是箭头起点，target_entity是终点。"
        "CAUSES=起点导致终点；HAS_SYMPTOM=起点部件具有终点损伤；CONTRIBUTES_TO=起点促进终点；"
        "REQUIRES_ACTION=起点问题在qualifiers限定范围内要求终点动作。polarity=negative表示否定，"
        "certainty和qualifiers必须一起读，不能将可能性或限定范围当成确定普适结论。"
        "保留型号和缩写可接受，Condition类型可以表示性能状态或结果。按实际字段核对方向，不能臆测字段。"
        "翻译可自然改写：提高检查频次等价于缩短检查间隔；不要机械逐字比较。"
        "原始数学符号、图示线性化或跨块断句无法确定时，标记source_corruption，不可凭领域记忆补造公式。"
        '返回 {"items":[{"id":"输入ID","status":"pass|minor|major","source_corruption":true或false,'
        '"issues":[{"original_quote":"精确原文","reason":"具体中文说明"}]}]}。'
        "major仅用于确实改变技术事实或原文不支持的关系；格式和可理解用词为minor。"
    )
    jobs = [("translation", trans[i : i + 2]) for i in range(0, len(trans), 2)] + [
        ("graph", edges[i : i + 4]) for i in range(0, len(edges), 4)
    ]

    def check(job):
        kind, batch = job
        values = checked_review(client, "holdout-source-pro-confirm-v1", instruction, batch, out)
        print("Pro source confirmation", kind, len(batch), flush=True)
        return kind, values

    result = {
        "snapshot_sha256": digest(data),
        "gold_access": False,
        "review_model": "deepseek-v4-pro",
        "expert_reviewed": False,
        "translations": [],
        "graph": [],
    }
    with ThreadPoolExecutor(max_workers=3) as pool:
        for kind, values in pool.map(check, jobs):
            result["translations" if kind == "translation" else "graph"].extend(values)
    save(out / "second-model-source-confirmation.json", result)
    print({k: len(v) for k, v in result.items() if isinstance(v, list)})


if __name__ == "__main__":
    main()
