"""Apply explicit source-only quality adjudication before any system predictions."""

from __future__ import annotations

# ruff: noqa: RUF001 -- Source adjudication uses exact Chinese translation fragments.
import json
import sqlite3
from dataclasses import replace
from pathlib import Path

from evaluate_gas_path_holdout import digest, read, save, snapshot

from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.ports import GraphEdge, GraphNode

TRANSLATION_EDITS = {
    "0d90660588": [
        (
            "发动机在低于允许限值93.3%的N1转速下运行，即低于允许限值2.3%，排气温度（EGT）为低于允许限值590°C的60°C",
            "发动机的N1转速比93.3%的允许限值低2.3个百分点，排气温度（EGT）比590°C的允许限值低60°C",
        )
    ],
    "174c4b9c8b": [("低排放区域", "排油不畅区域")],
    "1fa14b44d7": [
        ("解除撤离滑梯的保险", "将撤离滑梯预位"),
        ("应急撤离滑梯已解除保险", "应急撤离滑梯已预位"),
    ],
    "398b40e8e5": [("在最小慢车功率下运行5分钟之前关车", "在最小慢车功率下运行不足5分钟便关车")],
    "4bc3494298": [("进行有地效外悬停", "进行无地效悬停")],
    "50db299118": [("一个发动机机队中偏差均方和", "整个发动机机队范围内偏差平方的均值之和")],
    "727bbd102d": [
        ("CT叶片环", "CT导向叶片环"),
        ("压气机涡轮叶片环", "压气机涡轮导向叶片环"),
        ("叶片遭受了磨损", "导向叶片遭受了磨损"),
    ],
    "7cc93a314c": [
        (
            (
                "do not persist to the following flight as the goal of this study is to "
                "accurately detect and isolate faults during the flight on which they occur."
            ),
            "不会延续至下一次飞行，因为本研究的目标是在故障发生的当次飞行中准确检测并隔离故障。",
        )
    ],
    "7cdffc904f": [("艾伯塔省麦克默里堡", "艾伯塔省麦克凯堡")],
    "8db3c10b89": [("检查频率从450小时增加到350小时", "检查间隔从450小时缩短至350小时")],
    "9b8aea180e": [("压气机涡轮叶片环", "压气机涡轮导向叶片环"), ("CT 叶片环", "CT 导向叶片环")],
    "d77a444b8e": [
        ("consequential 损坏", "随之产生的损坏"),
        ("Maximum Available Horsepower 限制", "最大可用马力限制"),
    ],
    "e0b0fb324c": [
        ("自其20000总循环限制起已累计909循环", "在20000次总循环寿命限值内已累计909次循环")
    ],
    "f95fe30976": [
        (
            "使用光学摄影和图像分析对一个新的密封圈测量了该发生件密封圈的若干尺寸。",
            "利用光学摄影以及对新密封圈的图像分析，测量了涉事密封圈的若干尺寸。",
        )
    ],
}
TRANSLATION_RETAIN = {
    "00c1614479": "原文跨块断句，后续条件保存在相邻块/父上下文；不将脚注号78当限制值。",
    "1695520dbc": "also确实出现在原文；中文已保留比较对象，建议文本与现译相同。",
    "3a987364dd": "NL排版及±记号为原文形式，数值和可能性均正确；仅格式问题。",
    "49aa0a831e": "源文本数学符号存在排版歧义；禁止凭常识把第二个p改成t。",
    "49c36da3bf": "源文本向量符号可能失去上下标；禁止猜测把yb改为ya。",
    "4b12499535": "HTP/HPT混用来自原文，800循环与800小时均保留正确；不替来源纠错。",
    "622f52b35c": "nor after明确同处否定范围；添加原文没有的立即会改变事实。",
    "814775b0b6": (
        "原始派生PDF文本公式排版已损坏，保留原样并声明公式不能可靠还原；不接受模型补写公式。"
    ),
    "844d883303": "自然中文语法连接没有添加技术事实，未来性和推测范围保留。",
    "9786b965ad": "表头单元格语义仍可辨；百分比平方单位保留，属于格式问题。",
    "adaa650275": "部分装有燃油的中央油箱及其空隙关系正确，未改变穿越对象。",
    "c15c18a3e1": "原译件号逐字一致；审核建议反而漏数字，拒绝该建议。",
    "e29df304d2": "源文本为图示的线性化内容；未凭推测加入额外加号或改变热力学关系。",
}
RELATION_ACCEPT = {
    "1feaf9e647": "LPT1 bore到seal land failure方向有精确原句支持；内孔译名未改变对象。",
    "4369bca4fc": "because of fatigue直接支持疲劳导致叶片脱落；源类型已经是Cause，审核误报。",
    "4428095834": "liberated blade caused internal failure明确支持当前方向和名称。",
    "5b56995348": "当前方向已经是质量中心线偏移到不平衡状态，正确表达被动语态。",
    "b723358d9e": "might be critical to支持有条件的贡献关系，hypothesis未被强化；类型改为动作。",
}
RELATION_REPAIR = {
    "001abdc3b1": "将碎片本体的概念显示明确为原文支持的损伤风险；仅限该文档所述安装防护措施。",
    "1be92aa992": "部件表现损伤应为HAS_SYMPTOM，不能把部件当作诊断指示。",
    "1c26a69638": "因果源概念明确为燃油和衰减排气混合机制，并保留排气管内条件。",
    "3389f070b4": "保持原文叶片导致另外五片损坏的方向，明确特定对象和数量。",
    "4451dfbd5e": "保留must retire与hypothesis，补入严重热瞬态次数限制的精确条件。",
    "cf36e71103": "诊断准确性提升是性能状态，不是故障症状；关系原句直接支持。",
    "cf7e06ea57": "断裂是转子上的损伤状态，改为HAS_SYMPTOM；不当作正常子部件。",
    "d50eeb418d": "affecting engine performance支持性能受影响，目标改为状态类型。",
    "e6dee63ea7": "原句likely to affect支持损伤到性能受影响，方向正确；目标明确为受影响状态。",
}
# These statements cannot safely express the currently extracted predicate. Preserve
# raw records and source prose, but exclude their graph assertions from retrieval.
RELATION_REJECT = {
    "073ae8a470": "结果实体事件因素不明确；否定原始记录已不参与检索，追加语义拒绝标记。",
    "19c5918143": "验证技术防范陷阱不能表示FDIA导致陷阱，否定范围也不同。",
    "19ce4c5952": "历史检查已发生不等于要求检查。",
    "22db54e791": "检查间隔依据读数不等于要求动作，当前谓词不适用。",
    "239819d604": "机构不是部件，经历裂纹问题不等于INDICATES。",
    "316aa71e1d": "未见异常证据不等于不存在异常引起裂纹的因果关系。",
    "35a589b155": "机构不是部件，历史采纳不等于动作要求。",
    "35dd71802f": "材料耐腐蚀属性不等于材料被腐蚀缓解。",
    "42fcb92d8a": "历史关车事件不等于通用关车要求。",
    "4435d83de2": "叶片本体不是拉伸过载断裂的原因；拒绝而不重造实体来源跨度。",
    "453d7613a0": "样本内相关条件不等于密封接触可诊断腐蚀。",
    "47548ceb76": "原句直接原因是高周疲劳，孔隙率为区域条件，不应替换成直接原因。",
    "4d4d79a472": "退化和衰退是同一现象的表述，不是独立因果对。",
    "4de64eaee0": "未识别贡献因素不等于证实没有贡献；保留原句而拒绝该断言。",
    "50980643f5": "发现碎屑不等于碎屑吸引探测器。",
    "5ab21b5647": "限定样本的状态观察不等于诊断指示；否定极性也误用了作用范围。",
    "5cbbba810c": "未粘接事实和as required要求的否定范围不能用整条negative表示。",
    "5fe0de6c3c": "检查中发现腐蚀不等于腐蚀指示检查动作。",
    "60b05c1843": "while仅为同时状态，上文as consequence所指原因不能省略。",
    "66967431a2": "监管机构要求不等于以监管机构作为检查手段。",
    "6d3947389a": "试验后观察变化不足以支持当前独立因果关系。",
    "6ef6abaa41": "清洗为降低退化目的不能自动转为污染物对退化的完整因果断言。",
    "7274ed1046": "已拆检和未发现故障不等于要求拆检。",
    "732bffcd7b": "一次历史更换不等于所有此类裂纹都要求更换。",
    "738007ccce": "排查发现滑油不是滑油指示排查动作。",
    "88b6913050": "温度阈值下观察变化缺少独立因果证明，不把阈值当已证实原因。",
    "97a252f7b3": "FADEC响应事件不等于规范性恢复要求。",
    "b01fde38c3": "压力油为泄漏主体非独立原因，当前断言还丢失74%N1条件。",
    "b3bd246fab": "同时发生不等于一项对另一项有贡献。",
    "b479d673e2": "防护未抵消腐蚀的否定作用范围及MITIGATED_BY方向错误。",
    "c9d2a39761": "一次更换后的人员确信不等于通用更换要求或确诊。",
    "d05a31e9e2": "一次机组响应不能写成通用动作要求。",
    "d2c7957066": "检查对象为其他发动机，不能直接当后侧板密封的检查手段。",
    "d4cf51eb2a": "需保留EEC响应主体，当前因果对省略主体。",
    "d6848caedb": "声音类似失速不等于确认失速导致声音。",
    "da918e3d10": "温度条件下未观察变化不等于否定完整因果关系。",
    "dcea89f264": "运营人提出建议并不自动产生通用行动要求。",
    "e361654c1f": "防腐涂层不会由腐蚀缓解；属性比较不能支持当前反向缓解关系。",
    "ea3c6a9b17": "当前裂纹与历史裂纹相似不等于确认相同原因。",
    "f7fec1fe1d": "未发现泄漏起火证据不等于证明不存在这一因果机制。",
    "f9e6080f63": "滤波器输入不是要求采取动作。",
    "fa030a7842": "描述归一化方式不等于要求行动，当前实体和谓词不匹配。",
}


def main():
    root = Path.cwd().resolve()
    if (root / "evaluation/gas_path_holdout_v1_candidate/freeze.json").exists():
        raise ValueError("Never repair a frozen test run")
    before = root / ".runtime/experiments/holdout-v1-quality-20261014"
    final = root / ".runtime/experiments/holdout-v1-20261014"
    final.mkdir(parents=True, exist_ok=True)
    settings = RuntimeSettings.from_env(root / ".runtime/holdout-v1/profile")
    data = read(before / "snapshot.json")
    audit = read(before / "quality-audit.json")
    major_trans = [r for r in audit["translations"] if r["status"] == "major"]
    major_rel = [r for r in audit["entity_relations"] if r["status"] == "major"]
    all_trans = TRANSLATION_EDITS | TRANSLATION_RETAIN
    all_rel = RELATION_ACCEPT | RELATION_REPAIR | RELATION_REJECT
    if {r["id"][:10] for r in major_trans} != set(all_trans):
        raise ValueError("Every translation major needs an explicit adjudication")
    if {r["id"][:10] for r in major_rel} != set(all_rel):
        raise ValueError("Every relation major needs an explicit adjudication")
    # Capture all original source objects before mutation. No test questions are read.
    plan = {
        "before_snapshot_sha256": digest(data),
        "review_basis": "root AI source adjudication; no expert review",
        "gold_access": False,
        "translation_edits": TRANSLATION_EDITS,
        "translation_retain": TRANSLATION_RETAIN,
        "relation_accept": RELATION_ACCEPT,
        "relation_repair": RELATION_REPAIR,
        "relation_reject": RELATION_REJECT,
    }
    plan_path = before / "root-adjudication-plan.json"
    if plan_path.exists() and digest(read(plan_path)) != digest(plan):
        raise ValueError("Adjudication plan changed; preserve earlier version")
    save(plan_path, plan)
    backup = before / "before-repair.sqlite"
    if not backup.exists():
        with sqlite3.connect(settings.database_path) as source, sqlite3.connect(backup) as target:
            source.backup(target)
    vector = SQLiteVectorStore(settings.database_path)
    lexical = SQLiteLexicalStore(settings.database_path)
    changed_chunks = []
    for c in data["chunks"]:
        key = c["chunk_id"][:10]
        if key not in TRANSLATION_EDITS:
            continue
        value = c["metadata"]["display_content_zh"]
        for old, new in TRANSLATION_EDITS[key]:
            if old not in value:
                raise ValueError("Expected original translation fragment missing: " + key)
            value = value.replace(old, new)
        item = vector.get_chunk(c["chunk_id"])
        changed_chunks.append(
            replace(
                item,
                metadata={
                    **item.metadata,
                    "display_content_zh": value,
                    "quality_adjudication": {
                        "plan_sha256": digest(plan),
                        "reviewer": "codex:root",
                        "before_predictions": True,
                    },
                },
            )
        )
    vector.upsert(changed_chunks)
    lexical.upsert(changed_chunks)
    # Actual Neo4j records are retained; deprecated assertions cannot guide retrieval.
    from aerodiagnosis.adapters.persistence.neo4j_graph import Neo4jGraphStore

    graph = Neo4jGraphStore(
        uri=settings.neo4j_uri,
        user=settings.neo4j_user,
        password=settings.neo4j_password.get_secret_value(),
        database=settings.neo4j_database,
    )
    nodes = {n["node_id"]: GraphNode(**n) for n in data["nodes"]}
    edges = []
    changed_nodes = set()

    def node_change(edge, side, **kwargs):
        key = edge.source_id if side == "source" else edge.target_id
        old = nodes[key]
        props = {**old.properties, "quality_adjudication_plan_sha256": digest(plan)}
        if "name" in kwargs:
            props["name_zh"] = kwargs["name"]
        nodes[key] = replace(old, properties=props, **kwargs)
        changed_nodes.add(key)

    for raw in data["edges"]:
        edge = GraphEdge(**raw)
        key = edge.edge_id[:10]
        if key not in all_rel:
            continue
        props = {
            **edge.properties,
            "quality_review": {
                "reviewer": "codex:root",
                "expert": False,
                "before_predictions": True,
                "plan_sha256": digest(plan),
                "reason": all_rel[key],
            },
        }
        if key in RELATION_REJECT:
            props.update(deprecated=True, quality_review_status="rejected_source_semantics")
        else:
            props["quality_review_status"] = "source_adjudicated"
        relation = edge.relation
        if key == "001abdc3b1":
            node_change(edge, "source", name="发动机碎片损伤风险", kind="Failure")
        if key in {"1be92aa992", "cf7e06ea57"}:
            relation = "HAS_SYMPTOM"
        if key == "1c26a69638":
            node_change(edge, "source", name="燃油与衰减排气混合", kind="Mechanism")
        if key == "3389f070b4":
            node_change(edge, "source", name="涉事叶片")
            node_change(edge, "target", name="另五片叶片损坏")
        if key == "b723358d9e":
            node_change(edge, "source", kind="Action")
        if key in {"cf36e71103", "d50eeb418d"}:
            node_change(edge, "target", kind="Condition")
        if key == "e6dee63ea7":
            node_change(edge, "target", name="结构强度与飞行特性受影响")
        condition = {
            "001abdc3b1": "around the 3 rd- and 4 th-stage LPT wheels",
            "1c26a69638": "In the exhaust duct",
            "4451dfbd5e": "a maximum number of severe thermal transients",
            "b723358d9e": "Obstacle avoidance, emergency egress, and compensation for damage",
        }.get(key)
        if condition:
            quote = edge.properties["evidence_text"]
            if condition not in quote:
                raise ValueError("Condition not verbatim")
            props["qualifiers"] = {
                **props["qualifiers"],
                "conditions": [{"value": condition, "quote": condition}],
            }
        edges.append(replace(edge, relation=relation, properties=props))
    updated_nodes = [nodes[k] for k in sorted(changed_nodes)]
    graph.upsert_nodes(updated_nodes)
    graph.upsert_edges(edges)
    mirror = SQLiteGraphStore(settings.database_path)
    mirror.upsert_nodes(list(nodes.values()))
    corrected = {e.edge_id: e for e in edges}
    mirror.upsert_edges([corrected.get(e["edge_id"], GraphEdge(**e)) for e in data["edges"]])
    after, _ = snapshot(settings, final)
    record = {
        "snapshot_sha256": digest(after),
        "before_snapshot_sha256": digest(data),
        "original_audit_sha256": digest(audit),
        "adjudication_plan_sha256": digest(plan),
        "unresolved_major": 0,
        "translation_major_flags": len(major_trans),
        "translation_chunks_corrected": len(changed_chunks),
        "translation_flags_retained_with_reason": len(TRANSLATION_RETAIN),
        "relation_major_flags": len(major_rel),
        "graph_assertions_quarantined": len(RELATION_REJECT),
        "graph_assertions_repaired": len(RELATION_REPAIR),
        "graph_flags_rejected_as_false_positive": len(RELATION_ACCEPT),
        "source_limitations_retained": {
            k: TRANSLATION_RETAIN[k]
            for k in ["49aa0a831e", "49c36da3bf", "814775b0b6", "e29df304d2"]
        },
        "minor_flags": (
            "retained with AI-review records; meaning remains discernible; no expert validation"
        ),
        "source_only": True,
        "expert_reviewed": False,
        "test_predictions_started": False,
        "not_full_math_or_figure_reconstruction": True,
        "structural_prechecks": audit["structural_pass"],
    }
    save(final / "quality-adjudication.json", record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
