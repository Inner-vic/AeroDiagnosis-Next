# ruff: noqa: RUF001 -- Chinese research annotations and literal source rows.
"""Source-only, manually curated NONEXPERT upper-bound graph. Never reads questions."""

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path.cwd()
OUT = ROOT / "evaluation/retrieval_capability_v1_candidate"
# source paragraph, short Chinese entities, original identity surfaces, relation, certainty.
# Every assertion below is a disclosed AI/manual source annotation, NOT automatic extraction.
ROWS = {
    "v5-ao-2018-033": [
        (
            "7b3d039a9ff4",
            "衬套磨损",
            "bushing",
            "杠杆臂断裂",
            "lever arm",
            "CONTRIBUTES_TO",
            "established",
        ),
        ("f0e46ab58724", "杠杆臂断裂", "lever arm", "叶片疲劳", "fatigue", "CAUSES", "established"),
        ("d8c0c5e5438c", "叶片疲劳", "fatigue", "3号叶片失效", "number 3", "CAUSES", "established"),
        (
            "dd813743c823",
            "3号叶片失效",
            "number 3",
            "5号叶片损伤",
            "number 5",
            "CONTRIBUTES_TO",
            "likely",
        ),
        (
            "24a4750bcb82",
            "3号叶片失效",
            "blades",
            "下游损伤",
            "downstream",
            "CAUSES",
            "established",
        ),
        (
            "2247befec8aa",
            "杠杆臂断裂",
            "lever arm",
            "推荐检查",
            "inspection",
            "RECOMMENDS",
            "recommended_not_mandatory",
        ),
    ],
    "v5-ao-2011-062": [
        (
            "44e5264cd88e",
            "护罩互锁磨损",
            "wear",
            "叶片疲劳",
            "fatigue",
            "CONTRIBUTES_TO",
            "potential",
        ),
        (
            "a07cd7ab199f",
            "叶片疲劳",
            "fatigue",
            "发动机振动关车",
            "vibration",
            "CAUSES",
            "established",
        ),
        (
            "360036851f09",
            "护罩互锁磨损",
            "wear",
            "收紧翻修标准",
            "criteria",
            "MITIGATED_BY",
            "established",
        ),
        (
            "b8b7d82b53dc",
            "护罩互锁磨损",
            "interlocks",
            "孔探检查",
            "borescope",
            "CHECK_WITH",
            "recommended",
        ),
        (
            "01694772a941",
            "叶片疲劳",
            "fatigue",
            "机理未证实",
            "unable to verify",
            "HAS_LIMITATION",
            "unresolved",
        ),
    ],
    "v5-ao-2016-095": [
        (
            "d4b15afd6503",
            "齿轮共振",
            "resonance",
            "齿轮断裂漏油",
            "gear fracture",
            "CONTRIBUTES_TO",
            "potential",
        ),
        (
            "40b6b8ee3607",
            "齿轮断裂漏油",
            "failure",
            "已知通报模式",
            "72-0298",
            "CONSISTENT_WITH",
            "consistent",
        ),
        (
            "458c9f46352a",
            "已知通报模式",
            "72-0298",
            "GEnx适用范围",
            "GEnx",
            "APPLIES_TO",
            "established",
        ),
        (
            "65304d9ec35f",
            "已知通报模式",
            "72-0298",
            "改装时限",
            "CSN",
            "HAS_REQUIREMENT",
            "established",
        ),
        (
            "67dde6a35c65",
            "改装时限",
            "compliance",
            "事件改装计划",
            "31 December 2016",
            "HAS_PLAN",
            "established",
        ),
        (
            "4c35cee7942a",
            "油屑计数",
            "chip count",
            "油屑告警",
            "ENG OIL DMS R",
            "INDICATES",
            "established",
        ),
        (
            "13548adeb96b",
            "油屑计数",
            "chip count",
            "漏油关车",
            "shut down",
            "ASSOCIATED_WITH",
            "observed_sequence",
        ),
    ],
    "v5-ao-2018-046": [
        (
            "bd1568bd2f2b",
            "晶间氧化",
            "intergranular oxidation",
            "导向叶片脱落",
            "aerofoil",
            "CAUSES",
            "established",
        ),
        (
            "0231a541f699",
            "导向叶片脱落",
            "aerofoil",
            "下游涡轮损伤",
            "downstream",
            "CAUSES",
            "established",
        ),
        (
            "4de52d5b2dfb",
            "下游涡轮损伤",
            "case",
            "机匣包容性",
            "containment",
            "HAS_FINDING",
            "no_loss_of_containment",
        ),
        (
            "3b91cb6534ae",
            "导向叶片脱落",
            "NGV",
            "防磨垫片",
            "shim",
            "MITIGATED_BY",
            "initial_hypothesis",
        ),
        (
            "8c280668f17f",
            "防磨垫片",
            "shim",
            "氧化防护局限",
            "intergranular oxidation",
            "HAS_LIMITATION",
            "does_not_prevent",
        ),
        (
            "ba414439b638",
            "晶间氧化",
            "intergranular",
            "基体侵入",
            "base material",
            "HAS_MECHANISM",
            "established",
        ),
    ],
    "v5-ao-2023-007": [
        (
            "f10ebc4bd688",
            "径向传动轴断裂",
            "driveshaft",
            "燃油压力丧失",
            "fuel pump pressure",
            "CAUSES",
            "established",
        ),
        (
            "f10ebc4bd688",
            "燃油压力丧失",
            "fuel pump pressure",
            "非指令关车",
            "shutdown",
            "CAUSES",
            "established",
        ),
        (
            "34ac3f1eeede",
            "径向传动轴断裂",
            "driveshaft",
            "初始原因不明",
            "initiator",
            "HAS_LIMITATION",
            "unresolved",
        ),
        (
            "8514f2807669",
            "金属污染",
            "Metal contamination",
            "溯源监测更换",
            "monitor or replace",
            "REQUIRES_ACTION",
            "conditional",
        ),
        (
            "3c81a10b99fd",
            "非指令关车",
            "shutdown",
            "事件无预警",
            "no warning",
            "HAS_FINDING",
            "observed",
        ),
    ],
    "v5-ao-2024-001": [
        (
            "f28d964cb9e7",
            "非指令加速",
            "acceleration",
            "FCU疑似故障",
            "FCU",
            "POSSIBLY_CAUSED_BY",
            "plausible",
        ),
        (
            "62611551728c",
            "FCU疑似故障",
            "FCU",
            "内部故障无法确定",
            "could not be determined",
            "HAS_LIMITATION",
            "lost_in_transit",
        ),
        (
            "03c7ec364f67",
            "FCU疑似故障",
            "FCU",
            "先前台架调整",
            "adjusted",
            "HAS_HISTORY",
            "not_disassembled",
        ),
        (
            "a7bcd5286930",
            "非指令加速",
            "exceedance event",
            "参数超出量程",
            "range of sensor",
            "HAS_FINDING",
            "observed",
        ),
        (
            "37c2eae632d7",
            "参数超出量程",
            "range",
            "最大转速不明",
            "maximum Ng",
            "HAS_LIMITATION",
            "unresolved",
        ),
    ],
    "v5-ao-2024-057": [
        (
            "e62276996522",
            "最窄颈疲劳",
            "min-neck",
            "接触边缘裂纹",
            "edge-of-contact",
            "CAUSES",
            "established",
        ),
        (
            "d062ee4dce07",
            "接触边缘裂纹",
            "edge-of-contact",
            "七成截面扩展",
            "70%",
            "HAS_FINDING",
            "established",
        ),
        ("724a3412ab31", "七成截面扩展", "30%", "叶片脱落", "liberating", "CAUSES", "established"),
        (
            "11bc1782fa8a",
            "叶片脱落",
            "liberated blade",
            "相邻叶片过载",
            "overstress",
            "CAUSES",
            "previous_occurrences_mechanism",
        ),
        (
            "fa29114fa4b9",
            "叶片脱落",
            "liberated",
            "继发发动机损伤",
            "Damage",
            "CAUSES",
            "established",
        ),
        (
            "e7cab3f489ad",
            "最窄颈疲劳",
            "min-neck",
            "八成截面仍完整",
            "80%",
            "HAS_FINDING",
            "remaining_material_intact",
        ),
    ],
    "v5-a11c0079": [
        (
            "ea3ad39daf08",
            "金刚石污染",
            "diamond",
            "润滑退化",
            "grease",
            "CAUSES",
            "believed_origin",
        ),
        (
            "8d3d299db8d3",
            "润滑退化",
            "state",
            "轴承失效",
            "bearing",
            "CAUSES",
            "conditional_continued_operation",
        ),
        (
            "9769d89179ad",
            "轴承失效",
            "bearing",
            "调速器拆检",
            "PT governor",
            "FOUND_IN",
            "established",
        ),
        (
            "f1c725e78525",
            "调速器拆检",
            "PT governor",
            "无负荷超速",
            "109%",
            "HAS_FINDING",
            "observed",
        ),
        (
            "6ca47546d0ce",
            "无负荷超速",
            "100%",
            "更换调速器恢复",
            "overhauled PT governor",
            "MITIGATED_BY",
            "observed",
        ),
        (
            "031cff05689b",
            "燃油泵孔径",
            "orifice",
            "避免EOS熄火",
            "flame-out",
            "MITIGATES",
            "established",
        ),
    ],
    "v5-aaib-g-suez": [
        (
            "ddaef3e63daf",
            "二级叶片疲劳",
            "fatigue",
            "二级叶片断裂",
            "stage-two",
            "CAUSES",
            "indicated",
        ),
        (
            "5a9d6d8073ff",
            "二级叶片断裂",
            "stage-two",
            "下游损伤",
            "downstream",
            "CAUSES",
            "probable",
        ),
        (
            "58666f765d1a",
            "二级叶片疲劳",
            "crack",
            "起裂点无法确定",
            "initiation point",
            "HAS_LIMITATION",
            "secondary_damage",
        ),
        (
            "b10d2b85c152",
            "腐蚀环境",
            "corrosive",
            "腐蚀检查",
            "inspect",
            "REQUIRES_ACTION",
            "conditional",
        ),
        (
            "cdaefad4a83c",
            "腐蚀检查",
            "inspection",
            "维修引用歧义",
            "ambiguous",
            "HAS_LIMITATION",
            "established",
        ),
    ],
    "v5-aaib-g-omyt": [
        (
            "b051ac6f068e",
            "二型硫化腐蚀",
            "Sulphidation",
            "高压叶片断裂",
            "rupture",
            "CAUSES",
            "established",
        ),
        (
            "9304a49bf0f1",
            "高压叶片断裂",
            "detached blade",
            "涡轮碎屑卡阻",
            "debris",
            "CAUSES",
            "established",
        ),
        ("9304a49bf0f1", "涡轮碎屑卡阻", "debris", "转子卡死", "seizure", "CAUSES", "established"),
        (
            "e8a60827e09c",
            "高压叶片断裂",
            "blade loss",
            "喘振失衡损伤",
            "surge",
            "CAUSES",
            "established",
        ),
        (
            "5246e3a0751e",
            "二型硫化腐蚀",
            "sulphur",
            "硫来源不确定",
            "cannot be discounted",
            "HAS_LIMITATION",
            "volcanic_not_excluded",
        ),
        (
            "8a682a66e3ff",
            "二型硫化腐蚀",
            "Sulphidation",
            "高温含硫环境",
            "550",
            "APPLIES_TO",
            "conditional",
        ),
    ],
}


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    graph_path = OUT / "source-graph.json"
    if graph_path.exists():
        raise ValueError("Source graph already frozen; create an explicit revision")
    catalog = json.loads(
        (ROOT / ".runtime/holdout-v5/text-index/catalog.json").read_text(encoding="utf-8")
    )
    active = set(catalog["active_versions"])
    db = sqlite3.connect(
        "file:" + (ROOT / ".runtime/holdout-v5/text-index/chinese.sqlite").as_posix() + "?mode=ro",
        uri=True,
    )
    chunks = []
    for key, doc, version, content, meta in db.execute(
        "select chunk_id,document_id,version_id,content,metadata_json from vector_chunks"
    ):
        if version in active:
            chunks.append(
                dict(
                    chunk_id=key,
                    document_id=doc,
                    version_id=version,
                    content=content,
                    metadata=json.loads(meta),
                )
            )
    db.close()
    manifest = json.loads(
        (ROOT / "evaluation/text_hybrid_holdout_v5_candidate/source_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    texts = {}
    for source in manifest["sources"]:
        if "v5-" + source["document_id"] in ROWS:
            path = (
                ROOT / "evaluation/text_hybrid_holdout_v5_candidate" / source["saved_source_path"]
            )
            if digest(path) != source["saved_source_sha256"]:
                raise ValueError("Source hash mismatch")
            texts["v5-" + source["document_id"]] = path.read_text(encoding="utf-8")
    nodes, edges, source_rows = [], [], []
    by_prefix = {c["chunk_id"][:12]: c for c in chunks}
    for doc, assertions in ROWS.items():
        text = texts[doc]
        source_sha = hashlib.sha256(text.encode()).hexdigest()
        version = next(c["version_id"] for c in chunks if c["document_id"] == doc)
        source_key = doc + ":" + version
        source_rows.append(
            dict(
                source_key=source_key,
                document_id=doc,
                version_id=version,
                content=text,
                sha256=source_sha,
            )
        )
        for c in chunks:
            if c["document_id"] == doc:
                coordinates = c["metadata"]["locator"]["coordinates"]
                assert text[coordinates["start"] : coordinates["end"]] == c["content"]
                coordinates["canonical_sha256"] = source_sha
        for i, (prefix, label1, surface1, label2, surface2, relation, certainty) in enumerate(
            assertions
        ):
            c = by_prefix[prefix]
            assert c["document_id"] == doc
            start = c["metadata"]["locator"]["coordinates"]["start"]
            proof = dict(
                chunk_id=c["chunk_id"],
                document_id=doc,
                canonical_start=start,
                canonical_end=start + len(c["content"]),
                canonical_sha256=source_sha,
                evidence_text=c["content"],
            )
            endpoint_ids = []
            for j, (label, surface) in enumerate([(label1, surface1), (label2, surface2)]):
                # Role is manually reviewed source-event identity; no cross-event merge.
                # Stable role keys retain certainty on edges, never on identity links.
                identity = hashlib.sha256((version + label).encode()).hexdigest()
                node_id = doc + ":mention:" + str(i) + ":" + str(j)
                existing = next(
                    (n for n in nodes if n["properties"].get("reviewed_role_id") == identity), None
                )
                if existing is None:
                    assert surface.casefold() in c["content"].casefold(), (prefix, surface)
                canonical_proof = (
                    existing["properties"]["entity_link"]
                    if existing
                    else {
                        **proof,
                        "kind": "Concept",
                        "role": label,
                        "surface": surface,
                        "review_status": "source_verified_nonexpert",
                    }
                )
                nodes.append(
                    dict(
                        node_id=node_id,
                        name=label,
                        kind="Concept",
                        description="AI来源人工整理；非领域专家审核",
                        source_ref="document:" + doc,
                        version_id=version,
                        properties={
                            "reviewed_role_id": identity,
                            "entity_link": canonical_proof,
                            "mention_source": proof,
                        },
                    )
                )
                endpoint_ids.append(node_id)
            edges.append(
                dict(
                    edge_id=doc + ":assertion:" + str(i),
                    source_id=endpoint_ids[0],
                    target_id=endpoint_ids[1],
                    relation=relation,
                    source_ref="document:" + doc,
                    version_id=version,
                    confidence=1.0,
                    properties={
                        **proof,
                        "validation_status": "necessary_source_checks_passed",
                        "polarity": "positive",
                        "certainty": certainty,
                        "qualifiers": {
                            "scope": "this source event only",
                            "curation": "manual_ai_nonexpert_source_only",
                        },
                    },
                )
            )
    OUT.mkdir(exist_ok=True)
    for name, value in [
        (
            "source-graph.json",
            dict(schema="source-only-reviewed-upper-bound@1", nodes=nodes, edges=edges),
        ),
        ("corpus.json", dict(active_versions=sorted(active), chunks=chunks)),
        ("sources.json", source_rows),
    ]:
        (OUT / name).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    freeze = dict(
        created_at_utc=datetime.now(UTC).isoformat(),
        questions_authored=False,
        curation="manual_ai_nonexpert_upper_bound_not_automatic_graph",
        api_calls=0,
        documents=len(ROWS),
        assertions=len(edges),
        mentions=len(nodes),
        original_chunks=len(chunks),
        file_sha256={
            name: digest(OUT / name)
            for name in ["source-graph.json", "corpus.json", "sources.json"]
        },
        builder_sha256=digest(Path(__file__)),
    )
    (OUT / "source-graph-freeze.json").write_text(json.dumps(freeze, indent=2), encoding="utf-8")
    print(json.dumps(freeze, ensure_ascii=False))


if __name__ == "__main__":
    main()
