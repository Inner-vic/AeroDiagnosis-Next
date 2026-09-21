"""Idempotent built-in knowledge and cases for a useful first-run workspace."""

# ruff: noqa: RUF001

from __future__ import annotations

from aerodiagnosis.ingestion import DocumentIngestionService, DocumentManifest
from aerodiagnosis.ports import CaseRecord, CaseStore, GraphEdge, GraphNode, GraphStore

DEMO_DOCUMENT_ID = "demo-common-aero-knowledge"
DEMO_SOURCE = "航空发动机通用知识库 v1"

_DEMO_KNOWLEDGE = """# 航空发动机气路故障通用知识

> 本文档用于通用知识检索；涉及具体型号的检查与维修时，应以适用手册为准。

## 排气温度异常升高

排气温度（EGT）异常升高需要结合推力、转速、燃油流量、环境条件和历史趋势判断。可能方向包括压气机效率下降、热端效率变化、燃油喷嘴雾化异常以及温度测量链偏差。单个参数不能直接确定根因。

建议先核对工况与数据质量，再比较同工况趋势；随后检查相关告警、燃油系统和测量链。任何拆检或维修动作必须依据适用型号手册。

## 压气机喘振

喘振是压气机工作点越过稳定边界后出现的非稳定流动现象，可能伴随压力波动、异常声响、振动和推力变化。常见关联方向包括进气畸变、叶片污染或损伤、可调静子或放气系统异常，以及测量或控制链问题。

排查时应先确认事件工况与信号同步性，再沿进气条件、压气机气路、控制系统和传感链逐层缩小范围。

## 风扇振动异常

风扇振动升高可能与叶片积垢、侵蚀、外物损伤、不平衡或传感器链异常有关。应结合转速阶次、趋势变化、目视或孔探结果以及冗余传感器一致性判断。

## 燃油喷嘴堵塞

燃油喷嘴流量或雾化异常可能表现为温度分布不均、启动或加速品质变化以及燃烧稳定性下降。相似现象也可能来自点火、供油压力、温度传感器或热端状态，因此需要交叉证据。

## 知识增强诊断原则

诊断模型输出只是初步定位。知识增强 Agent 应把部件级结果扩展为有限候选根因，提出可验证的检查问题，
根据工程师反馈更新候选，再生成保留不确定性的分析和维修支持参考。
"""


def seed_demo_content(
    *,
    ingestion: DocumentIngestionService,
    manifest: DocumentManifest,
    graph_store: GraphStore,
    case_store: CaseStore,
) -> None:
    """Seed a small real backend dataset and keep its reserved document current."""

    del manifest  # The ingestion service owns manifest updates for this reserved document.
    result = ingestion.ingest(
        display_name="航空发动机气路故障通用知识.md",
        content=_DEMO_KNOWLEDGE.encode("utf-8"),
        document_id=DEMO_DOCUMENT_ID,
        force_vector_upsert=True,
    )
    version_id = result.version_id
    graph_store.delete_source(DEMO_SOURCE)

    nodes = (
        GraphNode(
            "demo-equipment-engine",
            "航空发动机",
            "Equipment",
            "用于气路故障知识演示的通用航空发动机。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-component-fan",
            "风扇",
            "Component",
            "发动机低压系统前端旋转部件。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-component-compressor",
            "压气机",
            "Component",
            "提高进入燃烧室的气流压力。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-component-fuel-nozzle",
            "燃油喷嘴",
            "Component",
            "负责燃油雾化与燃烧组织。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-symptom-egt",
            "EGT 异常升高",
            "Symptom",
            "排气温度高于同工况基线。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-symptom-surge",
            "压气机喘振",
            "Symptom",
            "压气机非稳定流动及压力波动。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-symptom-fan-vibration",
            "风扇振动异常",
            "Symptom",
            "风扇振动趋势或阶次异常。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-cause-compressor-fouling",
            "压气机叶片积垢",
            "Cause",
            "沉积改变叶片气动外形，降低效率与稳定裕度。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-cause-fuel-flow",
            "燃油喷嘴流量异常",
            "Cause",
            "喷嘴流量或雾化状态偏离，影响温度分布。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-solution-inspect",
            "趋势复核与孔探检查",
            "Solution",
            "先复核同工况趋势，再由具备资质人员按适用手册检查。",
            DEMO_SOURCE,
            version_id,
        ),
    )
    graph_store.upsert_nodes(nodes)
    graph_store.upsert_edges(
        (
            GraphEdge(
                "demo-edge-1",
                "demo-equipment-engine",
                "demo-component-fan",
                "HAS_COMPONENT",
                DEMO_SOURCE,
                version_id,
                1.0,
            ),
            GraphEdge(
                "demo-edge-2",
                "demo-equipment-engine",
                "demo-component-compressor",
                "HAS_COMPONENT",
                DEMO_SOURCE,
                version_id,
                1.0,
            ),
            GraphEdge(
                "demo-edge-3",
                "demo-equipment-engine",
                "demo-component-fuel-nozzle",
                "HAS_COMPONENT",
                DEMO_SOURCE,
                version_id,
                1.0,
            ),
            GraphEdge(
                "demo-edge-4",
                "demo-component-compressor",
                "demo-symptom-egt",
                "HAS_SYMPTOM",
                DEMO_SOURCE,
                version_id,
                0.92,
            ),
            GraphEdge(
                "demo-edge-5",
                "demo-component-compressor",
                "demo-symptom-surge",
                "HAS_SYMPTOM",
                DEMO_SOURCE,
                version_id,
                0.9,
            ),
            GraphEdge(
                "demo-edge-6",
                "demo-component-fan",
                "demo-symptom-fan-vibration",
                "HAS_SYMPTOM",
                DEMO_SOURCE,
                version_id,
                0.88,
            ),
            GraphEdge(
                "demo-edge-7",
                "demo-symptom-egt",
                "demo-cause-compressor-fouling",
                "CAUSED_BY",
                DEMO_SOURCE,
                version_id,
                0.82,
            ),
            GraphEdge(
                "demo-edge-8",
                "demo-symptom-egt",
                "demo-cause-fuel-flow",
                "CAUSED_BY",
                DEMO_SOURCE,
                version_id,
                0.76,
            ),
            GraphEdge(
                "demo-edge-9",
                "demo-symptom-surge",
                "demo-cause-compressor-fouling",
                "CAUSED_BY",
                DEMO_SOURCE,
                version_id,
                0.84,
            ),
            GraphEdge(
                "demo-edge-10",
                "demo-cause-compressor-fouling",
                "demo-solution-inspect",
                "SOLVED_BY",
                DEMO_SOURCE,
                version_id,
                0.9,
            ),
        )
    )

    for case in _demo_cases():
        case_store.upsert(case)


def _demo_cases() -> tuple[CaseRecord, ...]:
    return (
        CaseRecord(
            case_id="DEMO-CASE-EGT-001",
            version=1,
            summary=(
                "巡航同工况 EGT 趋势缓慢升高，复核传感器一致性后，"
                "将压气机效率退化列为重点检查方向。"
            ),
            attributes={
                "症状": "EGT 升高",
                "结论状态": "候选方向",
                "数据": "趋势数据",
            },
        ),
        CaseRecord(
            case_id="DEMO-CASE-SURGE-002",
            version=1,
            summary="加速阶段出现压力波动与异常声响，沿进气条件、可调机构和压气机气路逐层排查。",
            attributes={
                "症状": "喘振",
                "方法": "分层排查",
                "结果": "保留多候选",
            },
        ),
        CaseRecord(
            case_id="DEMO-CASE-FAN-003",
            version=1,
            summary="风扇振动升高，交叉核验测量链无异常后，通过目视检查发现叶片表面污染。",
            attributes={"部件": "风扇", "症状": "振动升高", "根因": "表面污染"},
        ),
        CaseRecord(
            case_id="DEMO-CASE-FUEL-004",
            version=1,
            summary="温度分布不均并伴随加速品质变化，燃油喷嘴只是候选之一，需要与点火、供油和温度测量证据交叉验证。",
            attributes={
                "系统": "燃油",
                "症状": "温度分布不均",
                "原则": "交叉证据",
            },
        ),
    )
