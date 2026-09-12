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
    )
    version_id = result.version_id

    nodes = (
        GraphNode(
            "demo-symptom-egt",
            "EGT 异常升高",
            "symptom",
            "排气温度高于同工况基线。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-symptom-surge",
            "压气机喘振",
            "symptom",
            "压气机非稳定流动及压力波动。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-component-fan",
            "风扇",
            "component",
            "发动机低压系统前端旋转部件。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-component-compressor",
            "压气机",
            "component",
            "提高进入燃烧室的气流压力。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-fault-fouling",
            "叶片积垢",
            "fault_mode",
            "沉积改变叶片气动外形。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-fault-fuel",
            "燃油喷嘴流量异常",
            "fault_mode",
            "喷嘴流量或雾化状态偏离。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-cause-ingestion",
            "进气畸变",
            "cause",
            "入口流场不均匀使稳定裕度下降。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-cause-sensor",
            "测量链偏差",
            "cause",
            "传感器或采集通道产生偏差。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-action-trend",
            "同工况趋势复核",
            "action",
            "比较相同工况下的历史参数趋势。",
            DEMO_SOURCE,
            version_id,
        ),
        GraphNode(
            "demo-action-inspect",
            "目视或孔探检查",
            "action",
            "由具备资质人员按适用手册执行。",
            DEMO_SOURCE,
            version_id,
        ),
    )
    graph_store.upsert_nodes(nodes)
    graph_store.upsert_edges(
        (
            GraphEdge(
                "demo-edge-1",
                "demo-symptom-egt",
                "demo-component-compressor",
                "可能关联",
                DEMO_SOURCE,
                version_id,
                0.78,
            ),
            GraphEdge(
                "demo-edge-2",
                "demo-symptom-egt",
                "demo-fault-fuel",
                "可能关联",
                DEMO_SOURCE,
                version_id,
                0.72,
            ),
            GraphEdge(
                "demo-edge-3",
                "demo-symptom-egt",
                "demo-cause-sensor",
                "需要排除",
                DEMO_SOURCE,
                version_id,
                0.68,
            ),
            GraphEdge(
                "demo-edge-4",
                "demo-symptom-surge",
                "demo-component-compressor",
                "发生于",
                DEMO_SOURCE,
                version_id,
                0.9,
            ),
            GraphEdge(
                "demo-edge-5",
                "demo-cause-ingestion",
                "demo-symptom-surge",
                "可能诱发",
                DEMO_SOURCE,
                version_id,
                0.78,
            ),
            GraphEdge(
                "demo-edge-6",
                "demo-fault-fouling",
                "demo-component-fan",
                "影响",
                DEMO_SOURCE,
                version_id,
                0.75,
            ),
            GraphEdge(
                "demo-edge-7",
                "demo-fault-fouling",
                "demo-action-inspect",
                "建议检查",
                DEMO_SOURCE,
                version_id,
                0.82,
            ),
            GraphEdge(
                "demo-edge-8",
                "demo-symptom-egt",
                "demo-action-trend",
                "建议核验",
                DEMO_SOURCE,
                version_id,
                0.88,
            ),
            GraphEdge(
                "demo-edge-9",
                "demo-cause-sensor",
                "demo-action-trend",
                "可通过核验",
                DEMO_SOURCE,
                version_id,
                0.7,
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
