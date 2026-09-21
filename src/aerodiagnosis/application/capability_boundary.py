# ruff: noqa: RUF001

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class CapabilityBoundary:
    id: str
    status: str
    statement: str
    impact: str


_BOUNDARIES = (
    CapabilityBoundary(
        "monitoring_data_coverage",
        "open",
        "当前诊断以用户上传 CSV、文本知识和历史案例为主，尚未接入实时飞参或机队时序流。",
        "无法自动发现未上传工况中的瞬态趋势。",
    ),
    CapabilityBoundary(
        "causal_rule_coverage",
        "open",
        "故障因果规则和阈值是教学/候选级资产，未覆盖全部型号手册。",
        "结论不能替代适用型号手册或适航放行判断。",
    ),
    CapabilityBoundary(
        "llm_confidence",
        "open",
        "检索融合分只表示证据排序相关度，不是故障概率或维修置信度。",
        "需要工程师现场反馈和人工复核后才能进入案例库。",
    ),
    CapabilityBoundary(
        "graph_scale",
        "open",
        "当前因果图规模有限，适合单机演示和中小知识库。",
        "尚未完成大规模图查询和灾难恢复压测。",
    ),
)


def capability_boundaries() -> tuple[CapabilityBoundary, ...]:
    return _BOUNDARIES
