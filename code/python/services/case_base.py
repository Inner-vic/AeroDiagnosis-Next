"""
案例推理服务 — 历史诊断案例匹配 (Case-Based Reasoning) — v2.1

职责:
  1. 记录历史诊断案例 (症状 → 诊断 → 维修动作 → 结果)
  2. 相似度检索匹配历史案例
  3. 预置航空发动机常见故障案例作为演示数据
  4. 支持案例导出 / 回访
"""

from __future__ import annotations

import time
from typing import Any

from utils.log import logger

# ── 预置航空发动机故障诊断案例 ────────────────────────────────────
PRESET_CASES = [
    {
        "id": "case-preset-001",
        "title": "CFM56-7B EGT超限 — 燃油喷嘴积碳",
        "engine_model": "CFM56-7B",
        "symptoms": "起飞阶段EGT指示870°C(限制850°C), N1=96%, N2=98%, FF偏高5%",
        "diagnosis": "燃油喷嘴积碳导致雾化不良, 燃烧室局部富油, 出口温度分布不均, EGT超限",
        "fault_type": "燃油喷嘴积碳",
        "actions": "1. 孔探检查燃烧室 2. 拆检并超声波清洗燃油喷嘴 3. 试车验证EGT恢复",
        "result": "清洗后EGT下降至820°C, 恢复正常, 确认故障为燃油喷嘴积碳",
        "outcome": "confirmed",
        "severity": "中等",
        "tags": ["EGT超限", "燃油喷嘴", "CFM56", "孔探检查"],
        "created_at": "2025-03-15",
    },
    {
        "id": "case-preset-002",
        "title": "V2500 HPC喘振 — 可调静子叶片(VSV)故障",
        "engine_model": "V2500",
        "symptoms": "巡航阶段P3压力波动±8psi, 伴随放炮声, N2波动±3%, 振动值升高至1.8ips",
        "diagnosis": "VSV作动环卡滞导致部分静子叶片角度异常, 引发HPC失速喘振",
        "fault_type": "VSV作动机构故障",
        "actions": "1. 检查VSV作动机构连杆 2. 更换磨损的作动环衬套 3. 重新校准VSV角度 4. 地面试车验证",
        "result": "修理后P3稳定, 喘振消失, 故障排除",
        "outcome": "confirmed",
        "severity": "严重",
        "tags": ["喘振", "VSV", "HPC", "V2500", "压气机"],
        "created_at": "2025-04-02",
    },
    {
        "id": "case-preset-003",
        "title": "GE90 性能衰退 — HPC叶片积垢",
        "engine_model": "GE90",
        "symptoms": "相同N1=90%时FF增加6%, EGT上升15°C, EPR下降0.04, 性能趋势持续恶化超过3个月",
        "diagnosis": "HPC叶片积垢导致压气机效率下降, 发动机热效率降低, 需更高燃油流量维持相同推力",
        "fault_type": "HPC叶片积垢",
        "actions": "1. 执行发动机性能恢复水洗 2. 水洗后评估性能参数 3. 如未恢复则安排孔探检查",
        "result": "水洗后EGT恢复12°C, FF恢复5%, 剩余偏差在可接受范围",
        "outcome": "confirmed",
        "severity": "中等",
        "tags": ["性能衰退", "HPC积垢", "GE90", "水洗", "EGT"],
        "created_at": "2025-05-20",
    },
    {
        "id": "case-preset-004",
        "title": "Trent 700 燃烧室出口温度分布异常 — 旋流器损坏",
        "engine_model": "Trent 700",
        "symptoms": "T4四个热电偶读数差异超过80°C, 热点集中在一侧, 燃油喷嘴流量测试正常",
        "diagnosis": "单个旋流器叶片局部烧蚀破损, 导致对应区域油气混合不均匀, 形成局部热点",
        "fault_type": "燃烧室旋流器烧蚀",
        "actions": "1. 孔探确认旋流器状态 2. 更换损坏的旋流器组件 3. 检查相邻燃烧室衬套有无热损伤",
        "result": "更换后T4分布恢复均匀, 差异<15°C, 故障排除",
        "outcome": "confirmed",
        "severity": "严重",
        "tags": ["燃烧室", "温度分布", "旋流器", "Trent 700", "孔探"],
        "created_at": "2025-06-08",
    },
    {
        "id": "case-preset-005",
        "title": "CFM56-5B N1/N2不匹配 — 涡轮叶片FOD",
        "engine_model": "CFM56-5B",
        "symptoms": "N1=85%时N2仅92%(正常应为94%), 振动值VIB增至2.3ips, 滑油光谱分析含铁量升高",
        "diagnosis": "HPT叶片前缘受外物打伤(FOD), 叶片效率下降, 涡轮做功能力降低导致N2相对N1偏低",
        "fault_type": "HPT叶片FOD",
        "actions": "1. 立即安排孔探检查HPT叶片 2. 评估叶片损伤是否在可修理范围内 3. 滑油滤检查金属屑 4. 安排换发/换叶片",
        "result": "孔探确认2片HPT叶片前缘打伤, 超出修理极限, 安排换发, 旧发送车间修理",
        "outcome": "confirmed",
        "severity": "严重",
        "tags": ["FOD", "涡轮叶片", "N1/N2不匹配", "CFM56", "振动"],
        "created_at": "2025-07-14",
    },
    {
        "id": "case-preset-006",
        "title": "CFM56-7B 启动悬挂 — 燃油计量活门故障",
        "engine_model": "CFM56-7B",
        "symptoms": "启动过程中N2在35%停滞不上升, EGT缓慢上升至600°C, FF低于正常启动值, 无异常声响",
        "diagnosis": "燃油计量活门(FMV)卡滞在低流量位, 供油不足导致启动悬挂",
        "fault_type": "FMV卡滞",
        "actions": "1. 中止启动 2. 检查FMU/FMV系统 3. 更换燃油计量组件 4. 重新启动验证",
        "result": "更换FMV后启动正常, N2加速曲线恢复",
        "outcome": "confirmed",
        "severity": "中等",
        "tags": ["启动悬挂", "FMV", "CFM56", "燃油系统"],
        "created_at": "2025-08-01",
    },
    {
        "id": "case-preset-007",
        "title": "V2500 滑油消耗异常 — 碳封严失效",
        "engine_model": "V2500",
        "symptoms": "滑油消耗率0.8qt/h(正常<0.4qt/h), 尾喷管有轻微白烟, 滑油压力正常, 磁性堵头少量金属屑",
        "diagnosis": "后轴承腔碳封严磨损, 滑油渗漏进入排气段, 导致消耗率异常升高",
        "fault_type": "碳封严磨损",
        "actions": "1. 孔探检查排气段滑油痕迹 2. 滑油光谱分析确认磨损元素 3. 如加剧则安排更换封严",
        "result": "监控运行中, 消耗稳定在0.5qt/h, 计划下一个A检更换封严",
        "outcome": "partial",
        "severity": "中等",
        "tags": ["滑油消耗", "碳封严", "V2500", "轴承腔"],
        "created_at": "2025-08-22",
    },
    {
        "id": "case-preset-008",
        "title": "GE90 排气温度裕度下降 — 涡轮叶尖间隙增大",
        "engine_model": "GE90",
        "symptoms": "EGT裕度(EGTM)从65°C下降至42°C, 性能趋势分析显示逐月下降2°C, 无其他异常参数",
        "diagnosis": "HPT叶尖间隙随使用循环增加而增大, 叶尖泄漏增加导致涡轮效率逐渐下降",
        "fault_type": "HPT叶尖间隙增大",
        "actions": "1. 确认EGTM下降趋势 2. 评估发动机剩余在翼寿命 3. 如低于30°C裕度则安排性能恢复或换发",
        "result": "继续监控中, 预计还可运行2000循环",
        "outcome": "partial",
        "severity": "轻微",
        "tags": ["EGT裕度", "叶尖间隙", "GE90", "性能趋势"],
        "created_at": "2025-09-10",
    },
]


class CaseBase:
    """内存级案例库 — 含预置演示案例 + 动态增量案例"""

    def __init__(self, vector_store: Any = None) -> None:
        self._cases: list[dict] = []
        self._vector_store = vector_store
        self._case_counter = 0

        # 加载预置案例
        for c in PRESET_CASES:
            self._cases.append(dict(c))
        logger.info(f"CaseBase: loaded {len(PRESET_CASES)} preset cases")

    async def add_case(
        self,
        question: str,
        intent: str,
        hypotheses: list[dict],
        final_answer: str,
        sources: list[dict],
        outcome: str = "",
    ) -> dict:
        """记录一个诊断案例"""
        self._case_counter += 1
        # Extract top hypothesis info
        top_h = hypotheses[0] if hypotheses else {}
        # Extract key terms from question for tags
        tags = []
        for kw in ["EGT", "N1", "N2", "喘振", "振动", "超限", "性能衰退", "温度", "CFM56", "V2500", "GE90"]:
            if kw.lower() in question.lower():
                tags.append(kw)

        case = {
            "id": f"case-{self._case_counter:04d}",
            "title": question[:80],
            "engine_model": "",
            "symptoms": question,
            "diagnosis": top_h.get("cause", "") if top_h else "",
            "fault_type": top_h.get("cause", "") if top_h else "",
            "actions": top_h.get("path", "") if top_h else "",
            "result": final_answer[:300],
            "outcome": outcome or "pending",
            "severity": "待评估",
            "tags": tags[:5],
            "created_at": time.strftime("%Y-%m-%d"),
        }
        self._cases.append(case)
        if len(self._cases) > 550:
            # Keep presets + latest dynamic
            presets = [c for c in self._cases if c["id"].startswith("case-preset-")]
            dynamics = [c for c in self._cases if not c["id"].startswith("case-preset-")]
            self._cases = presets + dynamics[-500:]
        logger.info(f"CaseBase: recorded {case['id']} (total: {len(self._cases)})")
        return case

    async def list_cases(
        self, page: int = 1, page_size: int = 20, search: str = "", tag: str = ""
    ) -> dict:
        """分页查询案例列表"""
        filtered = list(self._cases)
        if search:
            q = search.lower()
            filtered = [
                c for c in filtered
                if q in c.get("title", "").lower()
                or q in c.get("symptoms", "").lower()
                or q in c.get("diagnosis", "").lower()
                or any(q in t.lower() for t in c.get("tags", []))
            ]
        if tag:
            filtered = [c for c in filtered if tag in c.get("tags", [])]

        # Sort: most recent first (pre-created have fake dates)
        filtered.sort(key=lambda c: c.get("created_at", ""), reverse=True)
        total = len(filtered)
        start = (page - 1) * page_size
        items = filtered[start:start + page_size]

        return {
            "items": items,
            "total": total,
            "page": page,
            "page_size": page_size,
            "total_pages": max(1, (total + page_size - 1) // page_size),
        }

    async def get_all_tags(self) -> list[str]:
        """获取所有标签"""
        tags: set[str] = set()
        for c in self._cases:
            for t in c.get("tags", []):
                tags.add(t)
        return sorted(tags)

    async def search_similar(self, question: str, top_k: int = 3) -> list[dict]:
        """搜索与当前问题最相似的历史案例 — 关键词重叠 + 标签匹配"""
        if not self._cases:
            return []

        q_words = set(question.lower().split())
        scored: list[tuple[dict, float]] = []

        for case in self._cases:
            c_text = (case.get("symptoms", "") + " " + case.get("diagnosis", "") + " "
                       + " ".join(case.get("tags", []))).lower()
            c_words = set(c_text.split())
            if not q_words:
                continue

            # 关键词重叠
            overlap = len(q_words & c_words) / max(len(q_words), 1)
            # 标签匹配加成
            tag_bonus = sum(
                0.15 for tag in case.get("tags", [])
                if tag.lower() in question.lower()
            )
            # Confirmed 案例加成
            outcome_bonus = 0.1 if case.get("outcome") == "confirmed" else 0
            score = overlap + tag_bonus + outcome_bonus

            if score > 0.12:
                scored.append((case, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [c for c, _ in scored[:top_k]]

    async def get_case(self, case_id: str) -> dict | None:
        for c in self._cases:
            if c["id"] == case_id:
                return c
        return None

    async def get_stats(self) -> dict:
        preset_count = sum(1 for c in self._cases if c["id"].startswith("case-preset-"))
        dynamic_count = len(self._cases) - preset_count
        return {
            "total_cases": len(self._cases),
            "preset_cases": preset_count,
            "dynamic_cases": dynamic_count,
            "latest_case": self._cases[-1]["id"] if self._cases else None,
            "tags": await self.get_all_tags(),
        }
