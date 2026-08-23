"""
演示数据注入 — 一键填充航空发动机故障诊断演示数据

用法:
  POST /api/admin/seed  → 注入演示文档 + 实体 + 关系
  POST /api/admin/reset → 清除所有演示数据
"""

from __future__ import annotations

import os
import time
from typing import Any

from utils.log import logger

SAMPLE_DOC = """航空发动机气路故障诊断手册

1. 概述
航空发动机气路故障是指影响发动机进气道、压气机、燃烧室、涡轮和排气系统正常工作的各类异常状态。

2. 常见故障模式

2.1 排气温度(EGT)超限
   现象: EGT指示超过手册规定限制值
   可能原因: 燃油喷嘴积碳、涡轮叶片烧蚀、燃烧室出口温度分布异常
   处置: 检查燃油喷嘴、执行孔探检查涡轮叶片、校准EGT传感器
   关联参数: EGT, FF, T3

2.2 高压压气机(HPC)喘振
   现象: 压气机出口压力波动剧烈，伴随"放炮"声、推力下降
   可能原因: 进气畸变、HPC叶片结垢或损伤、可调静子叶片(VSV)故障
   处置: 清洗HPC、检查VSV作动机构、孔探检查HPC叶片
   关联参数: P3, N2, VIB, EGT

2.3 发动机性能衰退
   现象: 相同N1转速下推力下降，燃油消耗率上升
   可能原因: 叶片叶尖间隙增大、积垢、密封件磨损
   处置: 执行性能恢复水洗、更换密封件、评估叶片状态
   关联参数: EGT, FF, N1, N2, EPR

2.4 燃烧室出口温度分布异常
   现象: 多个热电偶读数差异超过允许范围
   可能原因: 燃油喷嘴堵塞/不均匀、燃烧室衬套裂纹、旋流器损坏
   处置: 检查燃油喷嘴雾化情况、孔探检查燃烧室
   关联参数: T3, T4, FF, P3

2.5 N1/N2转速不匹配
   现象: N1与N2转速关系偏离正常曲线
   可能原因: 涡轮叶片损伤、轴断裂或连接失效、轴承故障
   处置: 振动分析、滑油金属屑分析、孔探检查涡轮
   关联参数: N1, N2, VIB, EGT

3. 传感器参数说明
   - EGT: 排气温度 (Exhaust Gas Temperature), 单位°C
   - N1: 低压转子转速, 单位% RPM
   - N2: 高压转子转速, 单位% RPM
   - EPR: 发动机压力比 (Engine Pressure Ratio)
   - FF: 燃油流量 (Fuel Flow), 单位kg/h
   - VIB: 振动值, 单位mm/s
   - P3: 高压压气机出口压力
   - T3: 高压压气机出口温度
   - T4: 燃烧室出口温度

4. 发动机型号参考
   - CFM56-7B: 波音737NG系列
   - CFM56-5B: 空客A320系列
   - V2500: 空客A320系列
   - GE90: 波音777系列
   - Trent 700: 空客A330系列"""


async def seed_data(
    vector_store: Any = None,
    knowledge_graph: Any = None,
    hybrid_engine: Any = None,
) -> dict:
    """注入演示数据"""
    from agents.doc_parser_agent import DocParserAgent, DocType, DocumentChunk
    from knowledge.entity_extractor import EntityExtractor
    from agents.knowledge_extract_agent import Entity, Relation

    results = {}

    # 1. 手动分块注入向量库
    parser = DocParserAgent()
    chunks = parser._chunk_texts(
        [SAMPLE_DOC],
        doc_id="demo-manual-001",
        doc_type=DocType.TEXT,
        source="demo/航空发动机气路故障诊断手册.txt",
    )

    if vector_store:
        try:
            n = await vector_store.add_chunks(chunks)
            results["vectors"] = n
            logger.info(f"Seed: {n} vectors injected")
        except Exception as e:
            logger.exception(f"Seed vector store failed: {e}")
            results["vectors_error"] = str(e)

    # 2. 实体抽取 + 图谱写入
    if knowledge_graph:
        try:
            extractor = EntityExtractor()
            extraction = await extractor.extract_batch(chunks)
            e_count = 0
            r_count = 0
            for e in extraction.get("entities", []):
                entity = Entity(name=e["name"], type=e["type"], description=e.get("description", ""))
                await knowledge_graph.upsert_entity(entity, source="demo-seed")
                e_count += 1
            for r in extraction.get("relations", []):
                relation = Relation(
                    head=r["head"], relation=r["relation"], tail=r["tail"],
                    confidence=r.get("confidence", 0.9),
                )
                await knowledge_graph.add_relation(relation, source="demo-seed")
                r_count += 1
            results["entities"] = e_count
            results["relations"] = r_count
            logger.info(f"Seed: {e_count} entities, {r_count} relations")
        except Exception as e:
            logger.exception(f"Seed graph failed: {e}")
            results["graph_error"] = str(e)

    # 3. 刷新 BM25
    if hybrid_engine:
        try:
            await hybrid_engine.refresh_bm25()
            results["bm25"] = "refreshed"
        except Exception as e:
            logger.exception(f"Seed BM25 failed: {e}")
            results["bm25_error"] = str(e)

    results["status"] = "ok"
    return results


async def reset_data(
    vector_store: Any = None,
    knowledge_graph: Any = None,
) -> dict:
    """清除演示数据"""
    results = {}
    if knowledge_graph:
        try:
            deleted = await knowledge_graph.delete_by_source("demo-seed")
            results["graph_deleted"] = deleted
        except Exception as e:
            results["graph_error"] = str(e)
    return {"status": "ok", **results}
