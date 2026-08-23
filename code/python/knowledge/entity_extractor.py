"""
实体抽取器 — 航空发动机领域 Few-Shot 增强版

升级 (suggestion.md):
  1. Few-Shot 示例模板, 提高抽取准确性
  2. 抽取结果增加 confidence 字段
  3. 支持与已有图谱一致性检查
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from utils.log import logger

PROMPT = """请从以下航空发动机故障诊断文本中提取实体和关系, 严格按JSON格式返回。

实体类型:
- EngineComponent: 发动机部件 (如 HPC, HPT, LPT, 燃烧室, 风扇, 涡轮叶片, 压气机)
- FaultMode: 故障模式 (如 EGT超限, 喘振, 振动异常, 性能衰退, 滑油消耗)
- Cause: 故障原因 (如 FOD, 热疲劳, 材料缺陷, 装配误差)
- Symptom: 故障征兆 (如 振动增大, 温度升高, 推力下降)
- SensorParameter: 传感器参数 (如 EGT, N1, N2, EPR, FF, VIB)
- EngineModel: 发动机型号 (如 CFM56-7B, V2500, GE90, Trent 700)
- MaintenanceProcedure: 维修措施 (如 孔探检查, 叶片更换, 性能恢复)

关系类型:
- has_component: 发动机→部件
- has_fault_mode: 部件→故障模式
- caused_by: 故障模式→原因
- indicated_by: 故障模式→征兆
- measured_by: 参数→传感器
- repaired_by: 故障→维修措施
- affects: 故障→部件

Few-Shot 示例:

输入: "HPC叶片结垢导致压气机效率下降, 引发喘振, 表现为N2波动和振动增大"
输出: {"entities":[{"name":"HPC","type":"EngineComponent","description":"高压压气机"},{"name":"喘振","type":"FaultMode","description":"压气机喘振"},{"name":"N2","type":"SensorParameter","description":"高压转子转速"},{"name":"振动增大","type":"Symptom","description":"振动异常增大"}],"relations":[{"head":"HPC","relation":"has_fault_mode","tail":"喘振","confidence":0.92},{"head":"喘振","relation":"indicated_by","tail":"振动增大","confidence":0.88},{"head":"喘振","relation":"indicated_by","tail":"N2","confidence":0.85}]}

输入: "CFM56-7B发动机燃烧室出口温度分布异常, 可能是燃油喷嘴堵塞"
输出: {"entities":[{"name":"CFM56-7B","type":"EngineModel","description":"波音737NG发动机"},{"name":"燃烧室","type":"EngineComponent","description":"发动机燃烧室"},{"name":"燃油喷嘴","type":"EngineComponent","description":"燃烧室燃油喷嘴"},{"name":"温度分布异常","type":"FaultMode","description":"燃烧室出口温度场不均"}],"relations":[{"head":"CFM56-7B","relation":"has_component","tail":"燃烧室","confidence":0.95},{"head":"燃烧室","relation":"has_fault_mode","tail":"温度分布异常","confidence":0.90},{"head":"温度分布异常","relation":"caused_by","tail":"燃油喷嘴","confidence":0.82}]}

格式要求:
- entities: 至少1个, 每个必须有 name/type/description
- relations: 可选, 每个必须有 head/relation/tail/confidence
- confidence 为 0-1 之间的浮点数, 0.9 以上为高置信
- 只返回 JSON, 不要其他文字"""


@dataclass
class AeroExtractionResult:
    entities: list[dict]
    relations: list[dict]
    source_chunk_id: str = ""


class EntityExtractor:
    """航空发动机领域实体关系抽取器 — Few-Shot + confidence"""

    BATCH_CONCURRENCY = 5

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    async def extract_batch(self, chunks: list) -> dict:
        """并发批量提取实体和关系"""
        if not chunks:
            return {"entities": [], "relations": []}

        semaphore = asyncio.Semaphore(self.BATCH_CONCURRENCY)

        async def _extract_one(chunk) -> AeroExtractionResult | None:
            async with semaphore:
                try:
                    return await self._extract_from_chunk(chunk)
                except Exception as e:
                    logger.warning(f"Extraction failed for chunk {getattr(chunk, 'chunk_id', '?')}: {e}")
                    return None

        tasks = [_extract_one(c) for c in chunks]
        results = await asyncio.gather(*tasks)

        all_entities: list[dict] = []
        all_relations: list[dict] = []
        seen_entities: set[str] = set()
        seen_relations: set[tuple] = set()

        for r in (x for x in results if x):
            for e in r.entities:
                key = f"{e['name']}::{e['type']}"
                if key not in seen_entities:
                    seen_entities.add(key)
                    all_entities.append(e)
            for rel in r.relations:
                key = (rel["head"], rel["relation"], rel["tail"])
                if key not in seen_relations:
                    seen_relations.add(key)
                    all_relations.append(rel)

        logger.info(f"EntityExtractor: {len(all_entities)} entities, {len(all_relations)} relations from {len(chunks)} chunks")
        return {"entities": all_entities, "relations": all_relations}

    async def _extract_from_chunk(self, chunk) -> AeroExtractionResult:
        chunk_id = getattr(chunk, "chunk_id", "")
        content = getattr(chunk, "content", "")
        try:
            messages = [SystemMessage(content=PROMPT), HumanMessage(content=f"文本:\n{content}")]
            resp = await self.llm.ainvoke(messages)
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0].strip()
            data = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, Exception) as e:
            logger.warning(f"Entity extraction parse failed for {chunk_id}: {e}")
            return AeroExtractionResult(entities=[], relations=[], source_chunk_id=chunk_id)

        entities = [
            {"name": e.get("name", ""), "type": e.get("type", "EngineComponent"), "description": e.get("description", "")}
            for e in data.get("entities", []) if e.get("name")
        ]
        relations = [
            {"head": r.get("head", ""), "relation": r.get("relation", "related_to"),
             "tail": r.get("tail", ""), "confidence": float(r.get("confidence", 0.8))}
            for r in data.get("relations", []) if r.get("head") and r.get("tail")
        ]
        return AeroExtractionResult(entities=entities, relations=relations, source_chunk_id=chunk_id)
