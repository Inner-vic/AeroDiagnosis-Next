"""
查询理解 Agent — 意图识别 + 实体抽取

职责:
  1. 将用户自然语言问题解析为结构化意图
  2. 识别航空发动机领域实体
  3. 决定路由方向（检索 / 纯对话 / 诊断）
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from utils.log import logger

PROMPT = """You are an aero engine query analyzer. 分析用户问题并返回 JSON:
{
  "intent": "knowledge_qa|fault_diagnosis|procedure|parameter_analysis|maintenance_plan|chat",
  "entities": [
    {"name": "实体名称", "type": "EngineModel|EngineComponent|FaultMode|SensorParameter|MaintenanceProcedure"}
  ],
  "needs_retrieval": true,
  "needs_diagnosis": false
}

意图说明:
- knowledge_qa: 知识型问题
- fault_diagnosis: 故障诊断
- procedure: 操作流程
- parameter_analysis: 参数趋势分析或数据文件解读
- maintenance_plan: 排故计划生成
- chat: 闲聊或无法归类

实体类型:
- EngineModel: 发动机型号（CFM56-7B, V2500, GE90）
- EngineComponent: 发动机部件（HPC, HPT, LPT, 燃烧室）
- FaultMode: 故障模式（EGT超限, 喘振, 叶片损伤）
- SensorParameter: 传感器参数（EGT, N1, N2, EPR, FF, VIB）
- MaintenanceProcedure: 维修流程

Few-Shot 示例:
输入: "CFM56-7B的EGT超过850度是什么原因？"
输出: {"intent":"fault_diagnosis","entities":[{"name":"CFM56-7B","type":"EngineModel"},{"name":"EGT","type":"SensorParameter"},{"name":"EGT超限","type":"FaultMode"}],"needs_retrieval":true,"needs_diagnosis":true}

输入: "如何检查高压压气机叶片？"
输出: {"intent":"procedure","entities":[{"name":"HPC","type":"EngineComponent"}],"needs_retrieval":true,"needs_diagnosis":false}

输入: "你好"
输出: {"intent":"chat","entities":[],"needs_retrieval":false,"needs_diagnosis":false}

只返回 JSON，不要其他内容。"""


class QueryUnderstandingAgent:
    """查询理解 Agent — 意图分类 + 实体识别 + 路由决策"""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    async def analyze(self, question: str) -> dict:
        """分析用户问题，返回结构化意图"""
        try:
            messages = [
                SystemMessage(content=PROMPT),
                HumanMessage(content=question),
            ]
            resp = await self.llm.ainvoke(messages)
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, Exception) as e:
            logger.warning(f"Query analysis JSON parse failed: {e}, using fallback")
            result = {
                "intent": "chat",
                "entities": [],
                "needs_retrieval": True,
                "needs_diagnosis": False,
            }

        logger.info(f"QueryAnalyzer: intent={result.get('intent')}, entities={len(result.get('entities', []))}")
        return result
