"""
诊断推理 Agent — 航空发动机气路故障专家推理

职责:
  1. 基于检索上下文进行故障诊断
  2. 生成因果关系推理链
  3. 构建故障树
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from utils.log import logger

PROMPT = """你是航空发动机气路故障诊断专家。根据以下上下文信息进行诊断推理。

返回 JSON:
{
  "thinking": "推理链：逐步分析现象→可能原因→排除逻辑→最终结论",
  "hypotheses": [
    {
      "cause": "根本原因",
      "path": "故障传播路径（如：FOD→HPC叶片损伤→效率下降→EGT上升）",
      "confidence": 0.85
    }
  ],
  "fault_tree": {
    "symptom": "顶层故障现象",
    "branches": [
      {
        "condition": "中间事件/条件",
        "children": ["原因1", "原因2"]
      }
    ]
  }
}

诊断要求:
1. 基于提供的上下文信息推理，不要凭空编造
2. 给出至少 1 个、最多 5 个诊断假设
3. confidence 是 0-1 之间的置信度
4. fault_tree 从顶事件逐级分解到底事件
5. 只返回 JSON，不要其他文字。"""


class DiagnosticReasoningAgent:
    """诊断推理 Agent — 故障树 + 因果链推理"""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.1,
        )

    async def reason(self, question: str, context: str) -> dict:
        """基于检索上下文执行诊断推理"""
        try:
            messages = [
                SystemMessage(content=PROMPT),
                HumanMessage(content=f"上下文信息:\n{context}\n\n故障现象: {question}"),
            ]
            resp = await self.llm.ainvoke(messages)
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, Exception) as e:
            logger.warning(f"Diagnostic reasoning parse failed: {e}")
            result = {
                "thinking": "推理模块解析失败，返回原始分析",
                "hypotheses": [
                    {"cause": "信息不足以确定根本原因", "path": f"用户描述了故障现象: {question[:200]}", "confidence": 0.4}
                ],
                "fault_tree": {"symptom": question[:100], "branches": []},
            }

        logger.info(f"Diagnostic: {len(result.get('hypotheses', []))} hypotheses")
        return result
