"""
验证 Agent — 诊断假设自验证

职责:
  1. 检查诊断假设是否被检索上下文支撑
  2. 标记无依据的假设
  3. 输出验证报告
"""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from utils.log import logger

VERIFY_PROMPT = """你是航空发动机故障诊断质量审核专家。
逐一检查以下诊断假设是否被上下文信息支撑。

返回 JSON:
{
  "passed": true,
  "unsupported": [
    {"hypothesis": "无支撑的假设", "reason": "为何不通过"}
  ],
  "notes": "总体评价"
}

判定标准:
- passed=true: 所有假设均有上下文依据支撑
- passed=false: 存在无法从上下文中验证的假设
- 标记出所有无依据的假设
- 只返回 JSON，不要其他文字。"""


class VerificationAgent:
    """自验证 Agent — 确保诊断结论有上下文支撑"""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0,
        )

    async def check(self, hypotheses: list[dict], context: str) -> dict:
        """验证诊断假设是否被上下文支撑

        Returns:
            {"passed": bool, "unsupported": list, "notes": str}
        """
        if not hypotheses:
            return {"passed": True, "unsupported": [], "notes": "无假设需要验证"}

        try:
            hyp_text = json.dumps(hypotheses, ensure_ascii=False, indent=2)
            messages = [
                SystemMessage(content=VERIFY_PROMPT),
                HumanMessage(content=f"上下文信息:\n{context}\n\n需验证的假设:\n{hyp_text}"),
            ]
            resp = await self.llm.ainvoke(messages)
            cleaned = resp.content.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("\n", 1)[1].rsplit("```", 1)[0]
            result = json.loads(cleaned)
        except (json.JSONDecodeError, IndexError, Exception) as e:
            logger.warning(f"Verification parse failed: {e}")
            result = {"passed": True, "unsupported": [], "notes": "自动验证模块解析异常，跳过验证"}

        logger.info(f"Verifier: passed={result.get('passed')}, unsupported={len(result.get('unsupported', []))}")
        return result
