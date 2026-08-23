"""
报告生成 Agent — 诊断报告 + 来源标注

职责:
  1. Mode A: 有检索素材 → 基于上下文生成报告
  2. Mode B: 无检索素材 → 裸 LLM + 警告横幅
"""

from __future__ import annotations

from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from config import settings
from utils.log import logger

MODE_A_PROMPT = """你是航空发动机气路故障诊断专家。根据检索到的上下文信息，
为用户问题生成专业、准确的诊断报告。

要求:
1. 答案必须基于提供的上下文，不要编造
2. 如果存在诊断假设，按置信度排序呈现
3. 引用信息来源（如 [来源: xxx]）
4. 如涉及故障树，用简洁的文字呈现推理路径
5. 保持专业、准确、结构清晰"""

MODE_B_PROMPT = """你是航空发动机气路故障诊断专家。

⚠️ 警告: 知识库中未检索到与该问题直接相关的文档。
以下回答基于通用领域知识，可能不适用于具体机型或工况，
请以官方技术手册为准。

请根据你的专业知识回答用户问题，并在回答开头保留上述警告。"""


class ReportGenerationAgent:
    """报告生成 Agent — 双模式（有素材 / 无素材）"""

    def __init__(self) -> None:
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
            temperature=0.3,
        )

    async def generate(
        self,
        question: str,
        hypotheses: list[dict],
        verified: dict,
        context: list[dict],
        graph_context: list[dict],
    ) -> tuple[str, list[dict]]:
        """生成最终报告

        Returns:
            (final_answer, sources) 二元组
        """
        has_context = bool(context or graph_context)
        merged = self._merge_contexts(context, graph_context)

        if has_context and merged:
            answer = await self._mode_a(question, hypotheses, verified, merged)
        else:
            logger.warning("No retrieval context, falling back to Mode B (bare LLM)")
            answer = await self._mode_b(question)

        # 收集来源
        sources: list[dict] = []
        seen_sources: set[str] = set()
        for c in context[:10]:
            src = c.get("source", "unknown")
            if src not in seen_sources:
                seen_sources.add(src)
                sources.append({"source": src, "type": c.get("type", "vector"), "score": c.get("score", 0)})
        for g in graph_context[:5]:
            src = g.get("source", "knowledge_graph")
            if src not in seen_sources:
                seen_sources.add(src)
                sources.append({"source": src, "type": g.get("type", "graph"), "score": g.get("score", 0)})

        logger.info(f"Report generated, {len(sources)} sources cited")
        return answer, sources

    async def _mode_a(self, question: str, hypotheses: list[dict], verified: dict, context_str: str) -> str:
        """Mode A: 基于检索素材生成"""
        hyp_text = ""
        if hypotheses:
            sorted_hyps = sorted(hypotheses, key=lambda h: h.get("confidence", 0), reverse=True)
            hyp_lines = ["\n## 诊断假设 (按置信度排序)"]
            for i, h in enumerate(sorted_hyps):
                hyp_lines.append(f"{i + 1}. **{h.get('cause', '未知')}** (置信度: {h.get('confidence', 0):.0%})")
                if h.get("path"):
                    hyp_lines.append(f"   故障传播路径: {h['path']}")
            hyp_text = "\n".join(hyp_lines)

        if verified.get("unsupported"):
            hyp_text += "\n\n## 验证未通过的假设\n"
            for u in verified["unsupported"]:
                hyp_text += f"- ⚠️ {u.get('hypothesis', '')}: {u.get('reason', '')}\n"

        messages = [
            SystemMessage(content=MODE_A_PROMPT),
            HumanMessage(content=f"""检索上下文:
{context_str}

用户问题: {question}
{hyp_text}

请生成诊断报告:"""),
        ]
        resp = await self.llm.ainvoke(messages)
        return resp.content

    async def _mode_b(self, question: str) -> str:
        """Mode B: 无素材纯 LLM（带警告横幅）"""
        messages = [
            SystemMessage(content=MODE_B_PROMPT),
            HumanMessage(content=question),
        ]
        resp = await self.llm.ainvoke(messages)
        return resp.content

    @staticmethod
    def _merge_contexts(context: list[dict], graph_context: list[dict]) -> str:
        parts: list[str] = []
        for i, c in enumerate(context[:8]):
            parts.append(f"[来源{i + 1}: {c.get('source', '?')} | 类型:{c.get('type', '?')}]\n{c.get('content', '')}")
        for j, g in enumerate(graph_context[:5]):
            parts.append(f"[图谱来源{j + 1}]\n{g.get('content', '')}")
        return "\n\n---\n\n".join(parts)
