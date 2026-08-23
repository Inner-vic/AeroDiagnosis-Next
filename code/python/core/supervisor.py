"""
多 Agent 协同编排器 — Supervisor Agent (v2.2)

修复 (suggestion.md):
  1. 验证循环正确计数: for i in range(MAX) 天然自增 ✓ (v2.1 已修复)
  2. 达到上限后强制输出 report
  3. 添加 per-node 耗时追踪
  4. context 不再硬截断, 改为取前 N 条各自完整内容
"""

from __future__ import annotations

import time
from typing import Any
from utils.log import logger


class MultiAgentSupervisor:
    """多 Agent 编排器，串联诊断工作流"""

    MAX_ITERATIONS = 3  # 验证失败最多重试

    def __init__(self) -> None:
        self._query_agent: Any = None
        self._hybrid_engine: Any = None
        self._diagnostic_agent: Any = None
        self._verifier: Any = None
        self._report_agent: Any = None

    def build_graph(self, query_agent, hybrid_engine, diagnostic_agent, verifier, report_agent) -> None:
        self._query_agent = query_agent
        self._hybrid_engine = hybrid_engine
        self._diagnostic_agent = diagnostic_agent
        self._verifier = verifier
        self._report_agent = report_agent
        logger.info("Supervisor: graph built with hybrid engine")

    async def run(self, question: str) -> dict:
        timings: dict[str, float] = {}
        overall = time.perf_counter()

        # ── Node 1: 意图识别 ──
        t0 = time.perf_counter()
        analysis = await self._query_agent.analyze(question)
        timings["query_analyze"] = round((time.perf_counter() - t0) * 1000, 1)
        logger.info(f"[timing] query_analyze: {timings['query_analyze']}ms  intent={analysis.get('intent')}")

        context: list[dict] = []
        graph_context: list[dict] = []

        # ── Node 2: 混合检索 ──
        if analysis.get("needs_retrieval", True):
            t0 = time.perf_counter()
            vec, bm25, kg = await self._hybrid_engine.search_all(
                question, top_k=8, entities=analysis.get("entities", []),
            )
            context = await self._hybrid_engine.fuse_results(vec, bm25, kg, top_k=8)
            graph_context = kg
            timings["retrieval"] = round((time.perf_counter() - t0) * 1000, 1)
            logger.info(f"[timing] retrieval: {timings['retrieval']}ms  "
                        f"{len(vec)}V + {len(bm25)}B + {len(kg)}G -> {len(context)} fused")

        # ── Node 3 & 4: 诊断 + 验证 (最多重试 N 次) ──
        hypotheses: list[dict] = []
        verification: dict = {"passed": True, "unsupported": [], "notes": ""}
        context_str = self._merge_context(context, graph_context)

        if analysis.get("needs_diagnosis") and context_str:
            for i in range(self.MAX_ITERATIONS):
                t0 = time.perf_counter()
                diag = await self._diagnostic_agent.reason(question, context_str)
                hypotheses = diag.get("hypotheses", [])
                verification = await self._verifier.check(hypotheses, context_str)
                elapsed = round((time.perf_counter() - t0) * 1000, 1)
                timings[f"diag_verify_round_{i + 1}"] = elapsed

                if verification.get("passed", True):
                    logger.info(f"[timing] diagnosis+verify passed round {i + 1}: {elapsed}ms")
                    break
                logger.warning(
                    f"[timing] verification FAILED round {i + 1}/{self.MAX_ITERATIONS}: {elapsed}ms"
                )
            else:
                # 达到上限, 强制继续 (记录 warning)
                logger.warning(
                    f"Diagnosis verification exhausted all {self.MAX_ITERATIONS} retries, "
                    f"proceeding with best-effort hypotheses"
                )

        # ── Node 5: 报告生成 ──
        t0 = time.perf_counter()
        final_answer, sources = await self._report_agent.generate(
            question=question, hypotheses=hypotheses, verified=verification,
            context=context, graph_context=graph_context,
        )
        timings["report_generate"] = round((time.perf_counter() - t0) * 1000, 1)
        timings["total"] = round((time.perf_counter() - overall) * 1000, 1)
        logger.info(f"[timing] report: {timings['report_generate']}ms  total: {timings['total']}ms")

        return {
            "final_answer": final_answer,
            "intent": analysis.get("intent", "chat"),
            "entities": analysis.get("entities", []),
            "hypotheses": hypotheses,
            "verification": verification,
            "sources": sources,
            "timings": timings,
        }

    def _merge_context(self, context, graph_context) -> str:
        """合并上下文 — 每段完整保留, 取前 12 条"""
        parts = []
        for i, c in enumerate(context[:10]):
            parts.append(f"[C{i + 1}] {c.get('content', '')}")
        for j, g in enumerate(graph_context[:5]):
            parts.append(f"[KG{j + 1}] {g.get('content', '')}")
        return "\n\n".join(parts)


_supervisor: MultiAgentSupervisor | None = None


def get_supervisor() -> MultiAgentSupervisor:
    global _supervisor
    if _supervisor is None:
        _supervisor = MultiAgentSupervisor()
    return _supervisor
