"""Characterize the useful legacy control-flow behavior before LangGraph migration."""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
LEGACY_ROOT = PROJECT_ROOT / "code" / "python"
sys.path.insert(0, str(LEGACY_ROOT))

from core.supervisor import MultiAgentSupervisor  # noqa: E402


class QueryAgent:
    async def analyze(self, question: str) -> dict[str, Any]:
        return {
            "intent": "fault_diagnosis",
            "entities": [],
            "needs_retrieval": True,
            "needs_diagnosis": True,
        }


class HybridEngine:
    async def search_all(
        self, question: str, **kwargs: Any
    ) -> tuple[list[dict[str, Any]], list[Any], list[Any]]:
        return ([{"content": "EGT rise evidence", "source": "manual"}], [], [])

    async def fuse_results(
        self, vector: list[dict[str, Any]], *args: Any, **kwargs: Any
    ) -> list[dict[str, Any]]:
        return vector


class DiagnosticAgent:
    def __init__(self) -> None:
        self.calls = 0

    async def reason(self, question: str, context: str) -> dict[str, Any]:
        self.calls += 1
        return {"hypotheses": [{"cause": "compressor", "confidence": 0.8}]}


class Verifier:
    def __init__(self) -> None:
        self.calls = 0

    async def check(self, hypotheses: list[dict[str, Any]], context: str) -> dict[str, Any]:
        self.calls += 1
        return {"passed": self.calls >= 2, "unsupported": [], "notes": ""}


class ReportAgent:
    async def generate(self, **kwargs: Any) -> tuple[str, list[dict[str, Any]]]:
        return ("report", [{"source": "manual"}])


def test_legacy_supervisor_retries_and_still_generates_a_report() -> None:
    supervisor = MultiAgentSupervisor()
    diagnostic = DiagnosticAgent()
    verifier = Verifier()
    supervisor.build_graph(QueryAgent(), HybridEngine(), diagnostic, verifier, ReportAgent())

    result = asyncio.run(supervisor.run("EGT rise"))

    assert diagnostic.calls == 2
    assert verifier.calls == 2
    assert result["final_answer"] == "report"
    assert result["verification"]["passed"] is True
