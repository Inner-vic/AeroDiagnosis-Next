"""Use the existing application task contracts through the shared budgeted client."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from aerodiagnosis.adapters.llm.openai_compatible import _TASK_INSTRUCTIONS
from aerodiagnosis.adapters.llm.research_client import ResearchCallError, ResearchClient
from aerodiagnosis.ports import LanguageModelError


class BudgetedLanguageModel:
    requires_source_spans = True
    requires_question_coverage = True

    def __init__(self, client: ResearchClient, *, stage: str = "generation_evaluation") -> None:
        self._client = client
        self._stage = stage

    @property
    def identity(self) -> str:
        return "budgeted-deepseek@3:" + self._client.model

    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if task not in _TASK_INSTRUCTIONS:
            raise LanguageModelError("Unsupported application task")
        try:
            return self._client.complete(
                task=task,
                instruction=_TASK_INSTRUCTIONS[task],
                payload=payload,
                stage=self._stage,
                max_tokens=2048,
            ).value
        except ResearchCallError as exc:
            raise LanguageModelError(str(exc)) from None
