from __future__ import annotations

import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest

from aerodiagnosis.domain import DiagnosisCommand, RetrievalStrategy
from aerodiagnosis.tools.diagnostic import DiagnosticToolset


@pytest.fixture
def evaluation():
    path = Path(__file__).resolve().parents[2] / "scripts/evaluate_chinese_graph_ablation.py"
    spec = importlib.util.spec_from_file_location("chinese_graph_evaluation", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_text_ablation_has_no_graph_or_case_evidence(evaluation):
    class EmptyText:
        backend_name = "empty"
        embedding_identity = "unused"

        def search(self, query, **kwargs):
            return []

    tools = DiagnosticToolset(
        vector_store=EmptyText(),
        graph_store=evaluation.DisabledGraph(),
        case_store=evaluation.NoCases(),
    )
    result = tools.hybrid_search(
        command=DiagnosisCommand(session_id="test", question="压气机污染如何检查"),
        query="压气机 污染 检查",
        active_version_ids=frozenset({"active"}),
        strategy=RetrievalStrategy.WEIGHTED_RRF,
    )
    assert not result.hits
    assert all(route.candidate_count == 0 for route in result.routes)


def test_exact_quote_does_not_accept_unknown_or_malformed_citations(evaluation):
    packed = [{"id": "known", "text": "可能导致性能下降;不能据此确认故障。"}]
    claims = [
        {"evidence_ids": ["known"], "support_quote": "可能导致性能下降"},
        {"evidence_ids": ["known"], "support_quote": "一定导致性能下降"},
        {"evidence_ids": ["known", "invented"], "support_quote": "性能下降"},
        {"evidence_ids": "known", "support_quote": "性能下降"},
        {"evidence_ids": ["known"], "support_quote": ""},
        None,
    ]
    assert evaluation.citation_checks(claims, packed) == [True, False, False, False, False, False]
    with pytest.raises(ValueError, match="must be a list"):
        evaluation.citation_checks(None, packed)


def test_context_budget_counts_ids_and_prefers_chinese_without_truncating(evaluation):
    class CharacterTokenizer:
        def encode(self, text, **kwargs):
            return list(text)

    items = [
        SimpleNamespace(evidence_id="a", excerpt="English", locator={"display_text_zh": "中文"}),
        SimpleNamespace(evidence_id="b", excerpt="x" * 100, locator={}),
    ]
    packed, used = evaluation.pack_context(items, CharacterTokenizer(), budget=50)
    assert packed == [{"id": "a", "text": "中文"}]
    assert used <= 50
    packed, used = evaluation.pack_context(items, CharacterTokenizer(), budget=1)
    assert packed == []
    assert used == 0
