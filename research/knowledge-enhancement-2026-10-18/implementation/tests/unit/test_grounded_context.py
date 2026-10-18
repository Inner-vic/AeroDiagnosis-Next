import json
from types import SimpleNamespace

from aerodiagnosis.application.grounded_context import assemble_selected, pack_spans


class Tokenizer:
    def encode(self, value, **kwargs):
        return list(value)


def test_budget_includes_selector_source_metadata_and_wrapper():
    item = SimpleNamespace(
        evidence_id="evidence",
        excerpt="可能导致温度上升。原因未能确定。",
        locator={},
        document_metadata={"source_url": "https://source.example/report"},
    )
    bundle = pack_spans([item], Tokenizer(), budget=500)
    expected = len(json.dumps({"evidence": bundle["catalog"]}, ensure_ascii=False, sort_keys=True))
    assert expected == bundle["context_tokens"]
    assert expected <= 500
    too_small = pack_spans([item], Tokenizer(), budget=30)
    assert not too_small["catalog"]
    assert too_small["context_tokens"] <= 30


def test_compact_selectors_resolve_only_to_program_bound_exact_quotes():
    item = SimpleNamespace(
        evidence_id="evidence",
        excerpt="原因未能确定。",
        locator={},
        document_metadata={},
    )
    bundle = pack_spans([item], Tokenizer(), budget=200)
    reply = {
        "insufficient_evidence": False,
        "claims": [
            {"text": "起始原因未能确定。", "support_span_ids": ["s1"]},
        ],
    }
    answer = assemble_selected(reply, bundle)
    assert answer["gate_valid"]
    assert answer["claims"][0]["citations"][0]["support_quote"] == "原因未能确定。"
    reply["claims"][0]["support_span_ids"] = ["s999"]
    assert not assemble_selected(reply, bundle)["gate_valid"]


def test_malformed_model_output_is_a_recorded_failure_not_a_dropped_task():
    bundle = pack_spans([], Tokenizer(), budget=100)
    for value in [
        {"claims": None, "insufficient_evidence": False},
        {"claims": [], "insufficient_evidence": "true"},
    ]:
        result = assemble_selected(value, bundle)
        assert result["gate_valid"] is False
        assert result["gate_issues"] == [{"reason": "invalid_output_schema"}]
