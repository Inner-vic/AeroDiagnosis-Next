# ruff: noqa: RUF001 -- Chinese certainty regression examples.
from aerodiagnosis.application.grounded_answer import assemble_answer, spans_for_evidence


def test_quotes_are_built_from_offsets_without_model_rewriting():
    evidence = [{"id": "a", "text": "  疲劳裂纹导致断裂。\n环境很可能促成腐蚀。"}]
    spans = spans_for_evidence(evidence)
    for s in spans:
        assert evidence[0]["text"][s.start : s.end] == s.text
    result = assemble_answer(
        {
            "claims": [{"text": "疲劳裂纹导致断裂。", "support_span_ids": [spans[0].span_id]}],
            "insufficient_evidence": False,
        },
        spans,
    )
    assert (
        result["gate_valid"]
        and result["claims"][0]["citations"][0]["support_quote"] == "疲劳裂纹导致断裂。"
    )


def test_uncertainty_tag_or_unchecked_summary_cannot_rescue_stronger_main_claim():
    spans = spans_for_evidence([{"id": "a", "text": "环境很可能促成腐蚀。"}])
    bad = assemble_answer(
        {
            "answer": "可能有关系",
            "claims": [
                {
                    "text": "环境导致腐蚀。",
                    "support_span_ids": [spans[0].span_id],
                    "certainty": "hypothesis",
                }
            ],
            "insufficient_evidence": False,
        },
        spans,
    )
    assert not bad["gate_valid"] and bad["gate_issues"][0]["reason"] == "source_uncertainty_removed"
    good = assemble_answer(
        {
            "claims": [{"text": "环境可能促成腐蚀。", "support_span_ids": [spans[0].span_id]}],
            "insufficient_evidence": False,
        },
        spans,
    )
    assert good["gate_valid"] and good["answer"] == "环境可能促成腐蚀。"


def test_unknown_span_does_not_disappear_and_invented_fact_does_not_pass():
    spans = spans_for_evidence([{"id": "a", "text": "检查未能确定失效原因。"}])
    assert not assemble_answer(
        {
            "claims": [{"text": "可能是腐蚀导致失效。", "support_span_ids": [spans[0].span_id]}],
            "insufficient_evidence": False,
        },
        spans,
    )["gate_valid"]
    assert not assemble_answer(
        {
            "claims": [{"text": "事实", "support_span_ids": ["wrong"]}],
            "insufficient_evidence": False,
        },
        spans,
    )["gate_valid"]


def test_safe_abstention_requires_reason_and_month_is_not_uncertainty():
    spans = spans_for_evidence([{"id": "a", "text": "On 30 May the blade caused damage."}])
    assert assemble_answer(
        {
            "claims": [{"text": "叶片导致损坏。", "support_span_ids": [spans[0].span_id]}],
            "insufficient_evidence": False,
        },
        spans,
    )["gate_valid"]
    assert not assemble_answer({"claims": [], "insufficient_evidence": True}, spans)["gate_valid"]


def test_modal_word_does_not_hide_explicit_certainty_in_same_claim():
    from aerodiagnosis.application.grounded_answer import uncertainty_issues

    assert uncertainty_issues("可能有人认为，但裂纹必然导致失效。", ["裂纹可能导致失效。"])
    assert not uncertainty_issues("裂纹不一定导致失效。", ["裂纹可能导致失效。"])
    assert uncertainty_issues("原因已经确定。", ["起始原因不能确定。"])


def test_free_abstention_reason_cannot_insert_unverified_facts_in_main_answer():
    result = assemble_answer(
        {
            "insufficient_evidence": True,
            "claims": [],
            "abstention_reason": "证据不足，但是已经确定失效由腐蚀引起。",
        },
        (),
    )
    assert result["answer"] == "现有证据不足以完整回答该问题。"
    assert "腐蚀" not in result["answer"]
