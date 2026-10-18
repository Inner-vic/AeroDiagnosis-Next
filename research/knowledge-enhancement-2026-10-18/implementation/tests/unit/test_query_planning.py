from __future__ import annotations

# ruff: noqa: RUF001 -- Verify preservation of Chinese punctuation, units and negation.
import hashlib

import pytest

from aerodiagnosis.domain import EvidenceItem
from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.evidence_budget import pack_evidence_audited
from aerodiagnosis.retrieval.planned_candidates import PlannedCandidateStore
from aerodiagnosis.retrieval.query_planning import (
    build_query_plan,
    candidate_quotas,
    protected_tokens,
)


def test_aliases_keep_original_values_models_negation_and_certainty():
    query = "CFM56-7B压缩机未失速，EGT可能达到900°C，N1是否低于40%？"
    plan = build_query_plan(query, decompose=True)
    assert plan.variants[0].text == query
    assert len(plan.variants) == 2
    expanded = plan.variants[1].text
    assert "压气机" in expanded and "排气温度（EGT）" in expanded
    assert "未失速" in expanded and "可能达到900°C" in expanded
    assert protected_tokens(expanded) == protected_tokens(query)
    assert all(v.kind != "subquestion" for v in plan.variants)


@pytest.mark.parametrize("query", ["EGT1参数异常", "CFM56-EGT传感器", "XEGT异常"])
def test_egt_expansion_does_not_replace_parts_of_identifiers(query):
    assert len(build_query_plan(query).variants) == 1


def test_split_requires_shared_context_and_complete_unqualified_questions():
    query = "CFM56-7B：压气机污染有什么表现？涡轮叶片如何检查？"
    plan = build_query_plan(query, terminology=False, decompose=True)
    assert len(plan.variants) == 3
    assert all(v.text.startswith("CFM56-7B：") for v in plan.variants)
    for unsafe in [
        "若温度升高：压气机如何检查？涡轮如何检查？",
        "CFM56：N1低于40%怎么办？N2低于20%怎么办？",
        "压气机如何检查？涡轮如何检查？",
        "CFM56：压气机如何检查？如何处理？",
        "CFM56：压气机如何检查？补充上下文",
    ]:
        assert len(build_query_plan(unsafe, terminology=False, decompose=True).variants) == 1


def test_surges_and_shutdowns_are_not_introduced_as_synonyms():
    plan = build_query_plan("压气机失速与发动机停车的关系是什么？")
    assert len(plan.variants) == 1
    assert "喘振" not in plan.variants[0].text and "熄火" not in plan.variants[0].text


@pytest.mark.parametrize("total", range(1, 51))
def test_subqueries_share_a_fixed_total_candidate_budget(total):
    for count in range(1, min(3, total) + 1):
        quotas = candidate_quotas(total, count)
        assert len(quotas) == count and min(quotas) > 0
        assert sum(quotas) == total
        assert quotas[0] >= max(quotas[1:], default=0)


class Store:
    backend_name = "test"
    embedding_identity = "test"

    def __init__(self):
        self.calls = []

    def search(self, query, *, top_k=5, active_version_ids=None):
        self.calls.append((query, top_k, active_version_ids))
        text = "排气温度异常" if "排气温度" in query else "无关正文"
        key = "target" if "排气温度" in query else "original"
        return [VectorMatch(VectorChunk(key, "d", "v", text), 0.8)]


def test_planned_store_recovers_term_match_without_increasing_route_quota():
    store = Store()
    audits = []
    planned = PlannedCandidateStore(store, audit_sink=audits.append)
    results = planned.search("EGT异常", top_k=50, active_version_ids=frozenset({"v"}))
    assert {r.chunk.chunk_id for r in results} == {"target", "original"}
    assert sum(c[1] for c in store.calls) == 50
    assert all(c[2] == frozenset({"v"}) for c in store.calls)
    assert audits[0]["total_requested_candidates"] == 50
    assert audits[0]["votes"]["target"][0]["kind"] == "terminology"


def test_empty_scope_never_calls_any_store():
    store = Store()
    assert PlannedCandidateStore(store).search("EGT", active_version_ids=frozenset()) == []
    assert not store.calls


class Tokenizer:
    def encode(self, value, **kwargs):
        return list(value)


def evidence(text, key):
    return EvidenceItem(
        evidence_id=hashlib.sha256(key.encode()).hexdigest(),
        source_kind="document_chunk",
        source_ref=key,
        document_id="d",
        version_id="v",
        content_hash=hashlib.sha256(text.encode()).hexdigest(),
        excerpt=text,
        locator={},
        score=0.8,
    )


def test_budget_audit_preserves_whole_uncertain_negative_evidence_and_logs_omission():
    a = evidence("起因未能确定。该现象可能发生于低温条件。", "a")
    b = evidence("过长正文。" * 200, "b")
    bundle = pack_evidence_audited([a, b], Tokenizer(), budget=400)
    assert bundle["packed"][0]["text"] == a.excerpt
    assert bundle["selection_audit"]["selected_ids"] == [a.evidence_id]
    assert bundle["selection_audit"]["omitted"][0]["evidence_id"] == b.evidence_id
    assert bundle["context_tokens"] <= 400
    assert not bundle["selection_audit"]["semantic_completeness_evaluated"]


def test_evidence_budget_rejects_truncated_sources_and_duplicate_ids():
    a = evidence("可能失速。", "a")
    with pytest.raises(ValueError, match="Duplicate"):
        pack_evidence_audited([a, a], Tokenizer())
    with pytest.raises(ValueError, match="Truncated"):
        pack_evidence_audited([a.model_copy(update={"excerpt_truncated": True})], Tokenizer())


def test_real_bm25_alias_recovery_obeys_snapshot_scope(tmp_path):
    from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore

    store = SQLiteLexicalStore(tmp_path / "knowledge.sqlite")
    store.upsert(
        [
            VectorChunk("good", "current", "v", "排气温度升高"),
            VectorChunk("inactive", "old", "old-v", "EGT EGT 排气温度升高"),
        ]
    )
    assert not store.search("EGT", active_version_ids=frozenset({"v"}))
    hits = PlannedCandidateStore(store).search("EGT", top_k=50, active_version_ids=frozenset({"v"}))
    assert [h.chunk.chunk_id for h in hits] == ["good"]
    assert store.get_chunks(["good"])["good"].content == "排气温度升高"


def test_conflicting_mirror_identity_across_variants_is_discarded():
    class Conflicting(Store):
        def search(self, query, **kwargs):
            text = "原文" if "排气温度" not in query else "伪造不同原文"
            return [VectorMatch(VectorChunk("same", "d", "v", text), 0.8)]

    audits = []
    hits = PlannedCandidateStore(Conflicting(), audit_sink=audits.append).search("EGT", top_k=50)
    assert not hits
    assert audits[0]["conflicts"] == ["same"]


def test_one_candidate_budget_preserves_only_original():
    store = Store()
    hits = PlannedCandidateStore(store, candidate_budget=1).search("EGT", top_k=50)
    assert hits and len(store.calls) == 1
    assert store.calls[0][0] == "EGT" and store.calls[0][1] == 1
