from dataclasses import replace

import pytest

from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.fusion_policy import FusedCandidate, fuse_ranked, select_fused


def chunk(key, text="same evidence", version="v", **meta):
    return VectorChunk(
        key,
        "d",
        version,
        text,
        {
            "locator": {
                "kind": "range",
                "coordinates": {"start": 0, "end": len(text), "section_path": ["scope"]},
            },
            **meta,
        },
    )


def test_weighted_fusion_contributions_and_tie_order():
    a, b = chunk("a"), chunk("b")
    rows = fuse_ranked(
        {"B": [VectorMatch(a, 10), VectorMatch(b, 1)], "G": [VectorMatch(b, 1)]}, {"B": 1, "G": 0.5}
    )
    assert [r.chunk.chunk_id for r in rows] == ["b", "a"]
    assert rows[0].route_ranks == {"B": 2, "G": 1}
    assert rows[0].score == pytest.approx((1 / 62 + 0.5 / 61) / (1.5 / 61))
    with pytest.raises(ValueError):
        fuse_ranked({"B": []}, {"B": float("nan")})
    with pytest.raises(ValueError):
        fuse_ranked(
            {"B": [VectorMatch(a, 1)], "G": [VectorMatch(replace(a, content="forged"), 1)]},
            {"B": 1, "G": 1},
        )


def test_source_safe_dedup_keeps_different_qualifiers_and_versions():
    a = chunk("a", qualifiers={"engine": "one"})
    b = chunk("b", qualifiers={"engine": "two"})
    c = chunk("c", version="other", qualifiers={"engine": "one"})
    d = chunk("d", qualifiers={"engine": "one"})
    rows = [FusedCandidate(x, 1 - i * 0.01, {"B": i + 1}) for i, x in enumerate([a, b, c, d])]
    result, trace = select_fused(rows, token_count=lambda c: 1)
    assert [r.chunk.chunk_id for r, _ in result] == ["a", "b", "c"]
    assert trace["source_duplicates_skipped"] == 1
    legacy, _ = select_fused(rows, token_count=lambda c: 1, mode="legacy_dedup")
    assert [r.chunk.chunk_id for r, _ in legacy] == ["a", "c"]
    off, _ = select_fused(rows, token_count=lambda c: 1, mode="no_source_dedup")
    assert len(off) == 4


def test_section_context_and_noncontained_tails_are_not_deleted():
    a = chunk("a", "engine pressure 30 psi")
    b = chunk("b", "engine pressure 50 psi")
    c = replace(
        a,
        chunk_id="c",
        metadata={
            "locator": {
                "kind": "range",
                "coordinates": {
                    "start": 0,
                    "end": len(a.content),
                    "section_path": ["other engine"],
                },
            }
        },
    )
    rows = [FusedCandidate(x, 1 - i * 0.01, {"B": i + 1}) for i, x in enumerate([a, b, c])]
    result, _ = select_fused(rows, token_count=lambda c: 1)
    assert len(result) == 3


def test_mmr_is_soft_ranking_and_stays_within_token_and_document_budget():
    a, b, c = chunk("a", "alpha"), chunk("b", "alpha repeated"), chunk("c", "other")
    rows = [FusedCandidate(x, s, {"B": 1}) for x, s in [(a, 1), (b, 0.99), (c, 0.98)]]
    terms = {"a": frozenset({"alpha"}), "b": frozenset({"alpha"}), "c": frozenset({"other"})}
    result, trace = select_fused(
        rows, mode="mmr90", token_count=lambda c: 2, terms=terms, token_budget=4
    )
    assert [r.chunk.chunk_id for r, _ in result] == ["a", "c"]
    assert trace["context_tokens"] == 4
    result, _ = select_fused(
        rows, mode="mmr90", token_count=lambda c: 1, terms=terms, max_per_document=1
    )
    assert len(result) == 1


def test_agreement_ranking_does_not_inject_unretrieved_chunks():
    a, b = chunk("a", "first"), chunk("b", "second")
    rows = [FusedCandidate(a, 1, {"B": 1}), FusedCandidate(b, 0.99, {"B": 2, "D": 1})]
    result, _ = select_fused(rows, mode="agreement", token_count=lambda c: 1)
    assert [r.chunk.chunk_id for r, _ in result] == ["b", "a"]
