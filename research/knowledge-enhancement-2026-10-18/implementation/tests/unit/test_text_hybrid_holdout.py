from __future__ import annotations

from scripts.evaluate_text_hybrid_holdout import paired_cluster_delta


def test_paired_source_bootstrap_retains_cluster_dependence():
    rows = [{"cluster": "a", "new": True, "old": False} for _ in range(4)]
    rows += [{"cluster": "b", "new": False, "old": True} for _ in range(4)]
    result = paired_cluster_delta(rows, "new", "old", samples=1000)
    assert result["delta"] == 0
    assert result["source_cluster_95ci"] == [-1.0, 1.0]
    assert result["clusters"] == 2
    assert result["queries"] == 8
    assert result["wins"] == result["losses"] == 4


def test_zero_discordance_is_degenerate_not_equivalence():
    rows = [{"cluster": g, "new": True, "old": True} for g in ["a", "b", "c"]]
    result = paired_cluster_delta(rows, "new", "old", samples=100)
    assert result["delta"] == 0
    assert result["source_cluster_95ci"] == [0.0, 0.0]
    assert result["wins"] == result["losses"] == 0
