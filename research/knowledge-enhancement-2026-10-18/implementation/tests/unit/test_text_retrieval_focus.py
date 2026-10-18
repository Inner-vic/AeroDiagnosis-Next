from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "text_retrieval_focus",
    Path(__file__).resolve().parents[2] / "scripts/benchmark_text_retrieval_focus.py",
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def score(hits):
    fact = {
        "document_id": "A",
        "source_start": 0,
        "source_end": 5,
        "source_quote": "ab cd",
        "fact_key": "f",
    }
    return module.range_scores([fact], hits, {"A": "ab cd", "B": "ab cd"})


def test_duplicate_and_overlapping_hits_never_inflate_reference_coverage():
    hit = {"document_id": "A", "start": 0, "end": 5}
    result = score([hit, hit, {"document_id": "A", "start": 1, "end": 4}])
    assert result["all_reference_ranges_hit"]
    assert result["mean_reference_character_coverage"] == 1


def test_same_words_in_another_document_are_not_the_bound_reference():
    result = score([{"document_id": "B", "start": 0, "end": 5}])
    assert not result["all_reference_ranges_hit"]
    assert result["mean_reference_character_coverage"] == 0


def test_two_partial_children_can_cover_one_reference_without_whitespace_bias():
    assert (
        score([{"document_id": "A", "start": 0, "end": 2}])["mean_reference_character_coverage"]
        == 0.5
    )
    assert score(
        [{"document_id": "A", "start": 0, "end": 2}, {"document_id": "A", "start": 3, "end": 5}]
    )["all_reference_ranges_hit"]


def test_invalid_source_extent_fails_instead_of_clipping():
    with pytest.raises(ValueError, match="outside its source"):
        score([{"document_id": "A", "start": 0, "end": 6}])
