"""Source-range complete-evidence metrics; no model, graph oracle, or answer grading."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence
from typing import Any


def evidence_metrics(
    *,
    sources: Mapping[str, str],
    support_units: Sequence[Mapping[str, Any]],
    acceptable_support_sets: Sequence[Sequence[str]],
    hits: Sequence[Mapping[str, Any]],
    answerable: bool = True,
    ks: tuple[int, ...] = (5, 10),
) -> dict[str, Any]:
    """Compute minimum sufficient support-set coverage from validated canonical source ranges.

    source_key includes both document and version. The scorer derives coverage;
    retrieval systems never receive or self-report reference support IDs.
    Unjudged alternative evidence is not a negative. No precision/nDCG is computed.
    """
    if not ks or any(type(k) is not int or k < 1 for k in ks):
        raise ValueError("Positive evidence budgets required")
    units = {u["unit_id"]: u for u in support_units}
    if len(units) != len(support_units):
        raise ValueError("Duplicate support unit IDs")
    required = {}
    for key, u in units.items():
        text = sources[u["source_key"]]
        lo, hi = u["start"], u["end"]
        if (
            type(lo) is not int
            or type(hi) is not int
            or not 0 <= lo < hi <= len(text)
            or text[lo:hi] != u["quote"]
            or hashlib.sha256(text.encode()).hexdigest() != u["source_sha256"]
        ):
            raise ValueError("Gold support does not match its canonical source")
        positions = {i for i in range(lo, hi) if not text[i].isspace()}
        if not positions:
            raise ValueError("Empty non-whitespace support")
        required[key] = positions
    for h in hits:
        if h["source_key"] not in sources:
            raise ValueError("Unknown document/version in retrieved evidence")
        text = sources[h["source_key"]]
        lo, hi = h["start"], h["end"]
        if (
            type(lo) is not int
            or type(hi) is not int
            or not 0 <= lo < hi <= len(text)
            or text[lo:hi] != h["content"]
        ):
            raise ValueError("Retrieved content/range mismatch")
    if not answerable:
        return {
            "answerable": False,
            "retrieval_evidence_metrics": None,
            "abstention_requires_answer_output": True,
        }
    bundles = [set(b) for b in acceptable_support_sets]
    if not bundles or any(not b or not b <= units.keys() for b in bundles):
        raise ValueError("Non-empty, source-verified sufficient support sets required")

    def coverage(k):
        covered = {source: set() for source in sources}
        for h in hits[:k]:
            covered[h["source_key"]].update(range(h["start"], h["end"]))
        full = {key for key, u in units.items() if required[key] <= covered[u["source_key"]]}
        return any(b <= full for b in bundles), max(len(b & full) / len(b) for b in bundles)

    metrics = {}
    for k in ks:
        complete, rate = coverage(k)
        metrics[str(k)] = {
            "complete_necessary_evidence": complete,
            "best_sufficient_set_unit_coverage": rate,
        }
    maximum = max(ks)
    first = next((i for i in range(1, min(len(hits), maximum) + 1) if coverage(i)[0]), None)
    return {
        "answerable": True,
        "retrieval_evidence_metrics": metrics,
        "first_complete_support_rank": first,
        "complete_support_reciprocal_rank": 1 / first if first else 0.0,
        "path_reasoning_correctness": None,
        "semantic_precision": None,
        "limitations": ("Evidence availability is not proof of reasoning; "
                        "unknown alternatives are not negatives"),
    }
