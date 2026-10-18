import importlib.util
from pathlib import Path

import pytest

from aerodiagnosis.ports import VectorChunk, VectorMatch

spec = importlib.util.spec_from_file_location(
    "frozen_candidate_cache",
    Path(__file__).resolve().parents[2] / "scripts/frozen_candidate_cache.py",
)
assert spec and spec.loader
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
FrozenCandidateCache = module.FrozenCandidateCache


class Source:
    backend_name = "immutable_source"

    def __init__(self):
        self.calls = []
        self.last_trace = {"visits": 17}

    def search(self, query, *, top_k, active_version_ids):
        self.calls.append((query, top_k, active_version_ids))
        return [
            VectorMatch(
                VectorChunk(str(i), "document", next(iter(active_version_ids)), "source", {}),
                1 / (i + 1),
            )
            for i in range(top_k)
        ]


def test_reuses_exact_prefix_without_increasing_each_policy_input(tmp_path):
    source = Source()
    store = FrozenCandidateCache(source, "D", tmp_path)
    first = store.search("问题", top_k=25, active_version_ids=frozenset({"v1"}))
    second = store.search("问题", top_k=75, active_version_ids=frozenset({"v1"}))
    assert first == second[:25]
    assert source.calls == [("问题", 150, frozenset({"v1"}))]
    assert store.last_trace == {"visits": 17}
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_cache_never_leaks_between_queries_or_active_versions(tmp_path):
    source = Source()
    store = FrozenCandidateCache(source, "P", tmp_path)
    store.search("问题", active_version_ids=frozenset({"v1"}))
    changed = store.search("问题", active_version_ids=frozenset({"v2"}))
    store.search("另一题", active_version_ids=frozenset({"v2"}))
    assert len(source.calls) == 3
    assert all(m.chunk.version_id == "v2" for m in changed)
    with pytest.raises(ValueError):
        store.search("问题", top_k=151, active_version_ids=frozenset({"v2"}))
