from aerodiagnosis.ports import VectorChunk, VectorMatch
from aerodiagnosis.retrieval.fusion_runtime import ConfiguredFusionRetriever


class Store:
    backend_name = "fake"

    def __init__(self, chunks):
        self.chunks = chunks
        self.calls = []

    def search(self, query, *, top_k, active_version_ids):
        self.calls.append((query, top_k, active_version_ids))
        return [VectorMatch(c, 1) for c in self.chunks[:top_k]]


def test_frozen_runtime_enforces_per_channel_budgets_without_labels():
    chunk = VectorChunk("a", "d", "v", "source")
    stores = {name: Store([chunk]) for name in ["B", "D", "P"]}
    retriever = ConfiguredFusionRetriever(
        stores=stores,
        weights={"B": 1, "D": 0.25, "P": 0.5},
        budgets={"B": 100, "D": 25, "P": 25},
        load_chunks=lambda ids: {"a": chunk},
        token_count=lambda c: 1,
        mode="mmr90",
    )
    assert retriever.search("query") == []
    assert all(not store.calls for store in stores.values())
    result = retriever.search("query", active_version_ids=frozenset({"v"}))
    assert len(result) == 1
    assert {name: store.calls[0][1] for name, store in stores.items()} == {
        "B": 100,
        "D": 25,
        "P": 25,
    }
    assert retriever.last_trace["actual_candidates"] == 3
    assert result[0].chunk.metadata["fusion_retrieval"]["route_ranks"] == {"B": 1, "D": 1, "P": 1}


def test_runtime_rejects_forged_payload_and_inactive_source():
    valid = VectorChunk("a", "d", "v", "valid")
    forged = VectorChunk("a", "d", "v", "forged")
    inactive = VectorChunk("b", "d", "old", "inactive")
    retriever = ConfiguredFusionRetriever(
        stores={"B": Store([forged, inactive])},
        weights={"B": 1},
        budgets={"B": 150},
        load_chunks=lambda ids: {"a": valid, "b": inactive},
        token_count=lambda c: 1,
    )
    assert retriever.search("query", active_version_ids=frozenset({"v"})) == []
    assert retriever.last_trace["routes"][0]["returned"] == 2
    assert retriever.last_trace["routes"][0]["accepted"] == 0
