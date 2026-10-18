"""Eligible BM25 statistics and local authoritative source-copy regression tests."""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
from aerodiagnosis.ports import VectorChunk


def chunk(key: str, text: str, version: str = "active") -> VectorChunk:
    return VectorChunk(key, "doc:" + key, version, text, {"source": key})


def test_ineligible_versions_do_not_change_scoped_bm25_ranking_or_scores(tmp_path: Path) -> None:
    store = SQLiteLexicalStore(tmp_path / "all.sqlite3")
    isolated = SQLiteLexicalStore(tmp_path / "isolated.sqlite3")
    active = [
        chunk("a", "alpha alpha alpha beta"),
        chunk("b", "beta beta beta gamma"),
        chunk("c", "unrelated filler terms"),
    ]
    store.upsert(active)
    isolated.upsert(active)
    expected = isolated.search("alpha beta", top_k=5)
    assert [hit.chunk.chunk_id for hit in expected] == ["a", "b"]
    retired = [
        chunk(f"retired-{i}", "alpha " * (i + 1), "failed" if i % 2 else "superseded")
        for i in range(100)
    ]
    store.upsert(retired)
    # This fixture proves the issue: full-index statistics change eligible document ordering.
    unscoped = [
        hit for hit in store.search("alpha beta", top_k=200) if hit.chunk.version_id == "active"
    ]
    assert [hit.chunk.chunk_id for hit in unscoped] != [hit.chunk.chunk_id for hit in expected]
    scoped = store.search("alpha beta", top_k=5, active_version_ids=frozenset({"active"}))
    assert [hit.chunk for hit in scoped] == [hit.chunk for hit in expected]
    assert [hit.score for hit in scoped] == pytest.approx([hit.score for hit in expected])
    assert store.count() == len(active) + len(retired)
    assert store.search("alpha", active_version_ids=frozenset()) == []
    assert store.search("alpha", active_version_ids=frozenset({"missing-version"})) == []


def test_parallel_scopes_have_independent_fts_statistics_and_no_persistent_projection(
    tmp_path: Path,
) -> None:
    store = SQLiteLexicalStore(tmp_path / "scope.sqlite3")
    store.upsert(
        [
            chunk("left", "target left", "left-version"),
            chunk("right", "target right", "right-version"),
        ]
    )

    def search(version: str) -> list[str]:
        return [
            hit.chunk.chunk_id
            for hit in store.search("target", active_version_ids=frozenset({version}))
        ]

    versions = ["left-version", "right-version"] * 10
    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(search, versions))
    assert results == [["left"] if version == "left-version" else ["right"] for version in versions]
    assert store.count() == 2
    assert {hit.chunk.chunk_id for hit in store.search("target")} == {"left", "right"}


def test_batch_source_copy_lookup_is_complete_deduplicated_and_preserves_metadata(
    tmp_path: Path,
) -> None:
    store = SQLiteLexicalStore(tmp_path / "source.sqlite3")
    originals = [chunk(f"id-{i}", f"source text {i}", "v1") for i in range(1100)]
    store.upsert(originals)
    selected = store.get_chunks([record.chunk_id for record in originals] + ["id-1", "missing"])
    assert selected == {record.chunk_id: record for record in originals}
    assert store.get_chunks([]) == {}
    store.upsert([chunk("id-1", "new local source", "v2")])
    assert store.get_chunks(["id-1"])["id-1"].version_id == "v2"
