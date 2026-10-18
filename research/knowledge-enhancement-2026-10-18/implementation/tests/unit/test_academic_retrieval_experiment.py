from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("onnxruntime")
pytest.importorskip("tokenizers")
spec = importlib.util.spec_from_file_location(
    "academic_experiment",
    Path(__file__).resolve().parents[2] / "scripts/benchmark_academic_retrieval.py",
)
assert spec and spec.loader
exp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(exp)


def chunk(doc: str, start: int, end: int, tokens: int, **extra: Any) -> dict[str, Any]:
    return {
        "document_id": doc,
        "start": start,
        "end": end,
        "tokens": tokens,
        "canonical_sha256": "hash",
        **extra,
    }


def test_rrf_handles_disjoint_routes_and_rewards_agreement() -> None:
    assert exp.rrf([1, 2], [3, 2])[0] == 2
    assert exp.rrf([1], [3]) == [1, 3]


def test_bm25_does_not_return_zero_score_as_a_hit() -> None:
    bm25 = exp.BM25([{"text": "fuel ice"}, {"text": "fuel flow"}])
    assert exp.rank(bm25.scores("nothing"), 10, positive_only=True) == []
    assert exp.rank(bm25.scores("ice"), 10, positive_only=True) == [0]


def test_equal_context_budget_skips_oversized_and_deduplicates_parents() -> None:
    chunks = [chunk("A", 0, 10, 12), chunk("A", 10, 20, 4), chunk("B", 0, 8, 5)]
    hits, used = exp.context(chunks, [0, 1, 2], 10)
    assert used == 9 and [h["document_id"] for h in hits] == ["A", "B"]
    parent = {"start": 0, "end": 30, "tokens": 8}
    chunks = [chunk("A", 0, 10, 3, parent=parent), chunk("A", 10, 20, 3, parent=parent)]
    hits, used = exp.context(chunks, [0, 1], 10, parent=True)
    assert len(hits) == 1 and used == 8 and hits[0]["canonical_end"] == 30


def test_bootstrap_uses_fact_means_not_duplicated_query_rows() -> None:
    left = [
        {"sample_id": "q1", "fact_id": "f1", "statistical_cluster": "C1", "metric": 0.0},
        {"sample_id": "q2", "fact_id": "f1", "statistical_cluster": "C1", "metric": 0.0},
        {"sample_id": "q3", "fact_id": "f2", "statistical_cluster": "C2", "metric": 0.0},
    ]
    right = [{**r, "metric": 1.0 if r["sample_id"] == "q1" else 0.0} for r in left]
    effect = exp.paired_interval(left, right, "metric", repeats=100)
    assert effect["right_minus_left"] == 0.25
    assert effect["fact_count"] == 2 and effect["cluster_count"] == 2
    assert effect["ci95_family_component_percentile"] == [0, 0.5]


def test_bootstrap_rejects_mismatched_or_empty_pairs() -> None:
    with pytest.raises(ValueError):
        exp.paired_interval([], [], "metric")
    row = {"sample_id": "q1", "fact_id": "f1", "statistical_cluster": "C1", "metric": 0}
    with pytest.raises(ValueError):
        exp.paired_interval([row], [{**row, "fact_id": "wrong"}], "metric")
    with pytest.raises(ValueError):
        exp.paired_interval([row, row], [row], "metric")


@pytest.mark.parametrize("phase", ["dev", "pilot"])
def test_development_phase_never_encodes_test_or_translated_questions(phase: str) -> None:
    queries = [
        {"sample_id": "dev", "split": "dev", "query_variant_type": "native"},
        {"sample_id": "test", "split": "test", "query_variant_type": "native"},
        {"sample_id": "translation", "split": "dev", "query_variant_type": "translation"},
    ]
    assert [q["sample_id"] for q in exp.selected_queries(queries, phase)] == ["dev"]


def test_matrix_has_parent_expansion_as_an_independent_variable() -> None:
    configurations = exp.configs_for(exp.default_preset())
    assert len(configurations) == 12
    for c in configurations:
        if c["parent"]:
            assert c["cut"] == "hierarchical"
            assert {**c, "parent": False} in configurations
    assert len({exp.config_id(c) for c in configurations}) == 12


@pytest.mark.parametrize(
    "change",
    [{"candidate_k": True}, {"overlap_chars": 800}, {"parent_chars": 100}, {"context_tokens": 0}],
)
def test_presets_reject_invalid_or_ambiguous_budgets(change: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        exp.validate_presets([{**exp.default_preset(), **change}])


def test_exact_context_budget_counts_wrappers_and_joined_text() -> None:
    chunks = [chunk("A", 0, 3, 1, text="ice"), chunk("B", 0, 4, 1, text="fuel")]
    first_context = exp.render_context([(chunks[0], chunks[0])])
    hits, audit = exp.pack_context(chunks, [0, 1], len(first_context), token_counter=len)
    assert len(hits) == 1 and audit["over_budget_skips"] == 1
    assert audit["used_tokens"] == len(audit["rendered_context"]) == len(first_context)
    assert audit["rendered_context_sha256"] == exp.sha(first_context.encode())
    # The old sum-of-fragment budget would have falsely permitted both fragments.
    hits, audit = exp.pack_context(chunks, [0, 1], 2, token_counter=len)
    assert hits == [] and audit["used_tokens"] == 0


def test_parent_backfill_has_correct_anchor_and_does_not_merge_sources() -> None:
    parent = {"start": 0, "end": 10, "tokens": 2, "text": "ice fuel"}
    chunks = [
        chunk("A", 0, 3, 1, text="ice", parent=parent),
        chunk("A", 4, 8, 1, text="fuel", parent=parent),
        chunk("B", 0, 3, 1, text="ice", parent=parent),
    ]
    hits, audit = exp.pack_context(chunks, [0, 1, 2], 1000, parent=True, token_counter=len)
    assert [h["document_id"] for h in hits] == ["A", "B"]
    assert audit["duplicate_skips"] == 1
    assert hits[0]["canonical_end"] == 10 and hits[0]["anchor_end"] == 3
    assert hits[0]["expanded_to_parent"] is True


def dev_record(score: float = 1.0) -> dict[str, Any]:
    parameters = exp.configs_for(exp.default_preset())[0]
    return {
        "config": exp.config_id(parameters),
        "parameters": parameters,
        "split": "dev",
        "variant": "native",
        "answerable_fact_count": 1,
        "metrics": {"required_evidence_hit_at_5": score, "evidence_coverage_at_5": score},
    }


def selection_receipt(tmp_path: Path) -> tuple[Path, dict[str, Any]]:
    record = dev_record()
    scored = {
        "split": "dev",
        "query_variant": "native",
        "dataset_release_sha256": "dataset",
        "metrics": record["metrics"],
        "answerable_fact_count": 1,
    }
    data = json.dumps(scored).encode()
    (tmp_path / "dev.scores.json").write_bytes(data)
    record.update(scores_file="dev.scores.json", scores_sha256=exp.sha(data))
    receipt = {
        "schema": "dev-selection@1",
        "phase": "dev",
        "dataset_release_sha256": "dataset",
        "source_hashes": {"runner": "source"},
        "model": {"weights": "model"},
        "selected_config": record["parameters"],
        "development_records": [record],
    }
    path = tmp_path / "selection.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    summary = {"phase": "dev", "artifacts_sha256": {"selection.json": exp.sha(path.read_bytes())}}
    summary_data = json.dumps(summary).encode()
    (tmp_path / "summary.json").write_bytes(summary_data)
    state = {"status": "complete", "phase": "dev", "summary_sha256": exp.sha(summary_data)}
    (tmp_path / "run_state.json").write_text(json.dumps(state), encoding="utf-8")
    return path, receipt


def test_selection_rejects_test_feedback_and_edited_dev_scores(tmp_path: Path) -> None:
    path, receipt = selection_receipt(tmp_path)
    assert exp.verify_selection(path, "dataset", {"runner": "source"}, {"weights": "model"})
    (tmp_path / "dev.scores.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="changed after selection"):
        exp.verify_selection(path, "dataset", {"runner": "source"}, {"weights": "model"})
    path, receipt = selection_receipt(tmp_path)
    receipt["development_records"][0]["split"] = "test"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    summary = {"phase": "dev", "artifacts_sha256": {"selection.json": exp.sha(path.read_bytes())}}
    data = json.dumps(summary).encode()
    (tmp_path / "summary.json").write_bytes(data)
    (tmp_path / "run_state.json").write_text(
        json.dumps({"phase": "dev", "status": "complete", "summary_sha256": exp.sha(data)}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="only native development"):
        exp.verify_selection(path, "dataset", {"runner": "source"}, {"weights": "model"})


def test_test_phase_requires_selection_and_cannot_overwrite_old_outputs(tmp_path: Path) -> None:
    out = tmp_path / "old"
    out.mkdir()
    sentinel = out / "summary.json"
    sentinel.write_text("original", encoding="utf-8")
    with pytest.raises(ValueError, match="immutable"):
        exp.run(tmp_path, tmp_path, out)
    assert sentinel.read_text(encoding="utf-8") == "original"
    with pytest.raises(ValueError, match="dev selection receipt"):
        exp.run(tmp_path, tmp_path, tmp_path / "new", phase="test")
    assert not (tmp_path / "new").exists()


def test_bootstrap_rejects_a_fact_assigned_to_two_components() -> None:
    left = [
        {"sample_id": "a", "fact_id": "f", "statistical_cluster": "C1", "m": 0},
        {"sample_id": "b", "fact_id": "f", "statistical_cluster": "C2", "m": 0},
    ]
    with pytest.raises(ValueError, match="span statistical components"):
        exp.paired_interval(left, left, "m")


class FakeTokenizer:
    def __init__(self) -> None:
        self.texts: list[str] = []

    def encode_batch(self, texts: list[str]) -> list[Any]:
        self.texts.extend(texts)
        return [
            SimpleNamespace(ids=[1, 2, 0], attention_mask=[1, 1, 0], type_ids=[0, 0, 0])
            for _ in texts
        ]


class FakeSession:
    def __init__(self) -> None:
        self.calls = 0

    def get_inputs(self) -> list[Any]:
        return [SimpleNamespace(name=n) for n in ["input_ids", "attention_mask"]]

    def run(self, outputs: Any, inputs: Any) -> list[Any]:
        self.calls += 1
        batch = inputs["input_ids"].shape[0]
        values = np.zeros((batch, 3, 384), dtype=np.float32)
        values[:, 0, 0] = 1
        values[:, 1, 1] = 1
        values[:, 2, 0] = 100  # A padding state must not contaminate mean pooling.
        return [values]


def fake_encoder(tmp_path: Path) -> Any:
    model = exp.E5.__new__(exp.E5)
    model.cache = tmp_path
    model.model_hash, model.tokenizer_hash, model.config_hash = "weights", "tokenizer", "config"
    model.tokenizer, model.session = FakeTokenizer(), FakeSession()
    return model


def test_e5_masked_pooling_prefix_and_offline_cache_accounting(tmp_path: Path) -> None:
    model = fake_encoder(tmp_path)
    first = model.encode(["ice", "ice"], "query: ")
    assert model.tokenizer.texts == ["query: ice"]
    assert np.allclose(first[:, :2], np.sqrt(0.5))
    assert np.allclose(np.linalg.norm(first, axis=1), 1)
    assert model.last_encoding["unique_encoded_texts"] == 1
    assert model.last_encoding["cache_miss_rows"] == 2
    assert model.last_encoding["cache_state"] == "all_miss"
    assert np.array_equal(first, model.encode(["ice", "ice"], "query: "))
    assert model.session.calls == 1 and model.last_encoding["cache_hit_rows"] == 2
    model.encode(["ice"], "passage: ")
    assert model.session.calls == 2 and model.tokenizer.texts[-1] == "passage: ice"
    with pytest.raises(ValueError, match="explicit retrieval prefix"):
        model.encode(["ice"], "")


def test_embedding_cache_tampering_fails_closed(tmp_path: Path) -> None:
    model = fake_encoder(tmp_path)
    model.encode(["ice"], "query: ")
    array = next(tmp_path.glob("*.npy"))
    array.write_bytes(array.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="integrity mismatch"):
        model.encode(["ice"], "query: ")


def test_cache_identity_changes_when_config_or_weights_change(tmp_path: Path) -> None:
    model = fake_encoder(tmp_path)
    model.encode(["ice"], "query: ")
    model.config_hash = "new-config"
    model.encode(["ice"], "query: ")
    assert model.session.calls == 2


def test_failed_dev_run_cannot_authorize_test(tmp_path: Path) -> None:
    path, _ = selection_receipt(tmp_path)
    (tmp_path / "run_state.json").write_text(
        json.dumps({"status": "failed", "phase": "dev"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="completed development run"):
        exp.verify_selection(path, "dataset", {"runner": "source"}, {"weights": "model"})


class TinyOfflineModel:
    calls: ClassVar[list[tuple[list[str], str]]] = []

    def __init__(self, root: Path, cache: Path, threads: int) -> None:
        self.identity = {"fixture": "synthetic-unit-test-not-scientific-data"}
        self.context_tokenizer = SimpleNamespace(
            encode=lambda text: SimpleNamespace(ids=list(text))
        )
        self.last_encoding: dict[str, Any] = {}

    def token_count(self, text: str) -> int:
        return len(text)

    def encode(self, texts: list[str], prefix: str) -> Any:
        self.calls.append((texts, prefix))
        self.last_encoding = {"fixture": True}
        vectors = np.ones((len(texts), 384), dtype=np.float32)
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_dev_receipt_authorizes_only_frozen_test_matrix_without_answer_leakage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dataset = tmp_path / "fixture-dataset"
    dataset.mkdir()
    text = "Ice can restrict fuel. A sensor measures temperature."
    fingerprint = exp.sha(text.encode())
    evidence = [
        {
            "document_id": "D",
            "canonical_sha256": fingerprint,
            "canonical_start": 0,
            "canonical_end": 21,
        }
    ]
    queries = [
        {
            "sample_id": split,
            "query": query,
            "split": split,
            "query_variant_type": "native",
            "answerability": "answerable",
            "reference_answer": "SECRET_REFERENCE_NEVER_ENCODE",
            "evidence": evidence,
            "evidence_requirement": "all",
            "fact_id": split,
            "statistical_cluster": split,
            "task_type": "mechanism",
            "query_language": "en",
        }
        for split, query in [("dev", "ice fuel"), ("test", "sensor temperature")]
    ]
    exp.builder.write_jsonl(dataset / "queries.jsonl", queries)
    exp.builder.write_jsonl(dataset / "graph_claims.jsonl", [])
    exp.builder.write_json(
        dataset / "corpus_manifest.json",
        {
            "documents": [
                {
                    "document_id": "D",
                    "canonical_sha256": fingerprint,
                    "canonical_character_count": len(text),
                }
            ]
        },
    )
    exp.builder.write_json(
        dataset / "release.json",
        {
            "files": {
                name: exp.sha((dataset / name).read_bytes())
                for name in ["queries.jsonl", "graph_claims.jsonl", "corpus_manifest.json"]
            }
        },
    )
    monkeypatch.setattr(exp.builder, "validate_release", lambda *args: None)
    monkeypatch.setattr(
        exp.builder,
        "load_sources",
        lambda *args: {"D": {"text": text, "canonical_sha256": fingerprint}},
    )
    monkeypatch.setattr(exp, "source_hashes", lambda: {"fixture": "frozen"})
    monkeypatch.setattr(exp, "E5", TinyOfflineModel)
    TinyOfflineModel.calls = []
    dev = exp.run(dataset, tmp_path, tmp_path / "dev-output")
    assert len(dev["results"]) == 12 and dev["phase"] == "dev"
    assert TinyOfflineModel.calls[0] == (["ice fuel"], "query: ")
    assert not any("SECRET_REFERENCE" in t for texts, _ in TinyOfflineModel.calls for t in texts)
    assert not any(t == "sensor temperature" for texts, _ in TinyOfflineModel.calls for t in texts)
    test = exp.run(
        dataset,
        tmp_path,
        tmp_path / "test-output",
        phase="test",
        selection_file=tmp_path / "dev-output/selection.json",
    )
    assert len(test["results"]) == 12 and test["phase"] == "test"
    assert any(p == "query: " and t == ["sensor temperature"] for t, p in TinyOfflineModel.calls)
    assert (
        len([x for x in test["paired_differences"] if x.get("contrast") == "parent_expansion"]) == 3
    )
    assert any(
        x.get("contrast") == "primary_dev_selected_vs_fixed_bm25"
        for x in test["paired_differences"]
    )
    for file in (tmp_path / "test-output").glob("*.contexts.jsonl"):
        for audit in exp.scorer.rows(file):
            assert audit["used_tokens"] == len(audit["rendered_context"])
            assert audit["used_tokens"] <= audit["budget_tokens"]
            assert (
                audit["candidate_union"] <= audit["sparse_candidates"] + audit["dense_candidates"]
            )
