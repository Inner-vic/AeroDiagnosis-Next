"""Frozen source-evidence experiment with BM25, real E5 ONNX and RRF.

Research runner only. No production KB activation or answer-generation claim.
Install optional CPU dependencies from experiment requirements before using.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import io
import json
import math
import platform
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort  # type: ignore[import-untyped]
from numpy.typing import NDArray
from tokenizers import Tokenizer

from aerodiagnosis.adapters.persistence.sqlite_vector import HashingEmbedder
from aerodiagnosis.ingestion.parsers import PlainTextParser

ENCODING_VERSION = "e5-maskmean512@2"
RUNNER_VERSION = "academic-retrieval@2"


def helper(name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, Path(__file__).with_name(name + ".py"))
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = helper("build_academic_benchmark")
scorer = helper("score_evidence_benchmark")


def sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class E5:
    def __init__(self, root: Path, cache: Path, threads: int = 4) -> None:
        self.root, self.cache = root, cache
        self.cache.mkdir(parents=True, exist_ok=True)
        self.last_encoding: dict[str, Any] = {}
        self.tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.context_tokenizer = Tokenizer.from_file(str(root / "tokenizer.json"))
        self.context_tokenizer.no_truncation()
        self.context_tokenizer.no_padding()
        self.tokenizer.enable_truncation(max_length=512)
        # Load the documented pad ID rather than assume the model family.
        config = json.loads((root / "config.json").read_text(encoding="utf-8"))
        self.config_hash = sha((root / "config.json").read_bytes())
        if config["hidden_size"] != 384 or config["max_position_embeddings"] != 512:
            raise ValueError("This runner is pinned to the 384-dimensional, 512-token E5 model")
        pad_id = config["pad_token_id"]
        self.tokenizer.enable_padding(pad_id=pad_id, pad_token="<pad>")
        options = ort.SessionOptions()
        options.intra_op_num_threads = threads
        options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(root / "model_qint8_avx512_vnni.onnx"),
            sess_options=options,
            providers=["CPUExecutionProvider"],
        )
        self.model_hash = sha((root / "model_qint8_avx512_vnni.onnx").read_bytes())
        self.tokenizer_hash = sha((root / "tokenizer.json").read_bytes())
        self.identity = json.loads((root / "model-revision.json").read_text(encoding="utf-8"))
        self.identity.update(
            weights_sha256=self.model_hash,
            tokenizer_sha256=self.tokenizer_hash,
            config_sha256=self.config_hash,
            encoding_version=ENCODING_VERSION,
            pooling="attention-mask mean, L2 normalized",
            dimensions=384,
            runtime=ort.__version__,
            quantized=True,
            providers=self.session.get_providers(),
            threads=threads,
        )

    def token_count(self, text: str) -> int:
        return len(self.context_tokenizer.encode(text, add_special_tokens=False).ids)

    def encode(self, texts: list[str], prefix: str, batch_size: int = 16) -> NDArray[Any]:
        if prefix not in {"query: ", "passage: "} or batch_size < 1:
            raise ValueError("E5 requires an explicit retrieval prefix and a positive batch size")
        started = time.perf_counter()
        identity = {
            "weights": self.model_hash,
            "tokenizer": self.tokenizer_hash,
            "config": self.config_hash,
            "runtime": ort.__version__,
            "encoding_version": ENCODING_VERSION,
            "prefix": prefix,
        }
        keys = [sha(json.dumps({**identity, "text": t}, sort_keys=True).encode()) for t in texts]
        output = np.zeros((len(texts), 384), dtype=np.float32)
        missing: dict[str, list[int]] = defaultdict(list)
        cached = 0
        lookup_seconds = tokenize_seconds = inference_seconds = save_seconds = 0.0
        for i, key in enumerate(keys):
            tick = time.perf_counter()
            path, receipt_path = self.cache / (key + ".npy"), self.cache / (key + ".json")
            if path.exists():
                if not receipt_path.exists():
                    raise ValueError("Cached embedding lacks an integrity receipt")
                receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
                data = path.read_bytes()
                if receipt != {"cache_key": key, "npy_sha256": sha(data)}:
                    raise ValueError("Cached embedding integrity mismatch")
                vector = np.load(io.BytesIO(data), allow_pickle=False)
                if (
                    vector.shape != (384,)
                    or vector.dtype != np.float32
                    or not np.isfinite(vector).all()
                    or not np.isclose(np.linalg.norm(vector), 1.0, atol=1e-5)
                ):
                    raise ValueError("Invalid cached embedding")
                output[i] = vector
                cached += 1
            else:
                if receipt_path.exists():
                    raise ValueError("Cached embedding receipt has no array")
                missing[key].append(i)
            lookup_seconds += time.perf_counter() - tick
        ordered = sorted(missing, key=lambda k: len(texts[missing[k][0]]))
        for offset in range(0, len(ordered), batch_size):
            batch = ordered[offset : offset + batch_size]
            tick = time.perf_counter()
            encoded = self.tokenizer.encode_batch([prefix + texts[missing[k][0]] for k in batch])
            arrays = {
                "input_ids": np.asarray([e.ids for e in encoded], dtype=np.int64),
                "attention_mask": np.asarray([e.attention_mask for e in encoded], dtype=np.int64),
                "token_type_ids": np.asarray([e.type_ids for e in encoded], dtype=np.int64),
            }
            inputs = {i.name: arrays[i.name] for i in self.session.get_inputs()}
            tokenize_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            result = self.session.run(None, inputs)[0]
            if result.ndim != 3 or result.shape[-1] != 384:
                raise ValueError("Expected unpooled E5 token states [batch, sequence, 384]")
            mask = arrays["attention_mask"][..., None].astype(np.float32)
            vectors = (result * mask).sum(axis=1) / mask.sum(axis=1)
            vectors = vectors.astype(np.float32)
            norms = np.linalg.norm(vectors, axis=1, keepdims=True)
            if not np.isfinite(vectors).all() or np.any(norms <= 1e-12):
                raise ValueError("Nonfinite or zero E5 embedding")
            vectors /= norms
            inference_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            for key, vector in zip(batch, vectors, strict=True):
                buffer = io.BytesIO()
                np.save(buffer, vector, allow_pickle=False)
                data = buffer.getvalue()
                path, pending = self.cache / (key + ".npy"), self.cache / (key + ".pending")
                pending.write_bytes(data)
                pending.replace(path)
                builder.write_json(
                    self.cache / (key + ".json"), {"cache_key": key, "npy_sha256": sha(data)}
                )
                for i in missing[key]:
                    output[i] = vector
            save_seconds += time.perf_counter() - tick
            if offset == 0 or (offset // batch_size + 1) % 50 == 0:
                print(
                    f"embedding {prefix.strip()} "
                    f"{min(offset + batch_size, len(ordered))}/{len(ordered)}",
                    flush=True,
                )
        self.last_encoding = {
            "text_count": len(texts),
            "cache_hit_rows": cached,
            "cache_miss_rows": len(texts) - cached,
            "unique_encoded_texts": len(ordered),
            "cache_lookup_seconds": lookup_seconds,
            "tokenization_seconds": tokenize_seconds,
            "model_inference_seconds": inference_seconds,
            "cache_write_seconds": save_seconds,
            "total_seconds": time.perf_counter() - started,
            "cache_state": "all_hit"
            if cached == len(texts)
            else "all_miss"
            if cached == 0
            else "mixed",
            "scope": "batch encoding/cache wall time, not per-request online latency",
        }
        return output


class BM25:
    def __init__(self, chunks: list[dict[str, Any]]) -> None:
        self.n = len(chunks)
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        lengths = []
        for i, chunk in enumerate(chunks):
            counts = Counter(HashingEmbedder._tokens(chunk["text"]))
            lengths.append(sum(counts.values()))
            for term, count in counts.items():
                self.postings[term].append((i, count))
        mean = sum(lengths) / len(lengths)
        self.norm = 1.2 * (0.25 + 0.75 * np.asarray(lengths) / mean)

    def scores(self, query: str) -> NDArray[Any]:
        result = np.zeros(self.n, dtype=np.float64)
        for term in set(HashingEmbedder._tokens(query)):
            posting = self.postings.get(term, [])
            if not posting:
                continue
            idf = math.log(1 + (self.n - len(posting) + 0.5) / (len(posting) + 0.5))
            for i, freq in posting:
                result[i] += idf * freq * 2.2 / (freq + self.norm[i])
        return result


def rank(scores: NDArray[Any], top: int, *, positive_only: bool = False) -> list[int]:
    order = sorted(range(len(scores)), key=lambda i: (-float(scores[i]), i))
    return [i for i in order if not positive_only or scores[i] > 0][:top]


def rrf(left: list[int], right: list[int], k: int = 60) -> list[int]:
    scores: dict[int, float] = defaultdict(float)
    for route in (left, right):
        for position, i in enumerate(route, 1):
            scores[i] += 1 / (k + position)
    return sorted(scores, key=lambda i: (-scores[i], i))


def render_context(parts: list[tuple[dict[str, Any], dict[str, Any]]]) -> str:
    """Exact retrieval context including source wrappers, excluding generation instructions."""
    return "\n\n".join(
        f"[document={child['document_id']}; chars={selected['start']}:{selected['end']}]\n"
        + selected["text"]
        for child, selected in parts
    )


def pack_context(
    chunks: list[dict[str, Any]],
    order: list[int],
    budget: int,
    *,
    parent: bool = False,
    max_chunks: int = 10,
    token_counter: Any = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if budget < 1 or max_chunks < 1:
        raise ValueError("Context limits must be positive")
    hits, parts = [], []
    seen: set[tuple[str, int, int]] = set()
    used = duplicates = oversized = 0
    for i in order:
        c = chunks[i]
        selected = c["parent"] if parent and c.get("parent") else c
        key = (c["document_id"], selected["start"], selected["end"])
        if key in seen:
            duplicates += 1
            continue
        seen.add(key)
        proposed = [*parts, (c, selected)]
        size = (
            token_counter(render_context(proposed)) if token_counter else used + selected["tokens"]
        )
        if size > budget:
            oversized += 1
            continue
        used, parts = size, proposed
        hits.append(
            {
                "document_id": c["document_id"],
                "canonical_start": selected["start"],
                "canonical_end": selected["end"],
                "canonical_sha256": c["canonical_sha256"],
                "anchor_start": c["start"],
                "anchor_end": c["end"],
                "expanded_to_parent": bool(parent and c.get("parent")),
            }
        )
        if len(hits) == max_chunks:
            break
    rendered = render_context(parts) if token_counter else None
    return hits, {
        "used_tokens": used,
        "budget_tokens": budget,
        "selected_count": len(hits),
        "packing_input_candidates": len(order),
        "duplicate_skips": duplicates,
        "over_budget_skips": oversized,
        "rendered_context": rendered,
        "rendered_context_sha256": sha(rendered.encode()) if rendered else None,
        "counting_scope": "joined source wrappers and raw evidence; no query/prompt",
    }


def context(
    chunks: list[dict[str, Any]], order: list[int], budget: int, *, parent: bool = False
) -> tuple[list[dict[str, Any]], int]:
    # Legacy helper; the runner uses pack_context with exact joined-context token accounting.
    hits, audit = pack_context(chunks, order, budget, parent=parent)
    return hits, audit["used_tokens"]


def paired_interval(
    left: list[dict[str, Any]],
    right: list[dict[str, Any]],
    metric: str,
    seed: int = 20261013,
    repeats: int = 10000,
) -> dict[str, Any]:
    a, b = {r["sample_id"]: r for r in left}, {r["sample_id"]: r for r in right}
    if len(a) != len(left) or len(b) != len(right) or a.keys() != b.keys() or not a:
        raise ValueError("Paired query sets must agree and be unique/nonempty")
    facts: dict[str, list[float]] = defaultdict(list)
    fact_clusters: dict[str, str] = {}
    for sid, row in a.items():
        other = b[sid]
        if (
            row["fact_id"] != other["fact_id"]
            or row["statistical_cluster"] != other["statistical_cluster"]
        ):
            raise ValueError("Paired facts/clusters differ")
        facts[row["fact_id"]].append(other[metric] - row[metric])
        fact_clusters[row["fact_id"]] = row["statistical_cluster"]
    clusters: dict[str, list[float]] = defaultdict(list)
    for fact, values in facts.items():
        clusters[fact_clusters[fact]].append(sum(values) / len(values))
    if any(
        len({r["statistical_cluster"] for r in a.values() if r["fact_id"] == fact}) != 1
        for fact in facts
    ):
        raise ValueError("One fact cannot span statistical components")
    if repeats < 40:
        raise ValueError("Use at least 40 bootstrap replicates")
    groups = list(clusters.values())
    rng = random.Random(seed)
    values = []
    for _ in range(repeats):
        sampled = [rng.choice(groups) for _ in groups]
        values.append(sum(sum(g) for g in sampled) / sum(len(g) for g in sampled))
    ordered = sorted(values)
    return {
        "metric": metric,
        "right_minus_left": sum(sum(g) for g in groups) / len(facts),
        "ci95_family_component_percentile": [
            ordered[int(0.025 * repeats)],
            ordered[int(0.975 * repeats)],
        ],
        "fact_count": len(facts),
        "cluster_count": len(groups),
        "bootstrap_repeats": repeats,
        "seed": seed,
        "leave_one_component_out": [
            {
                "excluded": name,
                "remaining_facts": len(facts) - len(group),
                "right_minus_left": (
                    sum(sum(g) for n, g in clusters.items() if n != name)
                    / (len(facts) - len(group))
                ),
            }
            for name, group in clusters.items()
            if len(facts) > len(group)
        ],
        "interpretation": (
            "Descriptive paired effect with small-cluster uncertainty, no p-value claim"
        ),
    }


def source_hashes() -> dict[str, str]:
    import aerodiagnosis.adapters.persistence.sqlite_vector as lexical
    import aerodiagnosis.ingestion.parsers as parsers
    import aerodiagnosis.ingestion.structure as structure

    base = Path(__file__).resolve().parent
    paths = {
        "runner": Path(__file__),
        "builder": base / "build_academic_benchmark.py",
        "scorer": base / "score_evidence_benchmark.py",
        "parser": Path(parsers.__file__),
        "structure": Path(structure.__file__),
        "lexical_tokenizer": Path(lexical.__file__),
    }
    return {name: sha(path.read_bytes()) for name, path in paths.items()}


def default_preset() -> dict[str, Any]:
    return {
        "chunk_chars": 800,
        "overlap_chars": 100,
        "parent_chars": 3200,
        "candidate_k": 100,
        "rrf_k": 60,
        "context_tokens": 2048,
        "max_chunks": 10,
    }


def validate_presets(presets: list[dict[str, Any]]) -> None:
    if not 1 <= len(presets) <= 3:
        raise ValueError("Use 1-3 registered dev presets, not a Cartesian search")
    fingerprints = set()
    for p in presets:
        if (
            set(p) != set(default_preset())
            or any(type(v) is not int for v in p.values())
            or any(v <= 0 for k, v in p.items() if k != "overlap_chars")
        ):
            raise ValueError("Preset fields must match the declared positive-integer schema")
        if (
            p["overlap_chars"] < 0
            or p["overlap_chars"] >= p["chunk_chars"]
            or p["chunk_chars"] < 100
            or p["parent_chars"] < p["chunk_chars"]
            or p["candidate_k"] > 1000
            or p["max_chunks"] < 5
        ):
            raise ValueError("Invalid chunk/candidate/context preset")
        fp = builder.annotation_hash(p)
        if fp in fingerprints:
            raise ValueError("Duplicate presets")
        fingerprints.add(fp)


def configs_for(preset: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"cut": cut, "route": route, "parent": parent, **preset}
        for cut in ("fixed", "structure", "hierarchical")
        for parent in ((False, True) if cut == "hierarchical" else (False,))
        for route in ("bm25", "dense_e5", "hybrid_rrf")
    ]


def config_id(c: dict[str, Any]) -> str:
    return (
        f"{c['cut']}-{'parent' if c['parent'] else 'fine'}-{c['route']}-"
        + builder.annotation_hash(c)[:12]
    )


def selected_queries(queries: list[dict[str, Any]], phase: str) -> list[dict[str, Any]]:
    if phase not in {"dev", "test", "pilot"}:
        raise ValueError("Unknown experiment phase")
    split = "test" if phase == "test" else "dev"
    result = [
        q
        for q in queries
        if q["split"] == split and (phase == "test" or q["query_variant_type"] == "native")
    ]
    if not result or len({q["sample_id"] for q in result}) != len(result):
        raise ValueError("Selected queries must be nonempty and unique")
    return result


def choose_on_dev(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records or any(r["split"] != "dev" or r["variant"] != "native" for r in records):
        raise ValueError("Selection uses only native development records")
    if any(
        not r["answerable_fact_count"] or r["metrics"]["required_evidence_hit_at_5"] is None
        for r in records
    ):
        raise ValueError("Development selection requires answerable facts")
    return min(
        records,
        key=lambda r: (
            -r["metrics"]["required_evidence_hit_at_5"],
            -r["metrics"]["evidence_coverage_at_5"],
            r["parameters"]["candidate_k"] * (2 if r["parameters"]["route"] == "hybrid_rrf" else 1),
            r["config"],
        ),
    )


def verify_selection(
    path: Path, dataset_hash: str, code_hashes: dict[str, str], model_identity: dict[str, Any]
) -> dict[str, Any]:
    receipt = json.loads(path.read_text(encoding="utf-8"))
    state_path, summary_path = path.parent / "run_state.json", path.parent / "summary.json"
    if not state_path.exists() or not summary_path.exists():
        raise ValueError("Selection needs a completed development run")
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if (
        state.get("status") != "complete"
        or state.get("phase") != "dev"
        or state.get("summary_sha256") != sha(summary_path.read_bytes())
    ):
        raise ValueError("Selection needs an unchanged completed development run")
    if (
        receipt.get("schema") != "dev-selection@1"
        or receipt.get("phase") != "dev"
        or receipt.get("dataset_release_sha256") != dataset_hash
        or receipt.get("source_hashes") != code_hashes
        or receipt.get("model") != model_identity
    ):
        raise ValueError("Test requires a matching frozen development selection receipt")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("phase") != "dev" or summary.get("artifacts_sha256", {}).get(path.name) != sha(
        path.read_bytes()
    ):
        raise ValueError("Selection not bound to the completed dev summary")
    records = receipt["development_records"]
    for record in records:
        name = record["scores_file"]
        if Path(name).name != name:
            raise ValueError("Selection score paths must be local filenames")
        data = (path.parent / name).read_bytes()
        if sha(data) != record["scores_sha256"]:
            raise ValueError("Development scores changed after selection")
        scored = json.loads(data)
        if (
            scored["split"] != "dev"
            or scored["query_variant"] != "native"
            or scored["dataset_release_sha256"] != dataset_hash
            or scored["metrics"] != record["metrics"]
            or scored["answerable_fact_count"] != record["answerable_fact_count"]
        ):
            raise ValueError("Selection records disagree with frozen dev scores")
    chosen = choose_on_dev(records)
    if receipt["selected_config"] != chosen["parameters"]:
        raise ValueError("Selected configuration is not the recorded dev-only decision")
    return receipt


def run(
    dataset: Path,
    model_dir: Path,
    output: Path,
    threads: int = 4,
    *,
    phase: str = "dev",
    selection_file: Path | None = None,
    presets: list[dict[str, Any]] | None = None,
    cache_root: Path | None = None,
    ranking_repeats: int = 3,
) -> dict[str, Any]:
    if output.exists() and any(output.iterdir()):
        raise ValueError("Experiment outputs are immutable; use a new output directory")
    if phase == "test" and (selection_file is None or presets is not None):
        raise ValueError("Test needs a dev selection receipt and forbids new parameter presets")
    if phase not in {"dev", "test", "pilot"} or ranking_repeats < 2 or threads < 1:
        raise ValueError("Invalid phase, threads or ranking repeats")
    builder.validate_release(dataset, dataset / "protocol.json")
    sources = builder.load_sources(dataset / "corpus_manifest.json")
    queries = selected_queries(scorer.rows(dataset / "queries.jsonl"), phase)
    registered = presets if presets is not None else [default_preset()]
    validate_presets(registered)
    output.mkdir(parents=True, exist_ok=True)
    started = datetime.now(UTC).isoformat()
    builder.write_json(
        output / "run_state.json", {"status": "started", "phase": phase, "started_at": started}
    )
    try:
        return execute(
            dataset,
            model_dir,
            output,
            sources,
            queries,
            threads,
            phase,
            selection_file,
            registered,
            cache_root,
            ranking_repeats,
            started,
        )
    except Exception as exc:
        builder.write_json(
            output / "run_state.json",
            {
                "status": "failed",
                "phase": phase,
                "started_at": started,
                "finished_at": datetime.now(UTC).isoformat(),
                "failure_type": type(exc).__name__,
                "failure": str(exc),
            },
        )
        raise


def execute(
    dataset: Path,
    model_dir: Path,
    output: Path,
    sources: dict[str, Any],
    queries: list[dict[str, Any]],
    threads: int,
    phase: str,
    selection_file: Path | None,
    registered: list[dict[str, Any]],
    cache_root: Path | None,
    ranking_repeats: int,
    started: str,
) -> dict[str, Any]:
    init_start = time.perf_counter()
    model = E5(model_dir, cache_root or output / "embedding-cache", threads)
    initialization_seconds = time.perf_counter() - init_start
    code_hashes = source_hashes()
    dataset_hash = sha((dataset / "release.json").read_bytes())
    selection = (
        verify_selection(selection_file, dataset_hash, code_hashes, model.identity)
        if phase == "test" and selection_file
        else None
    )
    if selection:
        registered = [{k: selection["selected_config"][k] for k in default_preset()}]
    configs = [c for p in registered for c in configs_for(p)]
    if phase == "pilot":
        configs = [c for c in configs if c["cut"] == "fixed"][:3]
    policy = {
        "schema": RUNNER_VERSION,
        "phase": phase,
        "dataset_release_sha256": dataset_hash,
        "source_hashes": code_hashes,
        "model": model.identity,
        "configs": configs,
        "ranking_repeats": ranking_repeats,
        "selection_sha256": sha(selection_file.read_bytes()) if selection_file else None,
        "selection_rule": "dev native fact-mean hit@5, coverage@5; lower route budget; stable ID",
        "primary_comparison": {
            "left": config_id(
                {**selection["selected_config"], "cut": "fixed", "parent": False, "route": "bm25"}
            ),
            "right": config_id(selection["selected_config"]),
            "metric": "required_evidence_hit_at_5",
            "stratum": "test-native",
        }
        if selection
        else None,
        "candidate_budget": "per route; hybrid union can reach twice candidate_k",
        "context": "Exact joined source wrappers/raw text under E5 tokenizer; "
        "no generation query/system prompt; no clipping or gold-guided packing",
        "timing": "Model initialization, cache lookup, batch encoding, index preparation, "
        "first traversal and repeated traversal ranking/packing reported separately. "
        "No online end-to-end latency claim.",
        "source_scope": "Every frozen manifest source, no evidence-informed source filtering",
        "dependencies": {
            n: importlib.metadata.version(n) for n in ("numpy", "onnxruntime", "tokenizers")
        },
        "python": sys.version,
        "platform": platform.platform(),
        "selection": "Test parameters must match a completed dev-only receipt; "
        "test observations must never be used for selection",
    }
    builder.write_json(output / "policy.json", policy)
    query_vectors = model.encode([q["query"] for q in queries], "query: ")
    query_encoding = dict(model.last_encoding)
    results, internals, indices = [], {}, []
    grouped: dict[tuple[int, int, int, str], list[dict[str, Any]]] = defaultdict(list)
    for c in configs:
        grouped[(c["chunk_chars"], c["overlap_chars"], c["parent_chars"], c["cut"])].append(c)
    for (size, overlap, parent_size, cut), configurations in grouped.items():
        parser = PlainTextParser(
            strategy=cut,
            chunk_size=size,
            overlap=overlap,
            parent_budget=parent_size,
            child_budget=size,
        )
        chunks, parents = [], {}
        parse_start = time.perf_counter()
        for docid, source in sources.items():
            text = source["text"]
            parsed = parser.parse(text.encode())
            if sha(parsed.canonical_content) != source["canonical_sha256"]:
                raise ValueError("Parser changed frozen canonical text")
            for child in parsed.chunks:
                loc = child.locator.coordinates
                start, end = loc["start"], loc["end"]
                if text[start:end] != child.content:
                    raise ValueError("Child source interval mismatch")
                record = {
                    "document_id": docid,
                    "start": start,
                    "end": end,
                    "text": child.content,
                    "tokens": model.token_count(child.content),
                    "canonical_sha256": source["canonical_sha256"],
                }
                if child.parent:
                    pl = child.parent.locator.coordinates
                    if (
                        text[pl["start"] : pl["end"]] != child.parent.content
                        or not pl["start"] <= start < end <= pl["end"]
                    ):
                        raise ValueError("Parent source interval mismatch")
                    key = (docid, pl["start"], pl["end"])
                    if key not in parents:
                        parents[key] = {
                            "start": pl["start"],
                            "end": pl["end"],
                            "text": child.parent.content,
                            "tokens": model.token_count(child.parent.content),
                        }
                    record["parent"] = parents[key]
                chunks.append(record)
        parse_seconds = time.perf_counter() - parse_start
        lex_start = time.perf_counter()
        lexical = BM25(chunks)
        lexical_seconds = time.perf_counter() - lex_start
        print(f"index {cut}: {len(chunks)} chunks", flush=True)
        vectors = model.encode([c["text"] for c in chunks], "passage: ")
        indices.append(
            {
                "cut": cut,
                "chunk_chars": size,
                "overlap_chars": overlap,
                "parent_chars": parent_size,
                "chunk_count": len(chunks),
                "parent_count": len(parents),
                "parse_and_token_count_seconds": parse_seconds,
                "lexical_index_seconds": lexical_seconds,
                "encoding": dict(model.last_encoding),
                "vector_bytes": vectors.nbytes,
                "vector_sha256": sha(vectors.tobytes()),
                "parser_version": parser.parser_version,
                "actual_embedding_truncated_chunks": sum(
                    len(model.context_tokenizer.encode("passage: " + c["text"]).ids) > 512
                    for c in chunks
                ),
                "chunk_chars_p50_p95_max": [
                    float(np.percentile([len(c["text"]) for c in chunks], n)) for n in (50, 95, 100)
                ],
                "chunk_tokens_p50_p95_max": [
                    float(np.percentile([c["tokens"] for c in chunks], n)) for n in (50, 95, 100)
                ],
            }
        )
        for config in configurations:
            name = config_id(config)
            predictions, ledger = [], []
            for qi, query in enumerate(queries):
                observed, first = [], None
                for _repeat in range(ranking_repeats):
                    tick = time.perf_counter()
                    sparse = (
                        rank(
                            lexical.scores(query["query"]),
                            config["candidate_k"],
                            positive_only=True,
                        )
                        if config["route"] != "dense_e5"
                        else []
                    )
                    dense = (
                        rank(vectors @ query_vectors[qi], config["candidate_k"])
                        if config["route"] != "bm25"
                        else []
                    )
                    order = (
                        sparse
                        if config["route"] == "bm25"
                        else dense
                        if config["route"] == "dense_e5"
                        else rrf(sparse, dense, config["rrf_k"])
                    )
                    hits, packed = pack_context(
                        chunks,
                        order,
                        config["context_tokens"],
                        parent=config["parent"],
                        max_chunks=config["max_chunks"],
                        token_counter=model.token_count,
                    )
                    elapsed_ms = (time.perf_counter() - tick) * 1000
                    signature = (hits, packed["rendered_context_sha256"], packed["used_tokens"])
                    if first is None:
                        first = signature
                    elif first != signature:
                        raise ValueError("Deterministic retrieval changed across timing repeats")
                    observed.append(elapsed_ms)
                predictions.append(
                    {"sample_id": query["sample_id"], "abstained": False, "chunks": hits}
                )
                ledger.append(
                    {
                        "sample_id": query["sample_id"],
                        "config": name,
                        "sparse_candidates": len(sparse),
                        "dense_candidates": len(dense),
                        "candidate_union": len(set(sparse) | set(dense)),
                        "route_overlap": len(set(sparse) & set(dense)),
                        "first_traversal_ranking_packing_ms": observed[0],
                        "repeated_traversal_ranking_packing_ms": observed[1:],
                        **packed,
                    }
                )
            builder.write_jsonl(output / (name + ".contexts.jsonl"), ledger)
            for variant in sorted({q["query_variant_type"] for q in queries}):
                split = "test" if phase == "test" else "dev"
                ids = {q["sample_id"] for q in queries if q["query_variant_type"] == variant}
                key = name + "-" + split + "-" + variant
                prediction_path = output / (key + ".predictions.jsonl")
                builder.write_jsonl(
                    prediction_path, [p for p in predictions if p["sample_id"] in ids]
                )
                scored = scorer.evaluate(dataset, prediction_path, split, variant=variant)
                internals[key] = scored
                scores_path = output / (key + ".scores.json")
                builder.write_json(scores_path, scored)
                audit = [r for r in ledger if r["sample_id"] in ids]
                warm = [v for r in audit for v in r["repeated_traversal_ranking_packing_ms"]]
                results.append(
                    {
                        "config": name,
                        "parameters": config,
                        "split": split,
                        "variant": variant,
                        "query_count": scored["query_count"],
                        "answerable_fact_count": scored["answerable_fact_count"],
                        "metrics": scored["metrics"],
                        "scores_file": scores_path.name,
                        "scores_sha256": sha(scores_path.read_bytes()),
                        "first_traversal_ms_p50_p95": [
                            float(
                                np.percentile(
                                    [r["first_traversal_ranking_packing_ms"] for r in audit], n
                                )
                            )
                            for n in (50, 95)
                        ],
                        "repeated_traversal_ms_p50_p95": [
                            float(np.percentile(warm, n)) for n in (50, 95)
                        ],
                        "mean_context_tokens": float(np.mean([r["used_tokens"] for r in audit])),
                    }
                )
            print(name, flush=True)
    differences = []
    if phase == "test":
        for c in configs:
            if c["route"] != "hybrid_rrf":
                continue
            for route in ("bm25", "dense_e5"):
                left, right = config_id({**c, "route": route}), config_id(c)
                differences.append(
                    {
                        "left": left,
                        "right": right,
                        **paired_interval(
                            internals[left + "-test-native"]["per_query"],
                            internals[right + "-test-native"]["per_query"],
                            "required_evidence_hit_at_5",
                        ),
                    }
                )
        for c in configs:
            if not c["parent"]:
                continue
            left, right = config_id({**c, "parent": False}), config_id(c)
            differences.append(
                {
                    "contrast": "parent_expansion",
                    "left": left,
                    "right": right,
                    **paired_interval(
                        internals[left + "-test-native"]["per_query"],
                        internals[right + "-test-native"]["per_query"],
                        "required_evidence_hit_at_5",
                    ),
                }
            )
        main = policy["primary_comparison"]
        differences.append(
            {
                "contrast": "primary_dev_selected_vs_fixed_bm25",
                **main,
                **paired_interval(
                    internals[main["left"] + "-test-native"]["per_query"],
                    internals[main["right"] + "-test-native"]["per_query"],
                    "required_evidence_hit_at_5",
                ),
            }
        )
    if phase == "dev":
        chosen = choose_on_dev(results)
        builder.write_json(
            output / "selection.json",
            {
                "schema": "dev-selection@1",
                "phase": "dev",
                "dataset_release_sha256": dataset_hash,
                "source_hashes": code_hashes,
                "model": model.identity,
                "selected_config": chosen["parameters"],
                "development_records": results,
                "selection_rule": policy["selection_rule"],
                "test_matrix": configs_for({k: chosen["parameters"][k] for k in default_preset()}),
            },
        )
    if (
        source_hashes() != code_hashes
        or sha((dataset / "release.json").read_bytes()) != dataset_hash
    ):
        raise ValueError("Input or code changed during execution; run cannot be frozen")
    scorer.verify_dataset_fingerprints(dataset)
    artifacts = {
        p.name: sha(p.read_bytes())
        for p in output.iterdir()
        if p.is_file() and p.name != "run_state.json"
    }
    report = {
        "policy_sha256": sha((output / "policy.json").read_bytes()),
        "policy": policy,
        "phase": phase,
        "results": results,
        "paired_differences": differences,
        "model_initialization_seconds": initialization_seconds,
        "query_encoding": query_encoding,
        "indices": indices,
        "artifacts_sha256": artifacts,
        "selected_deployment_config": selection["selected_config"] if selection else None,
        "status": "pilot_not_formal_results" if phase == "pilot" else "source_verified_ablation",
        "limitations": [
            "Selected spans are not exhaustive relevance judgments; no nDCG/MRR claim",
            "No generation, calibrated rejection, KG fusion, reranking "
            "or maintenance decision evaluated",
            "One quantized embedding model; no quantization-effect or universal superiority claim",
            "Source dependency components may be few/unbalanced; paired intervals descriptive",
            "Repeated deterministic retrieval does not increase independent fact sample size",
            "First traversal is not a cold OS/model-cache experiment; embeddings are precomputed",
            "Hybrid union can be twice single-route candidates; context token budget is equal",
            "Context tokens include source wrappers, not a full generation prompt",
        ],
    }
    builder.write_json(output / "summary.json", report)
    builder.write_json(
        output / "run_state.json",
        {
            "status": "complete",
            "phase": phase,
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "summary_sha256": sha((output / "summary.json").read_bytes()),
        },
    )
    return report


def main() -> None:
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--dataset", type=Path, required=True)
    cli.add_argument("--model-dir", type=Path, required=True)
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--phase", choices=("dev", "test", "pilot"), default="dev")
    cli.add_argument("--selection-file", type=Path)
    cli.add_argument("--presets-file", type=Path, help="1-3 registered dev-only parameter presets")
    cli.add_argument(
        "--cache-root", type=Path, help="Reusable offline cache outside immutable runs"
    )
    cli.add_argument("--ranking-repeats", type=int, default=3)
    cli.add_argument("--threads", type=int, default=4)
    args = cli.parse_args()
    presets = (
        json.loads(args.presets_file.read_text(encoding="utf-8")) if args.presets_file else None
    )
    run(
        args.dataset,
        args.model_dir,
        args.output,
        args.threads,
        phase=args.phase,
        selection_file=args.selection_file,
        presets=presets,
        cache_root=args.cache_root,
        ranking_repeats=args.ranking_repeats,
    )


if __name__ == "__main__":
    main()
