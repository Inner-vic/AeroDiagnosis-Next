"""Frozen-corpus production retrieval benchmark; candidates never become gold labels."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import random
import statistics
import tempfile
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

ReviewStatus = Literal["unreviewed", "machine_candidate", "human_reviewed"]
Split = Literal["unassigned", "train", "dev", "test"]
SHA256_PATTERN = r"^[0-9a-f]{64}$"


def sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    return sha256(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode()
    )


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class BenchmarkDocument(FrozenModel):
    document_id: str = Field(min_length=1)
    processed_path: str = Field(min_length=1)
    sha256: str = Field(pattern=SHA256_PATTERN)
    canonical_sha256: str = Field(pattern=SHA256_PATTERN)
    cohort: Literal["historical_exact", "refreshed"]
    family_id: str = Field(min_length=1)
    family_reviewed: bool = False
    split: Split = "unassigned"


class ReviewedSpan(FrozenModel):
    document_id: str = Field(min_length=1)
    start: int = Field(ge=0, strict=True)
    end: int = Field(gt=0, strict=True)
    text_sha256: str = Field(pattern=SHA256_PATTERN)
    review_status: ReviewStatus = "unreviewed"
    reviewer: str | None = None

    @model_validator(mode="after")
    def ordered(self) -> ReviewedSpan:
        if self.end <= self.start:
            raise ValueError("evidence span must have positive length")
        if self.review_status == "human_reviewed" and not (self.reviewer or "").strip():
            raise ValueError("human-reviewed evidence requires a reviewer identity")
        return self


class BenchmarkQuery(FrozenModel):
    query_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    qrels: dict[str, Annotated[int, Field(ge=0, le=3, strict=True)]]
    review_status: ReviewStatus = "machine_candidate"
    reviewer: str | None = None
    answerability: Literal["unknown", "answerable", "unanswerable"] = "unknown"
    evidence_spans: tuple[ReviewedSpan, ...] = ()
    split: Split = "unassigned"
    intent: str = "unspecified"

    @model_validator(mode="after")
    def judgments(self) -> BenchmarkQuery:
        if any(type(grade) is not int or grade < 0 or grade > 3 for grade in self.qrels.values()):
            raise ValueError("qrels must be integer relevance grades 0..3")
        human = self.review_status == "human_reviewed"
        if human and not (self.reviewer or "").strip():
            raise ValueError("human-reviewed query requires a reviewer identity")
        if self.answerability != "unknown" and not human:
            raise ValueError("answerability labels require human review")
        positives = {document_id for document_id, grade in self.qrels.items() if grade > 0}
        if self.answerability == "unanswerable" and (positives or self.evidence_spans):
            raise ValueError("unanswerable query cannot contain positive qrels/evidence")
        if self.answerability != "unanswerable" and not positives:
            raise ValueError("query requires a positive qrel or human-reviewed unanswerable label")
        for span in self.evidence_spans:
            if span.document_id not in positives:
                raise ValueError("evidence span document must have positive relevance")
        return self


class BenchmarkDataset(FrozenModel):
    schema_version: Literal[1] = 1
    dataset_id: str = Field(min_length=1)
    snapshot_path: str = Field(min_length=1)
    snapshot_sha256: str = Field(pattern=SHA256_PATTERN)
    candidate_queries_sha256: str = Field(pattern=SHA256_PATTERN)
    documents: tuple[BenchmarkDocument, ...] = Field(min_length=1)
    queries: tuple[BenchmarkQuery, ...] = Field(min_length=1)
    unavailable_document_ids: tuple[str, ...] = ()
    annotation_protocol: str = (
        "Candidate qrels are not gold; review exact canonical spans and family splits."
    )
    test_policy_sha256: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @model_validator(mode="after")
    def isolation(self) -> BenchmarkDataset:
        docs = {doc.document_id: doc for doc in self.documents}
        if len(docs) != len(self.documents):
            raise ValueError("duplicate document identity")
        if len({query.query_id for query in self.queries}) != len(self.queries):
            raise ValueError("duplicate query identity")
        if set(docs) & set(self.unavailable_document_ids):
            raise ValueError("available and unavailable document identities overlap")
        families: dict[str, Split] = {}
        for doc in self.documents:
            if doc.family_id in families and families[doc.family_id] != doc.split:
                raise ValueError("document family crosses train/dev/test splits")
            families[doc.family_id] = doc.split
        known = set(docs) | set(self.unavailable_document_ids)
        for query in self.queries:
            if not set(query.qrels) <= known:
                raise ValueError("qrels reference an undeclared document")
            positive_docs = [
                docs[key] for key, grade in query.qrels.items() if grade > 0 and key in docs
            ]
            if query.split != "unassigned":
                if any(doc.split != query.split for doc in positive_docs):
                    raise ValueError("query judgments cross declared family splits")
                if any(key not in docs for key, grade in query.qrels.items() if grade > 0):
                    raise ValueError("split-assigned query has an unavailable positive source")
        return self

    @property
    def content_hash(self) -> str:
        return canonical_hash(self.model_dump(mode="json"))


class MatrixConfig(FrozenModel):
    chunk_strategies: tuple[Literal["fixed", "structure", "hierarchical"], ...] = (
        "fixed",
        "structure",
        "hierarchical",
    )
    retrieval_strategies: tuple[Literal["sparse_only", "vector_only", "hybrid_rrf"], ...] = (
        "sparse_only",
        "vector_only",
        "hybrid_rrf",
    )
    context_expansion: tuple[bool, ...] = (False, True)
    rerankers: tuple[Literal["identity", "lexical"], ...] = ("identity", "lexical")
    chunk_size: int = Field(default=800, ge=100)
    parent_budget: int = Field(default=3200, ge=100)
    child_budget: int = Field(default=800, ge=100)
    overlap: int = Field(default=100, ge=0)
    candidate_k: int = Field(default=50, ge=5, le=500)
    rrf_k: int = Field(default=60, ge=1, le=1000)
    max_per_document: int = Field(default=3, ge=0, le=500)
    vector_min_score: float = Field(default=0.05, ge=-1, le=1)
    rerank_char_budget: int = Field(default=24000, ge=100, le=1_000_000)
    lexical_rerank_weight: float = Field(default=0.45, ge=0, le=1)
    max_context_chars: int = Field(default=8000, ge=100, le=8000)
    top_k: Literal[5] = 5
    repeats: int = Field(default=3, ge=1)
    bootstrap_samples: int = Field(default=2000, ge=100)
    seed: int = 20261012
    purpose: Literal["candidate_regression", "heldout_test", "development"] = "candidate_regression"
    query_split: Literal["all", "train", "dev", "test"] = "all"

    @model_validator(mode="after")
    def frozen_policy(self) -> MatrixConfig:
        if self.parent_budget < self.child_budget:
            raise ValueError("parent budget cannot be smaller than child budget")
        if self.overlap >= self.chunk_size:
            raise ValueError("overlap must be smaller than chunk_size")
        if (
            not self.chunk_strategies
            or not self.retrieval_strategies
            or not self.context_expansion
            or not self.rerankers
        ):
            raise ValueError("matrix dimensions cannot be empty")
        if self.purpose == "heldout_test" and self.query_split != "test":
            raise ValueError("heldout_test must use the frozen test split")
        if self.purpose == "development" and self.query_split not in {"train", "dev"}:
            raise ValueError("development cannot use test/all queries")
        if any(
            len(values) != len(set(values))
            for values in (
                self.chunk_strategies,
                self.retrieval_strategies,
                self.context_expansion,
                self.rerankers,
            )
        ):
            raise ValueError("duplicate matrix policy")
        return self

    @property
    def policy_hash(self) -> str:
        payload = self.model_dump(mode="json")
        # Labeling the run as a test is not itself a policy change.
        payload.pop("purpose")
        payload.pop("query_split")
        return canonical_hash(payload)


@dataclass(frozen=True)
class BenchmarkHit:
    document_id: str
    chunk_id: str
    start: int | None = None
    end: int | None = None
    source_located: bool = False
    anchor_start: int | None = None
    anchor_end: int | None = None


class QueryScore(FrozenModel):
    query_id: str
    status: Literal["scored", "missing_source", "unjudged"]
    judgment_status: ReviewStatus
    doc_recall_at_5: float | None = None
    mrr_at_5: float | None = None
    ndcg_at_5: float | None = None
    reviewed_span_coverage: float | None = None
    reviewed_anchor_span_coverage: float | None = None
    no_answer_success: float | None = None
    retrieved_document_ids: tuple[str, ...] = ()
    retrieved_chunk_ids: tuple[str, ...] = ()
    missing_positive_document_ids: tuple[str, ...] = ()


def union_length(intervals: Sequence[tuple[int, int]]) -> int:
    total = 0
    right = -1
    for start, end in sorted(intervals):
        if end <= start:
            continue
        total += max(0, end - max(start, right))
        right = max(right, end)
    return total


def span_coverage(
    query: BenchmarkQuery, hits: Sequence[BenchmarkHit], *, anchor: bool = False
) -> float | None:
    annotations: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for span in query.evidence_spans:
        if span.review_status == "human_reviewed":
            annotations[span.document_id].append((span.start, span.end))
    denominator = sum(union_length(spans) for spans in annotations.values())
    if not denominator:
        return None
    intersections: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for hit in hits[:5]:
        if not hit.source_located:
            continue
        start = hit.anchor_start if anchor else hit.start
        end = hit.anchor_end if anchor else hit.end
        if start is None or end is None:
            continue
        for left, right in annotations[hit.document_id]:
            intersections[hit.document_id].append((max(start, left), min(end, right)))
    return sum(union_length(spans) for spans in intersections.values()) / denominator


def score_query(
    query: BenchmarkQuery, hits: Sequence[BenchmarkHit], available_document_ids: frozenset[str]
) -> QueryScore:
    top = hits[:5]
    positive = {key: grade for key, grade in query.qrels.items() if grade > 0}
    missing = tuple(sorted(set(positive) - available_document_ids))
    common: dict[str, Any] = {
        "query_id": query.query_id,
        "judgment_status": query.review_status,
        "retrieved_document_ids": tuple(hit.document_id for hit in top),
        "retrieved_chunk_ids": tuple(hit.chunk_id for hit in top),
    }
    if missing:
        return QueryScore(**common, status="missing_source", missing_positive_document_ids=missing)
    if query.review_status == "unreviewed":
        return QueryScore(**common, status="unjudged")
    if not positive:
        return QueryScore(**common, status="scored", no_answer_success=float(not top))
    seen = set()
    dcg = 0.0
    reciprocal = 0.0
    for rank, hit in enumerate(top, start=1):
        grade = positive.get(hit.document_id, 0) if hit.document_id not in seen else 0
        seen.add(hit.document_id)
        dcg += (2**grade - 1) / math.log2(rank + 1)
        if grade and not reciprocal:
            reciprocal = 1 / rank
    ideal = sum(
        (2**grade - 1) / math.log2(rank + 1)
        for rank, grade in enumerate(sorted(positive.values(), reverse=True)[:5], start=1)
    )
    return QueryScore(
        **common,
        status="scored",
        doc_recall_at_5=len(set(positive) & seen) / len(positive),
        mrr_at_5=reciprocal,
        ndcg_at_5=dcg / ideal,
        reviewed_span_coverage=span_coverage(query, top),
        reviewed_anchor_span_coverage=span_coverage(query, top, anchor=True),
    )


def percentile(values: Sequence[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    offset = (len(ordered) - 1) * probability
    left, right = math.floor(offset), math.ceil(offset)
    return ordered[left] + (ordered[right] - ordered[left]) * (offset - left)


def paired_bootstrap(
    left: Mapping[str, float],
    right: Mapping[str, float],
    *,
    seed: int = 20261012,
    samples: int = 2000,
    groups: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    ids = sorted(set(left) & set(right))
    if not ids:
        return {
            "paired_query_count": 0,
            "mean_delta_b_minus_a": None,
            "ci95": None,
            "wins_b": 0,
            "ties": 0,
            "losses_b": 0,
        }
    deltas = {key: right[key] - left[key] for key in ids}
    clusters: dict[str, list[str]] = defaultdict(list)
    for key in ids:
        clusters[groups[key] if groups is not None else key].append(key)
    cluster_ids = sorted(clusters)
    rng = random.Random(seed)
    distribution = []
    for _ in range(samples):
        chosen = [key for _ in cluster_ids for key in clusters[rng.choice(cluster_ids)]]
        distribution.append(statistics.fmean(deltas[key] for key in chosen))
    return {
        "paired_query_count": len(ids),
        "bootstrap_unit": "document_family" if groups else "query",
        "bootstrap_cluster_count": len(clusters),
        "bootstrap_samples": samples,
        "seed": seed,
        "mean_delta_b_minus_a": statistics.fmean(deltas.values()),
        "ci95": [percentile(distribution, 0.025), percentile(distribution, 0.975)],
        "wins_b": sum(value > 1e-12 for value in deltas.values()),
        "ties": sum(abs(value) <= 1e-12 for value in deltas.values()),
        "losses_b": sum(value < -1e-12 for value in deltas.values()),
        "interpretation": (
            "Exploratory paired bootstrap; small or unverified family groups limit inference."
        ),
    }


def load_dataset(path: Path) -> BenchmarkDataset:
    return BenchmarkDataset.model_validate_json(path.read_text(encoding="utf-8"))


def verify_corpus(
    dataset: BenchmarkDataset, dataset_path: Path
) -> tuple[dict[str, str], dict[str, Any]]:
    snapshot_path = (dataset_path.parent / dataset.snapshot_path).resolve()
    payload = snapshot_path.read_bytes()
    if sha256(payload) != dataset.snapshot_sha256:
        raise ValueError("frozen snapshot hash changed")
    snapshot = json.loads(payload)
    pinned = {row["document_id"]: row for row in snapshot["documents"]}
    if set(pinned) != {doc.document_id for doc in dataset.documents}:
        raise ValueError("dataset document inventory differs from frozen snapshot")
    if set(snapshot.get("unavailable_document_ids", [])) != set(dataset.unavailable_document_ids):
        raise ValueError("unavailable document inventory differs from snapshot")
    texts = {}
    for doc in dataset.documents:
        row = pinned[doc.document_id]
        if (
            doc.sha256 != row["sha256"]
            or doc.cohort != row["cohort"]
            or doc.processed_path != row["processed_path"]
        ):
            raise ValueError("document lineage differs from frozen snapshot")
        path = (snapshot_path.parent / doc.processed_path).resolve()
        if not path.is_relative_to(snapshot_path.parent):
            raise ValueError("processed path escapes frozen snapshot")
        content = path.read_bytes()
        if sha256(content) != doc.sha256:
            raise ValueError(f"canonical text hash changed: {doc.document_id}")
        text = content.decode("utf-8-sig").replace("\r\n", "\n").replace("\r", "\n")
        if sha256(text.encode()) != doc.canonical_sha256:
            raise ValueError(f"canonical LF text hash changed: {doc.document_id}")
        texts[doc.document_id] = text
        for artifact in row["artifacts"]:
            raw = (snapshot_path.parent / "raw" / artifact["name"]).resolve()
            if not raw.is_relative_to(snapshot_path.parent / "raw"):
                raise ValueError("artifact path escapes frozen snapshot")
            data = raw.read_bytes()
            if sha256(data) != artifact["sha256"] or len(data) != artifact["bytes"]:
                raise ValueError(f"frozen raw artifact changed: {artifact['name']}")
    for query in dataset.queries:
        for span in query.evidence_spans:
            annotated_text = texts.get(span.document_id)
            if (
                annotated_text is None
                or span.end > len(annotated_text)
                or sha256(annotated_text[span.start : span.end].encode()) != span.text_sha256
            ):
                raise ValueError(f"evidence annotation span/hash mismatch: {query.query_id}")
    return texts, snapshot


def prepare_dataset(
    snapshot_path: Path,
    queries_path: Path,
    output: Path,
    family_plan: Mapping[str, Any] | None = None,
) -> BenchmarkDataset:
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    document_rows = []
    family_plan = family_plan or {}
    if set(family_plan) - {row["document_id"] for row in snapshot["documents"]}:
        raise ValueError("family plan contains documents outside frozen snapshot")
    for row in snapshot["documents"]:
        plan = family_plan.get(row["document_id"], {})
        document_rows.append(
            BenchmarkDocument(
                document_id=row["document_id"],
                processed_path=row["processed_path"],
                sha256=row["sha256"],
                canonical_sha256=sha256(
                    (snapshot_path.parent / row["processed_path"])
                    .read_bytes()
                    .decode("utf-8-sig")
                    .replace("\r\n", "\n")
                    .replace("\r", "\n")
                    .encode()
                ),
                cohort=row["cohort"],
                family_id=plan.get("family_id", "unassigned:" + row["document_id"]),
                split=plan.get("split", "unassigned"),
                family_reviewed=False,
            )
        )
    queries = []
    document_splits = {doc.document_id: doc.split for doc in document_rows}
    for line in queries_path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            positive_keys = {key for key, grade in row["qrels"].items() if grade > 0}
            splits = {document_splits[key] for key in positive_keys if key in document_splits}
            split: Split = (
                next(iter(splits))
                if len(splits) == 1 and positive_keys <= document_splits.keys()
                else "unassigned"
            )
            # Preparation is mechanical and never promotes existing candidate labels.
            queries.append(
                BenchmarkQuery(
                    query_id=row["query_id"],
                    query=row["query"],
                    qrels=row["qrels"],
                    intent=row.get("intent", "unspecified"),
                    review_status="machine_candidate",
                    split=split,
                )
            )
    dataset = BenchmarkDataset(
        dataset_id=snapshot["dataset_id"] + "-retrieval-annotation",
        snapshot_path=os.path.relpath(snapshot_path.resolve(), output.parent.resolve()).replace(
            "\\", "/"
        ),
        snapshot_sha256=sha256(snapshot_path.read_bytes()),
        candidate_queries_sha256=sha256(queries_path.read_bytes()),
        documents=tuple(document_rows),
        queries=tuple(queries),
        unavailable_document_ids=tuple(snapshot.get("unavailable_document_ids", [])),
    )
    verify_corpus(dataset, output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dataset.model_dump_json(indent=2), encoding="utf-8")
    return dataset


def validate_run(dataset: BenchmarkDataset, config: MatrixConfig) -> None:
    selected_queries = [
        query
        for query in dataset.queries
        if config.query_split == "all" or query.split == config.query_split
    ]
    # Run labels cannot bypass a reviewed test set's frozen policy. Candidate
    # regression is also prohibited from probing alternate policies on gold test queries.
    reviewed_test = any(
        query.split == "test" and query.review_status == "human_reviewed"
        for query in selected_queries
    )
    if reviewed_test and dataset.test_policy_sha256 != config.policy_hash:
        raise ValueError("reviewed test queries require their frozen policy in every run mode")
    if config.purpose == "heldout_test":
        if dataset.test_policy_sha256 != config.policy_hash:
            raise ValueError(
                "test policies must be frozen in the reviewed dataset before execution"
            )
        selected = [query for query in dataset.queries if query.split == "test"]
        if not selected or any(query.review_status != "human_reviewed" for query in selected):
            raise ValueError("heldout test requires human-reviewed test queries")
        if any(not doc.family_reviewed or doc.split == "unassigned" for doc in dataset.documents):
            raise ValueError("heldout test requires reviewed document-family assignment")


METRICS = (
    "doc_recall_at_5",
    "mrr_at_5",
    "ndcg_at_5",
    "reviewed_span_coverage",
    "reviewed_anchor_span_coverage",
    "no_answer_success",
)


def summarize_scores(scores: Sequence[QueryScore], latency_ms: Sequence[float]) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "query_count": len(scores),
        "status_counts": dict(Counter(score.status for score in scores)),
        "latency_sample_count": len(latency_ms),
        "latency_p50_ms": percentile(latency_ms, 0.5),
        "latency_p95_ms": percentile(latency_ms, 0.95),
    }
    for metric in METRICS:
        values = [value for score in scores if (value := getattr(score, metric)) is not None]
        summary[metric] = statistics.fmean(values) if values else None
        summary[metric + "_query_count"] = len(values)
    return summary


def export_report(report: Mapping[str, Any], output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2), encoding="utf-8"
    )
    with (output / "query_scores.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["policy", "query_id", "status", "judgment_status", *METRICS]
        )
        writer.writeheader()
        for row in report.get("policies", []):
            for score in row["query_scores"]:
                writer.writerow(
                    {
                        "policy": row["name"],
                        **{key: score.get(key) for key in writer.fieldnames if key != "policy"},
                    }
                )


def _source_hashes() -> dict[str, str]:
    package_root = Path(__file__).resolve().parent
    return {
        name: sha256((package_root / name.removeprefix("src/aerodiagnosis/")).read_bytes())
        for name in (
            "src/aerodiagnosis/retrieval_benchmark.py",
            "src/aerodiagnosis/retrieval/manual.py",
            "src/aerodiagnosis/retrieval/reranker.py",
            "src/aerodiagnosis/ingestion/parsers.py",
            "src/aerodiagnosis/ingestion/structure.py",
            "src/aerodiagnosis/ingestion/service.py",
            "src/aerodiagnosis/adapters/persistence/sqlite_lexical.py",
            "src/aerodiagnosis/adapters/persistence/sqlite_vector.py",
            "src/aerodiagnosis/adapters/persistence/sqlite_context.py",
        )
    }


def _match_record(match: Any, texts: Mapping[str, str]) -> tuple[BenchmarkHit, dict[str, Any]]:
    chunk = match.chunk
    if chunk.document_id not in texts:
        raise ValueError("retrieval returned document outside frozen corpus")
    locator = dict(chunk.metadata.get("locator", {}))
    coordinates = locator.get("coordinates", {})
    start, end = coordinates.get("start"), coordinates.get("end")
    valid_range = type(start) is int and type(end) is int and 0 <= start < end
    source_located = bool(valid_range and texts[chunk.document_id][start:end] == chunk.content)
    if not source_located:
        raise ValueError("returned production evidence does not match frozen canonical span")
    retrieval = dict(chunk.metadata.get("retrieval", {}))
    child = retrieval.get("child_locator", locator).get("coordinates", {})
    anchor_start, anchor_end = child.get("start"), child.get("end")
    if type(anchor_start) is not int or type(anchor_end) is not int:
        anchor_start, anchor_end = None, None
    elif not (start <= anchor_start < anchor_end <= end):
        raise ValueError("retrieval anchor escapes returned source context")
    hit = BenchmarkHit(
        chunk.document_id, chunk.chunk_id, start, end, source_located, anchor_start, anchor_end
    )
    return hit, {
        "document_id": chunk.document_id,
        "chunk_id": chunk.chunk_id,
        "version_id": chunk.version_id,
        "score": match.score,
        "locator": locator,
        "source_located": source_located,
        "content_sha256": sha256(chunk.content.encode()),
        "evidence_id": chunk.metadata.get("evidence_id"),
        "retrieval": retrieval,
        "excerpt": chunk.content[:400],
        "excerpt_truncated": len(chunk.content) > 400,
    }


def run_matrix(
    dataset: BenchmarkDataset, dataset_path: Path, config: MatrixConfig
) -> dict[str, Any]:
    # These are the production parser, ingestion service, storage and retrieval modules.
    from aerodiagnosis.adapters.persistence.sqlite_context import SQLiteParentContextStore
    from aerodiagnosis.adapters.persistence.sqlite_lexical import SQLiteLexicalStore
    from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
    from aerodiagnosis.ingestion.manifest import DocumentManifest
    from aerodiagnosis.ingestion.parsers import ParserRegistry, PlainTextParser
    from aerodiagnosis.ingestion.service import DocumentIngestionService
    from aerodiagnosis.retrieval.manual import ManualEvidenceRetriever, ManualRetrievalPolicy
    from aerodiagnosis.retrieval.reranker import IdentityReranker, LexicalReranker, Reranker

    validate_run(dataset, config)
    texts, _snapshot = verify_corpus(dataset, dataset_path)
    source_hashes = _source_hashes()
    queries = [
        query
        for query in dataset.queries
        if config.query_split == "all" or query.split == config.query_split
    ]
    if not queries:
        raise ValueError("selected query split is empty")
    available = frozenset(texts)
    policies: list[dict[str, Any]] = []
    indexes = []
    scores_by_policy: dict[str, list[QueryScore]] = {}
    doc_families = {doc.document_id: doc.family_id for doc in dataset.documents}
    groups = {
        query.query_id: "+".join(
            sorted(
                {
                    doc_families[key]
                    for key, grade in query.qrels.items()
                    if grade > 0 and key in doc_families
                }
            )
        )
        or query.query_id
        for query in queries
    }
    with tempfile.TemporaryDirectory(prefix="aero-retrieval-benchmark-") as temporary:
        for cut in config.chunk_strategies:
            path = Path(temporary) / (cut + ".sqlite3")
            manifest = DocumentManifest(path)
            vector_store = SQLiteVectorStore(path)
            lexical_store = SQLiteLexicalStore(path)
            context_store = SQLiteParentContextStore(path)
            index_retriever = ManualEvidenceRetriever(
                vector_store=vector_store, lexical_store=lexical_store, context_store=context_store
            )
            parser_impl = PlainTextParser(
                strategy=cut,
                chunk_size=config.chunk_size,
                overlap=config.overlap,
                parent_budget=config.parent_budget,
                child_budget=config.child_budget,
            )
            parsed_docs = {key: parser_impl.parse(text.encode()) for key, text in texts.items()}
            lengths = [
                len(chunk.content) for parsed in parsed_docs.values() for chunk in parsed.chunks
            ]
            packing_budget = config.child_budget if cut == "hierarchical" else config.chunk_size
            service = DocumentIngestionService(
                manifest,
                index_retriever,
                graph_store=None,
                parsers=ParserRegistry({".txt": parser_impl}),
                context_store=context_store,
            )
            index_start = time.perf_counter_ns()
            indexed = [
                service.ingest(
                    display_name=doc.document_id.lower() + ".txt",
                    content=texts[doc.document_id].encode(),
                    document_id=doc.document_id,
                )
                for doc in dataset.documents
            ]
            indexes.append(
                {
                    "chunk_strategy": cut,
                    "indexing_ms": (time.perf_counter_ns() - index_start) / 1e6,
                    "parser_version": parser_impl.parser_version,
                    "packing_budget_characters": packing_budget,
                    "chunk_characters_total": sum(lengths),
                    "chunk_characters_p50": percentile(lengths, 0.5),
                    "chunk_characters_p95": percentile(lengths, 0.95),
                    "chunk_characters_max": max(lengths),
                    "chunks_over_packing_budget": sum(
                        length > packing_budget for length in lengths
                    ),
                    "budget_note": (
                        "Packing budget, not a universal hard split limit; "
                        "actual lengths are reported."
                    ),
                    "vector_chunks": vector_store.count(),
                    "lexical_chunks": lexical_store.count(),
                    "embedding_identity": vector_store.embedding_identity,
                    "documents": [
                        {
                            "document_id": result.document_id,
                            "version_id": result.version_id,
                            "chunk_count": result.chunk_count,
                        }
                        for result in indexed
                    ],
                }
            )
            versions = manifest.active_version_ids()
            for route, expand, reranker_name in itertools.product(
                config.retrieval_strategies, config.context_expansion, config.rerankers
            ):
                name = f"{cut}/{route}/context_{'on' if expand else 'off'}/{reranker_name}"
                reranker: Reranker = (
                    IdentityReranker()
                    if reranker_name == "identity"
                    else LexicalReranker(lexical_weight=config.lexical_rerank_weight)
                )
                retriever = ManualEvidenceRetriever(
                    vector_store=vector_store,
                    lexical_store=lexical_store,
                    context_store=context_store,
                    policy=ManualRetrievalPolicy(
                        strategy=route,
                        candidate_k=config.candidate_k,
                        rrf_k=config.rrf_k,
                        max_per_document=config.max_per_document,
                        expand_context=expand,
                        vector_min_score=config.vector_min_score,
                        max_context_chars=config.max_context_chars,
                        rerank_char_budget=config.rerank_char_budget,
                    ),
                    reranker=reranker,
                )
                latency_ms = []
                query_scores = []
                results: list[dict[str, Any]] = []
                for query in queries:
                    initial = score_query(query, (), available)
                    if initial.status in {"missing_source", "unjudged"}:
                        query_scores.append(initial)
                        results.append(
                            {
                                "query_id": query.query_id,
                                "hits": [],
                                "latency_ms": [],
                                "skip_reason": initial.status,
                            }
                        )
                        continue
                    expected_records = None
                    first_hits: list[BenchmarkHit] = []
                    timings = []
                    for _ in range(config.repeats):
                        started = time.perf_counter_ns()
                        matches = retriever.search(
                            query.query, top_k=config.top_k, active_version_ids=versions
                        )
                        timings.append((time.perf_counter_ns() - started) / 1e6)
                        pairs = [_match_record(match, texts) for match in matches]
                        hits = [pair[0] for pair in pairs]
                        records = [pair[1] for pair in pairs]
                        if expected_records is not None and expected_records != records:
                            raise ValueError("repeated quality results are nondeterministic")
                        expected_records = records
                        first_hits = hits
                    latency_ms.extend(timings)
                    query_scores.append(score_query(query, first_hits, available))
                    results.append(
                        {
                            "query_id": query.query_id,
                            "hits": expected_records,
                            "latency_ms": timings,
                        }
                    )
                scores_by_policy[name] = query_scores
                policies.append(
                    {
                        "name": name,
                        "chunk_strategy": cut,
                        "retrieval_strategy": route,
                        "context_expansion": expand,
                        "reranker_identity": reranker.identity,
                        "production_manual_policy_identity": retriever.policy.identity,
                        "production_manual_policy": asdict(retriever.policy),
                        "summary": summarize_scores(query_scores, latency_ms),
                        "query_scores": [score.model_dump(mode="json") for score in query_scores],
                        "results": results,
                    }
                )
    if source_hashes != _source_hashes():
        raise ValueError("production source changed during benchmark execution")
    baseline = "fixed/vector_only/context_off/identity"
    if baseline not in scores_by_policy:
        baseline = next(iter(scores_by_policy))
    pairwise = []
    for name, scores in scores_by_policy.items():
        if name == baseline:
            continue
        for metric in METRICS:
            left = {
                score.query_id: value
                for score in scores_by_policy[baseline]
                if (value := getattr(score, metric)) is not None
            }
            right = {
                score.query_id: value
                for score in scores
                if (value := getattr(score, metric)) is not None
            }
            if not left and not right:
                continue
            pairwise.append(
                {
                    "policy_a": baseline,
                    "policy_b": name,
                    "metric": metric,
                    **paired_bootstrap(
                        left,
                        right,
                        seed=config.seed,
                        samples=config.bootstrap_samples,
                        groups=groups,
                    ),
                }
            )
    candidate_count = sum(query.review_status == "machine_candidate" for query in queries)
    partial = bool(dataset.unavailable_document_ids)
    ready = config.purpose == "heldout_test" and not partial and not candidate_count
    quality_hash = canonical_hash(
        {
            "dataset_hash": dataset.content_hash,
            "policy_hash": config.policy_hash,
            "production_source_sha256": source_hashes,
            "policies": [
                {
                    "name": row["name"],
                    "query_scores": row["query_scores"],
                    "results": [
                        {"query_id": result["query_id"], "hits": result["hits"]}
                        for result in row["results"]
                    ],
                }
                for row in policies
            ],
        }
    )
    return {
        "schema_version": 1,
        "status": "partial_candidate_regression" if partial or candidate_count else "complete",
        "publication_ready": ready,
        "dataset_id": dataset.dataset_id,
        "dataset_hash": dataset.content_hash,
        "snapshot_sha256": dataset.snapshot_sha256,
        "candidate_queries_sha256": dataset.candidate_queries_sha256,
        "dataset_protocol": dataset.model_dump(mode="json"),
        "production_source_sha256": source_hashes,
        "config": config.model_dump(mode="json"),
        "policy_hash": config.policy_hash,
        "quality_hash": quality_hash,
        "document_count": len(dataset.documents),
        "selected_query_count": len(queries),
        "candidate_query_count": candidate_count,
        "unavailable_document_ids": dataset.unavailable_document_ids,
        "cohort_counts": dict(Counter(doc.cohort for doc in dataset.documents)),
        "indexes": indexes,
        "policies": policies,
        "pairwise": pairwise,
        "metric_protocol": {
            "rank_unit": (
                "Top 5 returned chunks/context hits; document IDs deduplicated for "
                "relevance gains only."
            ),
            "docRecall@5chunk": (
                "Distinct relevant documents among top 5 chunks / all positive qrel documents."
            ),
            "MRR@5": "Reciprocal original chunk rank of first relevant document, zero if none.",
            "nDCG@5": (
                "Graded 2^relevance-1; duplicate document hits gain zero at their original ranks."
            ),
            "span_coverage": (
                "Union of human-reviewed canonical character spans covered by returned "
                "evidence; child anchor reported separately."
            ),
            "no_answer": (
                "Retrieval abstention only, exclusively human-reviewed unanswerable "
                "queries; not answer correctness."
            ),
            "latency": (
                "Local perf_counter search calls, repeated fixed order, no "
                "generation/network; indexing separate; no hardware-independent claim."
            ),
            "paired_ci": (
                "Fixed-seed percentile paired bootstrap over document-family query "
                "groups; no significance claim from candidate judgments."
            ),
        },
        "limitations": [
            "Hashing@256 is a lexical vector baseline, not a trained semantic embedding.",
            "Machine-candidate qrels support regression only, not gold academic accuracy.",
            "Missing positive-source queries are excluded, never converted to zero recall.",
            (
                "Unjudged documents are treated as zero gain; incomplete qrels limit "
                "precision of conclusions."
            ),
            (
                "Family assignments remain provisional until reviewed; prepare does not "
                "guess reviewed spans or no-answer labels."
            ),
            "Canonical TXT parsing only; native PDF/Office layouts are outside this matrix.",
            "No parameter fitting is performed; heldout policies must be frozen before running.",
            (
                "Fixed/structure use chunk_size; hierarchy uses explicit "
                "child_budget/parent_budget, so unequal budgets must not be treated as a "
                "single-variable ablation."
            ),
            "Different repeated-run latency samples are expected; quality_hash excludes timings.",
        ],
    }


def compact_report(report: Mapping[str, Any]) -> dict[str, Any]:
    """Shareable aggregate receipt; full hit locators stay in the runtime report."""
    keys = (
        "schema_version",
        "status",
        "publication_ready",
        "dataset_id",
        "dataset_hash",
        "snapshot_sha256",
        "candidate_queries_sha256",
        "production_source_sha256",
        "config",
        "policy_hash",
        "quality_hash",
        "document_count",
        "selected_query_count",
        "candidate_query_count",
        "unavailable_document_ids",
        "cohort_counts",
        "indexes",
        "pairwise",
        "metric_protocol",
        "limitations",
    )
    result = {key: report[key] for key in keys}
    protocol = report["dataset_protocol"]
    result["documents"] = protocol["documents"]
    queries = protocol["queries"]
    result["judgment_summary"] = {
        "review_status_counts": dict(Counter(query["review_status"] for query in queries)),
        "query_split_counts": dict(Counter(query["split"] for query in queries)),
        "answerability_counts": dict(Counter(query["answerability"] for query in queries)),
        "human_reviewed_span_count": sum(
            span["review_status"] == "human_reviewed"
            for query in queries
            for span in query["evidence_spans"]
        ),
        "family_assignments_reviewed": all(doc["family_reviewed"] for doc in protocol["documents"]),
    }
    result["policies"] = [
        {
            key: row[key]
            for key in (
                "name",
                "chunk_strategy",
                "retrieval_strategy",
                "context_expansion",
                "reranker_identity",
                "production_manual_policy_identity",
                "production_manual_policy",
                "summary",
            )
        }
        for row in report["policies"]
    ]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    prepare = subparsers.add_parser(
        "prepare",
        help="Generate unreviewed span/answerability annotation template from real frozen files",
    )
    prepare.add_argument("--snapshot", type=Path, required=True)
    prepare.add_argument("--queries", type=Path, required=True)
    prepare.add_argument("--output", type=Path, required=True)
    prepare.add_argument("--families", type=Path)
    evaluate = subparsers.add_parser(
        "run", help="Production retrieval policy matrix, without fitting/tuning"
    )
    evaluate.add_argument("--dataset", type=Path, required=True)
    evaluate.add_argument("--config", type=Path)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument(
        "--compact-output", type=Path, help="Optional shareable aggregate JSON receipt"
    )
    args = parser.parse_args()
    if args.command == "prepare":
        families = json.loads(args.families.read_text(encoding="utf-8")) if args.families else None
        dataset = prepare_dataset(args.snapshot, args.queries, args.output, families)
        print(
            json.dumps(
                {
                    "dataset_hash": dataset.content_hash,
                    "documents": len(dataset.documents),
                    "candidate_queries": len(dataset.queries),
                    "is_gold": False,
                }
            )
        )
        return 0
    dataset = load_dataset(args.dataset)
    config = (
        MatrixConfig.model_validate_json(args.config.read_text(encoding="utf-8"))
        if args.config
        else MatrixConfig()
    )
    report = run_matrix(dataset, args.dataset, config)
    export_report(report, args.output)
    if args.compact_output:
        args.compact_output.parent.mkdir(parents=True, exist_ok=True)
        args.compact_output.write_text(
            json.dumps(compact_report(report), ensure_ascii=False, sort_keys=True, indent=2),
            encoding="utf-8",
        )
    print(
        json.dumps(
            {
                "status": report["status"],
                "publication_ready": report["publication_ready"],
                "output": str(args.output),
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
