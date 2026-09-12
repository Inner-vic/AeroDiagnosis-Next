"""Reproducible single-route, fusion and RRF sensitivity experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from aerodiagnosis.application.use_cases import HybridRetrieval
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import RetrievalStrategy
from aerodiagnosis.evaluation import (
    RetrievalMetrics,
    evaluate_retrieval_sources,
    evidence_source_key,
)


class RetrievalBenchmarkCase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    case_id: str = Field(min_length=1)
    query: str = Field(min_length=2)
    relevant_source_keys: frozenset[str] = Field(min_length=1)


class RetrievalBenchmark(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str = Field(min_length=1)
    version: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)
    judgment_protocol: str = Field(min_length=1)
    cases: tuple[RetrievalBenchmarkCase, ...] = Field(min_length=1)

    @property
    def content_hash(self) -> str:
        payload = self.model_dump(mode="json")
        for case in payload["cases"]:
            case["relevant_source_keys"] = sorted(case["relevant_source_keys"])
        canonical = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(canonical.encode()).hexdigest()


class RetrievalPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    name: str = Field(pattern=r"^[a-z0-9_]+$")
    strategy: RetrievalStrategy
    rrf_k: int = Field(default=60, ge=1, le=1000)
    manual_weight: float = Field(default=1.0, gt=0, le=1)
    graph_weight: float = Field(default=0.88, gt=0, le=1)
    case_weight: float = Field(default=0.94, gt=0, le=1)
    enforce_route_coverage: bool = False

    @property
    def weights(self) -> dict[str, float]:
        return {
            "manual": self.manual_weight,
            "graph": self.graph_weight,
            "case": self.case_weight,
        }


DEFAULT_POLICIES = (
    RetrievalPolicy(name="manual_only", strategy=RetrievalStrategy.MANUAL_ONLY),
    RetrievalPolicy(name="graph_only", strategy=RetrievalStrategy.GRAPH_ONLY),
    RetrievalPolicy(name="case_only", strategy=RetrievalStrategy.CASE_ONLY),
    RetrievalPolicy(name="rrf_k10", strategy=RetrievalStrategy.RRF, rrf_k=10),
    RetrievalPolicy(name="rrf_k30", strategy=RetrievalStrategy.RRF, rrf_k=30),
    RetrievalPolicy(name="rrf_k60", strategy=RetrievalStrategy.RRF, rrf_k=60),
    RetrievalPolicy(name="rrf_k90", strategy=RetrievalStrategy.RRF, rrf_k=90),
    RetrievalPolicy(name="weighted_rrf_k60", strategy=RetrievalStrategy.WEIGHTED_RRF),
    RetrievalPolicy(
        name="production_weighted_rrf_k60",
        strategy=RetrievalStrategy.WEIGHTED_RRF,
        enforce_route_coverage=True,
    ),
    RetrievalPolicy(name="normalized_score", strategy=RetrievalStrategy.NORMALIZED_SCORE),
)


class RetrievalQueryScore(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: str
    case_id: str
    query: str
    algorithm: str
    rrf_k: int
    retrieved_source_keys: tuple[str, ...]
    relevant_source_keys: tuple[str, ...]
    metrics: RetrievalMetrics


class RetrievalPolicySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy: str
    query_count: int = Field(ge=1)
    mean_precision_at_k: float = Field(ge=0, le=1)
    mean_recall_at_k: float = Field(ge=0, le=1)
    mean_reciprocal_rank: float = Field(ge=0, le=1)
    mean_ndcg_at_k: float = Field(ge=0, le=1)
    mean_source_coverage: float = Field(ge=0, le=1)


class PairwiseComparison(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    policy_a: str
    policy_b: str
    metric: str
    mean_delta_b_minus_a: float
    wins_b: int = Field(ge=0)
    ties: int = Field(ge=0)
    losses_b: int = Field(ge=0)


class RetrievalExperimentReport(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    dataset_id: str
    dataset_version: str
    dataset_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    top_k: int
    policies: tuple[RetrievalPolicy, ...]
    summaries: tuple[RetrievalPolicySummary, ...]
    query_scores: tuple[RetrievalQueryScore, ...]
    pairwise: tuple[PairwiseComparison, ...]


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _metric(score: RetrievalQueryScore, name: str) -> float:
    return float(getattr(score.metrics, name))


def run_experiment(
    retrieval: HybridRetrieval,
    benchmark: RetrievalBenchmark,
    policies: Sequence[RetrievalPolicy] = DEFAULT_POLICIES,
) -> RetrievalExperimentReport:
    """Execute every policy over the same frozen queries and knowledge snapshot."""

    query_scores: list[RetrievalQueryScore] = []
    for policy in policies:
        for case in benchmark.cases:
            result = retrieval.execute(
                case.query,
                top_k=benchmark.top_k,
                min_relevance=0.0,
                strategy=policy.strategy,
                rrf_k=policy.rrf_k,
                route_weights=policy.weights,
                enforce_route_coverage=policy.enforce_route_coverage,
            )
            query_scores.append(
                RetrievalQueryScore(
                    policy=policy.name,
                    case_id=case.case_id,
                    query=case.query,
                    algorithm=result.algorithm,
                    rrf_k=result.rrf_k,
                    retrieved_source_keys=tuple(
                        evidence_source_key(hit.evidence.source_kind, hit.evidence.source_ref)
                        for hit in result.hits
                    ),
                    relevant_source_keys=tuple(sorted(case.relevant_source_keys)),
                    metrics=evaluate_retrieval_sources(result, case.relevant_source_keys),
                )
            )

    metrics = (
        "precision_at_k",
        "recall_at_k",
        "reciprocal_rank",
        "ndcg_at_k",
        "source_coverage",
    )
    by_policy = {
        policy.name: tuple(score for score in query_scores if score.policy == policy.name)
        for policy in policies
    }
    summaries = tuple(
        RetrievalPolicySummary(
            policy=policy.name,
            query_count=len(by_policy[policy.name]),
            mean_precision_at_k=_mean(
                [_metric(score, "precision_at_k") for score in by_policy[policy.name]]
            ),
            mean_recall_at_k=_mean(
                [_metric(score, "recall_at_k") for score in by_policy[policy.name]]
            ),
            mean_reciprocal_rank=_mean(
                [_metric(score, "reciprocal_rank") for score in by_policy[policy.name]]
            ),
            mean_ndcg_at_k=_mean(
                [_metric(score, "ndcg_at_k") for score in by_policy[policy.name]]
            ),
            mean_source_coverage=_mean(
                [_metric(score, "source_coverage") for score in by_policy[policy.name]]
            ),
        )
        for policy in policies
    )
    pairwise = []
    for left, right in itertools.combinations(policies, 2):
        for metric in metrics:
            deltas = [
                _metric(right_score, metric) - _metric(left_score, metric)
                for left_score, right_score in zip(
                    by_policy[left.name], by_policy[right.name], strict=True
                )
            ]
            pairwise.append(
                PairwiseComparison(
                    policy_a=left.name,
                    policy_b=right.name,
                    metric=metric,
                    mean_delta_b_minus_a=_mean(deltas),
                    wins_b=sum(delta > 1e-12 for delta in deltas),
                    ties=sum(abs(delta) <= 1e-12 for delta in deltas),
                    losses_b=sum(delta < -1e-12 for delta in deltas),
                )
            )
    return RetrievalExperimentReport(
        dataset_id=benchmark.dataset_id,
        dataset_version=benchmark.version,
        dataset_hash=benchmark.content_hash,
        top_k=benchmark.top_k,
        policies=tuple(policies),
        summaries=summaries,
        query_scores=tuple(query_scores),
        pairwise=tuple(pairwise),
    )


def _write_csv(path: Path, rows: Sequence[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"cannot write empty experiment table: {path.name}")
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_report(report: RetrievalExperimentReport, output_dir: Path) -> None:
    """Write JSON plus spreadsheet-friendly UTF-8 CSV experiment artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "summary.json").write_text(
        report.model_dump_json(indent=2), encoding="utf-8"
    )
    _write_csv(
        output_dir / "strategy_summary.csv",
        [summary.model_dump(mode="json") for summary in report.summaries],
    )
    _write_csv(
        output_dir / "per_query.csv",
        [
            {
                "policy": score.policy,
                "case_id": score.case_id,
                "query": score.query,
                "algorithm": score.algorithm,
                "rrf_k": score.rrf_k,
                **score.metrics.model_dump(mode="json"),
                "retrieved_source_keys": " | ".join(score.retrieved_source_keys),
                "relevant_source_keys": " | ".join(score.relevant_source_keys),
            }
            for score in report.query_scores
        ],
    )
    _write_csv(
        output_dir / "pairwise.csv",
        [comparison.model_dump(mode="json") for comparison in report.pairwise],
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="aerodiagnosis-retrieval-experiment",
        description="Compare single-route retrieval, RRF variants and production fusion.",
    )
    parser.add_argument("benchmark", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument(
        "--runtime-dir",
        type=Path,
        help="Use an explicit runtime snapshot; omitted uses an isolated seeded runtime.",
    )
    return parser


def _run_with_runtime(
    benchmark: RetrievalBenchmark,
    output_dir: Path,
    runtime_dir: Path,
) -> RetrievalExperimentReport:
    settings = RuntimeSettings(
        runtime_dir=runtime_dir,
        database_path=runtime_dir / "data" / "retrieval-experiment.db",
        seed_demo_content=True,
    )
    report = run_experiment(bootstrap(settings).hybrid_retrieval, benchmark)
    write_report(report, output_dir)
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        benchmark = RetrievalBenchmark.model_validate_json(
            args.benchmark.read_text(encoding="utf-8")
        )
        def runner(runtime: Path) -> RetrievalExperimentReport:
            return _run_with_runtime(benchmark, args.output_dir, runtime)
        if args.runtime_dir is not None:
            report = runner(args.runtime_dir)
        else:
            with tempfile.TemporaryDirectory(prefix="aerodiagnosis-retrieval-") as temporary:
                report = runner(Path(temporary))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        print(json.dumps({"error": str(exc)}, ensure_ascii=False), file=sys.stderr)
        return 1
    print(
        json.dumps(
            {
                "dataset": report.dataset_id,
                "dataset_hash": report.dataset_hash,
                "query_count": len(benchmark.cases),
                "policy_count": len(report.policies),
                "output_dir": str(args.output_dir.resolve()),
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
