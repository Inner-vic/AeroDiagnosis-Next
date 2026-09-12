from __future__ import annotations

import json
from pathlib import Path

from aerodiagnosis.bootstrap import Application, bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.retrieval_experiment import (
    DEFAULT_POLICIES,
    RetrievalBenchmark,
    main,
    run_experiment,
    write_report,
)


def _application(tmp_path: Path) -> Application:
    runtime = tmp_path / "runtime"
    return bootstrap(
        RuntimeSettings(
            runtime_dir=runtime,
            database_path=runtime / "data" / "experiment.db",
            seed_demo_content=True,
        )
    )


def _benchmark() -> RetrievalBenchmark:
    path = Path(__file__).parents[2] / "evaluation" / "retrieval" / "benchmark-v1.json"
    return RetrievalBenchmark.model_validate_json(path.read_text(encoding="utf-8"))


def test_experiment_compares_single_routes_rrf_and_production_policy(tmp_path: Path) -> None:
    benchmark = _benchmark()
    report = run_experiment(_application(tmp_path).hybrid_retrieval, benchmark)

    assert len(report.summaries) == len(DEFAULT_POLICIES)
    assert len(report.query_scores) == len(DEFAULT_POLICIES) * 4
    assert {summary.policy for summary in report.summaries} >= {
        "manual_only",
        "graph_only",
        "case_only",
        "rrf_k10",
        "rrf_k60",
        "production_weighted_rrf_k60",
    }
    assert report.dataset_hash == benchmark.content_hash
    rebuilt = benchmark.model_copy(
        update={
            "cases": tuple(
                case.model_copy(
                    update={
                        "relevant_source_keys": frozenset(
                            reversed(sorted(case.relevant_source_keys))
                        )
                    }
                )
                for case in benchmark.cases
            )
        }
    )
    assert rebuilt.content_hash == benchmark.content_hash
    assert report.pairwise


def test_experiment_writes_json_and_thesis_ready_csv(tmp_path: Path) -> None:
    report = run_experiment(
        _application(tmp_path).hybrid_retrieval,
        _benchmark(),
        DEFAULT_POLICIES[:4],
    )
    output = tmp_path / "results"
    write_report(report, output)

    assert json.loads((output / "summary.json").read_text(encoding="utf-8"))["dataset_hash"]
    assert "mean_ndcg_at_k" in (output / "strategy_summary.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "retrieved_source_keys" in (output / "per_query.csv").read_text(
        encoding="utf-8-sig"
    )
    assert "mean_delta_b_minus_a" in (output / "pairwise.csv").read_text(
        encoding="utf-8-sig"
    )


def test_retrieval_experiment_cli_uses_explicit_runtime(tmp_path: Path) -> None:
    benchmark = Path(__file__).parents[2] / "evaluation" / "retrieval" / "benchmark-v1.json"
    output = tmp_path / "cli-results"

    result = main(
        [str(benchmark), str(output), "--runtime-dir", str(tmp_path / "cli-runtime")]
    )

    assert result == 0
    assert (output / "per_query.csv").is_file()
