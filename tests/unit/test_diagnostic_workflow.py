from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aerodiagnosis.adapters.persistence import (
    SQLiteCaseStore,
    SQLiteCheckpointStore,
    SQLiteConversationMemory,
)
from aerodiagnosis.adapters.persistence.sqlite_graph import SQLiteGraphStore
from aerodiagnosis.adapters.persistence.sqlite_operation_ledger import (
    SQLiteOperationLedger,
)
from aerodiagnosis.adapters.persistence.sqlite_vector import SQLiteVectorStore
from aerodiagnosis.application.diagnostic_workflow import (
    STATE_SCHEMA_VERSION,
    WORKFLOW_VERSION,
    DiagnosisRunNotFound,
    DiagnosisWorkflowState,
    DiagnosticWorkflow,
    IncompatibleWorkflowError,
)
from aerodiagnosis.bootstrap import bootstrap
from aerodiagnosis.config import RuntimeSettings
from aerodiagnosis.domain import (
    DiagnosisCommand,
    DiagnosisReportStatus,
    ParameterObservation,
)
from aerodiagnosis.tools import DiagnosticToolset


class FakeLanguageModel:
    identity = "fake:diagnostic-model"

    def __init__(self, *, verification_passes: bool = True) -> None:
        self.verification_passes = verification_passes
        self.calls: list[str] = []

    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        self.calls.append(task)
        if task == "plan_evidence":
            return {
                "search_queries": [str(payload["question"])],
                "tools": ["search_manual_chunks"],
                "rationale": "Search the active manual snapshot.",
            }
        if task == "repair_evidence_plan":
            return {
                "search_queries": [f"{payload['question']} maintenance cause"],
                "tools": ["search_manual_chunks"],
                "rationale": "Change the query after insufficient evidence.",
            }
        if task == "generate_diagnosis":
            evidence = payload["evidence"]
            return {
                "summary": "The retrieved evidence supports a compressor-stall candidate.",
                "claims": [
                    {
                        "statement": "Compressor stall can cause exhaust temperature rise.",
                        "evidence_ids": [evidence[0]["evidence_id"]],
                        "confidence": 0.8,
                    }
                ],
            }
        if task == "verify_diagnosis":
            if self.verification_passes:
                return {
                    "sufficient": True,
                    "supported_claim_indexes": [0],
                    "issues": [],
                    "corrected_queries": [],
                    "corrected_tools": [],
                }
            return {
                "sufficient": False,
                "supported_claim_indexes": [],
                "issues": ["The causal link needs another source."],
                "corrected_queries": ["compressor stall causal evidence"],
                "corrected_tools": ["traverse_fault_graph"],
            }
        raise AssertionError(f"unexpected task: {task}")


class MultiToolLanguageModel(FakeLanguageModel):
    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if task == "plan_evidence":
            self.calls.append(task)
            return {
                "search_queries": ["compressor EGT case cause"],
                "tools": [
                    "search_manual_chunks",
                    "traverse_fault_graph",
                    "find_similar_cases",
                    "analyze_gas_path_parameters",
                ],
                "rationale": "Combine all evidence types selected by the model.",
            }
        return super().complete_json(task, payload)


class RedundantHybridLanguageModel(FakeLanguageModel):
    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if task == "plan_evidence":
            self.calls.append(task)
            return {
                "search_queries": ["compressor stall EGT"],
                "tools": [
                    "hybrid_retrieve_evidence",
                    "search_manual_chunks",
                    "traverse_fault_graph",
                    "analyze_gas_path_parameters",
                ],
                "rationale": "The model redundantly selected hybrid and its subroutes.",
            }
        return super().complete_json(task, payload)


def _settings(tmp_path: Path) -> RuntimeSettings:
    return RuntimeSettings(runtime_dir=tmp_path, database_path=tmp_path / "runtime.db")


def _workflow(settings: RuntimeSettings, model: FakeLanguageModel) -> DiagnosticWorkflow:
    memory = SQLiteConversationMemory(settings.database_path)
    tools = DiagnosticToolset(
        vector_store=SQLiteVectorStore(settings.database_path),
        graph_store=SQLiteGraphStore(settings.database_path),
        case_store=SQLiteCaseStore(settings.database_path),
    )
    return DiagnosticWorkflow(
        tools=tools,
        language_model=model,
        active_versions=frozenset,
        checkpoints=SQLiteCheckpointStore(settings.database_path),
        memory=memory,
        ledger=SQLiteOperationLedger(settings.database_path),
    )


def test_workflow_uses_llm_planner_generator_verifier_and_persists_memory(
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    application = bootstrap(settings)
    session_id = application.conversation_sessions.create()
    application.ingest_document.execute(
        display_name="manual.txt",
        content=b"Compressor stall may cause an exhaust gas temperature rise.",
    )
    model = FakeLanguageModel()

    report = application.run_diagnosis.start(
        DiagnosisCommand(
            session_id=session_id,
            question="Does compressor stall cause temperature rise?",
            min_relevance=0,
        ),
        model,
    )
    resumed = application.run_diagnosis.resume(report.run_id, model)
    messages = application.conversation_sessions.messages(session_id)

    assert report.status is DiagnosisReportStatus.EVIDENCE_READY
    assert report.claims[0].verification_method == "llm_verifier@1"
    assert model.calls == ["plan_evidence", "generate_diagnosis", "verify_diagnosis"]
    assert resumed == report
    assert [message.role for message in messages] == ["user", "assistant"]


def test_workflow_records_model_and_tool_operations(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    ledger = SQLiteOperationLedger(settings.database_path)
    application = bootstrap(settings)
    session_id = application.conversation_sessions.create()
    application.ingest_document.execute(
        display_name="manual.txt",
        content=b"Compressor stall may cause an exhaust gas temperature rise.",
    )
    model = FakeLanguageModel()

    report = application.run_diagnosis.start(
        DiagnosisCommand(
            session_id=session_id,
            question="Does compressor stall cause temperature rise?",
            min_relevance=0,
        ),
        model,
    )
    operations = ledger.list_run(report.run_id)

    assert [item.kind for item in operations] == [
        "model",
        "tool",
        "model",
        "model",
    ]
    assert [item.name for item in operations] == [
        "plan_evidence",
        "search_manual_chunks",
        "generate_diagnosis",
        "verify_diagnosis",
    ]


def test_streaming_workflow_emits_agent_events(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    application = bootstrap(settings)
    session_id = application.conversation_sessions.create()
    application.ingest_document.execute(
        display_name="manual.txt",
        content=b"Compressor stall may cause an exhaust gas temperature rise.",
    )
    events: list[tuple[str, object]] = []

    report = application.run_diagnosis.start_streaming(
        DiagnosisCommand(
            session_id=session_id,
            question="Does compressor stall cause temperature rise?",
            min_relevance=0,
        ),
        FakeLanguageModel(),
        lambda event, payload: events.append((event, payload)),
    )

    assert report.status is DiagnosisReportStatus.EVIDENCE_READY
    event_names = [event for event, _payload in events]
    assert "run_started" in event_names
    assert "model_call_started" in event_names
    assert "tool_call_started" in event_names
    assert "report_ready" in event_names


def test_workflow_changes_plan_then_refuses_when_no_evidence_exists(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))
    session_id = application.conversation_sessions.create()
    model = FakeLanguageModel()

    report = application.run_diagnosis.start(
        DiagnosisCommand(session_id=session_id, question="Why did the engine surge?"),
        model,
    )

    assert report.status is DiagnosisReportStatus.INSUFFICIENT_EVIDENCE
    assert report.claims == ()
    assert len(report.attempted_queries) == 2
    assert report.attempted_queries[0] != report.attempted_queries[1]
    assert model.calls == ["plan_evidence", "repair_evidence_plan"]


def test_failed_verifier_forces_replanning_and_eventual_refusal(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))
    session_id = application.conversation_sessions.create()
    application.ingest_document.execute(
        display_name="manual.md",
        content=b"Compressor stall and temperature rise require causal confirmation.",
    )
    model = FakeLanguageModel(verification_passes=False)

    report = application.run_diagnosis.start(
        DiagnosisCommand(
            session_id=session_id,
            question="Does compressor stall cause temperature rise?",
            min_relevance=0,
        ),
        model,
    )

    assert report.status is DiagnosisReportStatus.INSUFFICIENT_EVIDENCE
    assert report.claims == ()
    assert "another source" in report.refusal_reason
    assert "repair_evidence_plan" in model.calls
    assert model.calls.count("verify_diagnosis") == 2


def test_non_terminal_checkpoint_can_resume_with_frozen_versions(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    application = bootstrap(settings)
    session_id = application.conversation_sessions.create()
    ingested = application.ingest_document.execute(
        display_name="manual.md",
        content=b"Fuel nozzle blockage can produce an uneven temperature pattern.",
    )
    state = DiagnosisWorkflowState(
        run_id="interrupted-run",
        snapshot_id="e" * 64,
        active_version_ids=(ingested.version_id,),
        model_identity=FakeLanguageModel.identity,
        command=DiagnosisCommand(
            session_id=session_id,
            question="What produces an uneven temperature pattern?",
            min_relevance=0,
        ),
    )
    store = SQLiteCheckpointStore(settings.database_path)
    store.save(
        run_id=state.run_id,
        workflow_version=WORKFLOW_VERSION,
        state_schema_version=STATE_SCHEMA_VERSION,
        snapshot_id=state.snapshot_id,
        state_json=state.model_dump_json(),
    )
    model = FakeLanguageModel()
    workflow = _workflow(settings, model)

    report = workflow.resume(state.run_id)

    assert report.status is DiagnosisReportStatus.EVIDENCE_READY
    assert report.evidence[0].version_id == ingested.version_id


def test_resume_rejects_missing_incompatible_or_different_model(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    memory = SQLiteConversationMemory(settings.database_path)
    memory.create_session("session")
    store = SQLiteCheckpointStore(settings.database_path)
    workflow = _workflow(settings, FakeLanguageModel())

    with pytest.raises(DiagnosisRunNotFound, match="unknown diagnosis run"):
        workflow.resume("missing")

    state = DiagnosisWorkflowState(
        run_id="old-run",
        snapshot_id="f" * 64,
        active_version_ids=(),
        model_identity=FakeLanguageModel.identity,
        command=DiagnosisCommand(session_id="session", question="Why did the engine surge?"),
    )
    store.save(
        run_id=state.run_id,
        workflow_version="old-workflow",
        state_schema_version=STATE_SCHEMA_VERSION,
        snapshot_id=state.snapshot_id,
        state_json=state.model_dump_json(),
    )
    with pytest.raises(IncompatibleWorkflowError, match="incompatible"):
        workflow.resume(state.run_id)

    different = FakeLanguageModel()
    different.identity = "fake:different-model"
    current = state.model_copy(update={"run_id": "current-run"})
    store.save(
        run_id=current.run_id,
        workflow_version=WORKFLOW_VERSION,
        state_schema_version=STATE_SCHEMA_VERSION,
        snapshot_id=current.snapshot_id,
        state_json=current.model_dump_json(),
    )
    with pytest.raises(IncompatibleWorkflowError, match="same language-model"):
        _workflow(settings, different).resume(current.run_id)


def test_model_can_route_to_all_four_tools_and_parameter_evidence_completes_run(
    tmp_path: Path,
) -> None:
    application = bootstrap(_settings(tmp_path))
    session_id = application.conversation_sessions.create()
    model = MultiToolLanguageModel()

    report = application.run_diagnosis.start(
        DiagnosisCommand(
            session_id=session_id,
            question="Analyze this EGT value with manuals, graph and cases.",
            parameters=(
                ParameterObservation(
                    name="EGT",
                    value=760,
                    expected_min=600,
                    expected_max=720,
                    unit="C",
                ),
            ),
        ),
        model,
    )

    assert report.status is DiagnosisReportStatus.EVIDENCE_READY
    assert {execution.tool_name for execution in report.tool_executions} == {
        "search_manual_chunks",
        "traverse_fault_graph",
        "find_similar_cases",
        "analyze_gas_path_parameters",
    }
    assert report.evidence[0].source_kind == "parameter_analysis"


def test_tool_call_budget_stops_oversized_model_plan(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))
    session_id = application.conversation_sessions.create()
    model = MultiToolLanguageModel()

    report = application.run_diagnosis.start(
        DiagnosisCommand(
            session_id=session_id,
            question="Search everything for this unknown fault.",
            max_tool_calls=2,
        ),
        model,
    )

    assert report.status is DiagnosisReportStatus.INSUFFICIENT_EVIDENCE
    assert len(report.tool_executions) == 2


def test_hybrid_plan_removes_redundant_subroute_calls(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))
    session_id = application.conversation_sessions.create()
    model = RedundantHybridLanguageModel()

    report = application.run_diagnosis.start(
        DiagnosisCommand(
            session_id=session_id,
            question="Analyze compressor stall with all available knowledge.",
            parameters=(
                ParameterObservation(
                    name="EGT",
                    value=760,
                    expected_min=600,
                    expected_max=720,
                    unit="C",
                ),
            ),
        ),
        model,
    )

    assert report.status is DiagnosisReportStatus.EVIDENCE_READY
    assert {execution.tool_name for execution in report.tool_executions} == {
        "hybrid_retrieve_evidence",
        "analyze_gas_path_parameters",
    }


def test_hybrid_plan_skips_parameter_tool_without_parameters(tmp_path: Path) -> None:
    application = bootstrap(_settings(tmp_path))
    session_id = application.conversation_sessions.create()
    application.ingest_document.execute(
        display_name="manual.txt",
        content=b"Compressor stall can cause an exhaust gas temperature rise.",
    )

    report = application.run_diagnosis.start(
        DiagnosisCommand(
            session_id=session_id,
            question="Analyze compressor stall with all available knowledge.",
        ),
        RedundantHybridLanguageModel(),
    )

    assert report.status is DiagnosisReportStatus.EVIDENCE_READY
    assert {execution.tool_name for execution in report.tool_executions} == {
        "hybrid_retrieve_evidence"
    }
