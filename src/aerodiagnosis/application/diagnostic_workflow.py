"""LLM-directed, recoverable and fail-closed diagnostic workflow."""

from __future__ import annotations

import hashlib
import json
import uuid
from collections.abc import Callable
from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from aerodiagnosis.application.agent_models import (
    CandidateDiagnosis,
    EvidencePlan,
    MemoryTurn,
    VerificationDecision,
)
from aerodiagnosis.domain import (
    DiagnosisCommand,
    DiagnosisReport,
    DiagnosisReportStatus,
    EvidenceClaim,
    EvidenceItem,
    ToolExecution,
    ToolExecutionStatus,
)
from aerodiagnosis.ports import (
    CheckpointStore,
    ConversationMemoryStore,
    LanguageModel,
    LanguageModelError,
    MessageRole,
    OperationLedger,
)
from aerodiagnosis.tools.diagnostic import (
    CASE_TOOL,
    GRAPH_TOOL,
    HYBRID_TOOL,
    MANUAL_TOOL,
    PARAMETER_TOOL,
    DiagnosticToolset,
)

WORKFLOW_VERSION = "llm-evidence-workflow@3"
STATE_SCHEMA_VERSION = "2.0"
MAX_REASONING_ROUNDS = 2
AVAILABLE_TOOLS = (HYBRID_TOOL, MANUAL_TOOL, GRAPH_TOOL, CASE_TOOL, PARAMETER_TOOL)
HYBRID_SUBROUTES = frozenset({MANUAL_TOOL, GRAPH_TOOL, CASE_TOOL})


class WorkflowStage(StrEnum):
    CREATED = "created"
    PLANNING = "planning"
    RETRIEVING = "retrieving"
    ASSESSING = "assessing"
    GENERATING = "generating"
    VERIFYING = "verifying"
    EVIDENCE_READY = "evidence_ready"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class DiagnosisWorkflowState(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    run_id: str = Field(min_length=1)
    workflow_version: str = WORKFLOW_VERSION
    state_schema_version: str = STATE_SCHEMA_VERSION
    snapshot_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    active_version_ids: tuple[str, ...]
    model_identity: str = Field(min_length=1)
    command: DiagnosisCommand
    memory_context: tuple[MemoryTurn, ...] = ()
    stage: WorkflowStage = WorkflowStage.CREATED
    plans: tuple[EvidencePlan, ...] = ()
    evidence: tuple[EvidenceItem, ...] = ()
    tool_executions: tuple[ToolExecution, ...] = ()
    candidate: CandidateDiagnosis | None = None
    verification: VerificationDecision | None = None
    report: DiagnosisReport | None = None

    @model_validator(mode="after")
    def terminal_state_requires_matching_report(self) -> Self:
        terminal = {
            WorkflowStage.EVIDENCE_READY,
            WorkflowStage.INSUFFICIENT_EVIDENCE,
        }
        if self.stage in terminal and self.report is None:
            raise ValueError("terminal workflow states require a report")
        if self.stage not in terminal and self.report is not None:
            raise ValueError("non-terminal workflow states cannot contain a report")
        return self


class DiagnosisRunNotFound(KeyError):
    pass


class IncompatibleWorkflowError(RuntimeError):
    pass


def _snapshot_id(
    active_version_ids: tuple[str, ...],
    tool_backends: tuple[str, ...],
    model_identity: str,
) -> str:
    payload = {
        "active_version_ids": active_version_ids,
        "tool_backends": tool_backends,
        "model_identity": model_identity,
        "workflow_version": WORKFLOW_VERSION,
        "state_schema_version": STATE_SCHEMA_VERSION,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


class DiagnosticWorkflow:
    def __init__(
        self,
        *,
        tools: DiagnosticToolset,
        language_model: LanguageModel,
        active_versions: Callable[[], frozenset[str]],
        checkpoints: CheckpointStore,
        memory: ConversationMemoryStore,
        ledger: OperationLedger,
    ) -> None:
        self._tools = tools
        self._model = language_model
        self._active_versions = active_versions
        self._checkpoints = checkpoints
        self._memory = memory
        self._ledger = ledger

    def start(self, command: DiagnosisCommand) -> DiagnosisReport:
        active_versions = tuple(sorted(self._active_versions()))
        run_id = str(uuid.uuid4())
        self._memory.append(
            session_id=command.session_id,
            message_id=self._memory_message_id(run_id, MessageRole.USER),
            role=MessageRole.USER,
            content=command.question,
        )
        memory_context = tuple(
            MemoryTurn(role=message.role, content=message.content)
            for message in self._memory.recent(command.session_id)
        )
        state = DiagnosisWorkflowState(
            run_id=run_id,
            snapshot_id=_snapshot_id(
                active_versions,
                self._tools.backend_identity,
                self._model.identity,
            ),
            active_version_ids=active_versions,
            model_identity=self._model.identity,
            command=command,
            memory_context=memory_context,
        )
        self._save(state)
        return self._execute(state)

    def resume(self, run_id: str) -> DiagnosisReport:
        checkpoint = self._checkpoints.latest(run_id)
        if checkpoint is None:
            raise DiagnosisRunNotFound(f"unknown diagnosis run: {run_id}")
        if (
            checkpoint.workflow_version != WORKFLOW_VERSION
            or checkpoint.state_schema_version != STATE_SCHEMA_VERSION
        ):
            raise IncompatibleWorkflowError("checkpoint workflow version is incompatible")
        state = DiagnosisWorkflowState.model_validate_json(checkpoint.state_json)
        if state.snapshot_id != checkpoint.snapshot_id:
            raise IncompatibleWorkflowError("checkpoint snapshot identity does not match state")
        if state.model_identity != self._model.identity:
            raise IncompatibleWorkflowError("resume requires the same language-model identity")
        if state.report is not None:
            return state.report
        return self._execute(state)

    @staticmethod
    def _memory_message_id(run_id: str, role: MessageRole) -> str:
        return hashlib.sha256(f"{run_id}:{role}".encode()).hexdigest()

    @staticmethod
    def _memory_content(report: DiagnosisReport) -> str:
        return json.dumps(
            {
                "run_id": report.run_id,
                "status": report.status,
                "summary": report.summary,
                "refusal_reason": report.refusal_reason,
                "claims": [
                    {
                        "statement": claim.statement,
                        "evidence_ids": claim.evidence_ids,
                    }
                    for claim in report.claims
                ],
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )

    def _save(self, state: DiagnosisWorkflowState) -> None:
        self._checkpoints.save(
            run_id=state.run_id,
            workflow_version=state.workflow_version,
            state_schema_version=state.state_schema_version,
            snapshot_id=state.snapshot_id,
            state_json=state.model_dump_json(),
        )

    @staticmethod
    def _updated(state: DiagnosisWorkflowState, **updates: Any) -> DiagnosisWorkflowState:
        payload = state.model_dump(mode="python")
        payload.update(updates)
        return DiagnosisWorkflowState.model_validate(payload)

    def _model_output(
        self,
        run_id: str,
        task: str,
        payload: dict[str, Any],
        model: type[BaseModel],
    ) -> BaseModel:
        try:
            raw = self._model.complete_json(task, payload)
        except Exception as exc:
            self._ledger.record_model(
                run_id=run_id,
                task=task,
                payload=payload,
                result={},
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        try:
            output = model.model_validate(raw)
        except ValidationError as exc:
            self._ledger.record_model(
                run_id=run_id,
                task=task,
                payload=payload,
                result=dict(raw) if isinstance(raw, dict) else {},
                error=str(exc),
            )
            raise LanguageModelError(f"invalid structured output for {task}") from exc
        self._ledger.record_model(
            run_id=run_id,
            task=task,
            payload=payload,
            result=output.model_dump(mode="json"),
        )
        return output

    def _execute(self, state: DiagnosisWorkflowState) -> DiagnosisReport:
        while True:
            if state.stage is WorkflowStage.CREATED:
                state = self._plan(state)
            elif state.stage in {WorkflowStage.PLANNING, WorkflowStage.RETRIEVING}:
                state = self._retrieve(state)
            elif state.stage is WorkflowStage.ASSESSING:
                if state.evidence and state.verification is None:
                    state = self._generate(state)
                elif len(state.plans) < MAX_REASONING_ROUNDS:
                    state = self._plan(state)
                else:
                    return self._refuse(state)
            elif state.stage is WorkflowStage.GENERATING:
                state = self._verify(state)
            elif state.stage is WorkflowStage.VERIFYING:
                if state.verification is not None and state.verification.sufficient:
                    report = self._approved_report(state)
                    if report is not None:
                        return self._complete(state, report)
                if len(state.plans) < MAX_REASONING_ROUNDS:
                    state = self._updated(
                        state,
                        stage=WorkflowStage.ASSESSING,
                        candidate=None,
                    )
                    self._save(state)
                else:
                    return self._refuse(state)
            elif state.report is not None:
                return state.report
            else:  # pragma: no cover - enum exhaustiveness guard
                raise RuntimeError(f"unsupported workflow stage: {state.stage}")

    def _plan(self, state: DiagnosisWorkflowState) -> DiagnosisWorkflowState:
        task = "plan_evidence" if not state.plans else "repair_evidence_plan"
        payload = {
            "question": state.command.question,
            "parameters": [item.model_dump(mode="json") for item in state.command.parameters],
            "available_tools": AVAILABLE_TOOLS,
            "retrieval_policy": (
                f"Prefer {HYBRID_TOOL} when evidence may span manuals, graph paths and cases. "
                "Never combine it with its manual, graph or case subroutes in the same plan. "
                "Use a single-route tool only when the question explicitly requires that source. "
                f"Select {PARAMETER_TOOL} only when parameters are supplied."
            ),
            "previous_plans": [plan.model_dump(mode="json") for plan in state.plans],
            "verification_issues": (
                list(state.verification.issues) if state.verification is not None else []
            ),
            "evidence_ids": [item.evidence_id for item in state.evidence],
            "conversation": [item.model_dump(mode="json") for item in state.memory_context],
        }
        plan = self._model_output(state.run_id, task, payload, EvidencePlan)
        assert isinstance(plan, EvidencePlan)
        unknown = set(plan.tools) - set(AVAILABLE_TOOLS)
        if unknown:
            raise LanguageModelError(f"planner selected unknown tools: {sorted(unknown)}")
        normalized_tools = plan.tools
        if HYBRID_TOOL in normalized_tools:
            normalized_tools = tuple(
                tool for tool in normalized_tools if tool not in HYBRID_SUBROUTES
            )
        if not state.command.parameters:
            normalized_tools = tuple(
                tool for tool in normalized_tools if tool != PARAMETER_TOOL
            )
        if not normalized_tools:
            raise LanguageModelError("planner selected no applicable evidence tools")
        if normalized_tools != plan.tools:
            plan = plan.model_copy(update={"tools": normalized_tools})
        if state.plans and plan == state.plans[-1]:
            raise LanguageModelError("repair plan did not change retrieval actions")
        planned = self._updated(
            state,
            stage=WorkflowStage.PLANNING,
            plans=(*state.plans, plan),
            verification=None,
        )
        self._save(planned)
        return planned

    def _retrieve(self, state: DiagnosisWorkflowState) -> DiagnosisWorkflowState:
        retrieving = self._updated(state, stage=WorkflowStage.RETRIEVING)
        self._save(retrieving)
        plan = retrieving.plans[-1]
        evidence = {item.evidence_id: item for item in retrieving.evidence}
        trace = list(retrieving.tool_executions)
        active_versions = frozenset(retrieving.active_version_ids)
        budget_exhausted = False
        for query in plan.search_queries:
            for tool_name in plan.tools:
                if len(trace) >= retrieving.command.max_tool_calls:
                    budget_exhausted = True
                    break
                try:
                    items = self._tools.execute(
                        tool_name,
                        command=retrieving.command,
                        query=query,
                        active_version_ids=active_versions,
                    )
                except Exception as exc:  # tool failures are recorded and remain fail-closed
                    call_id = hashlib.sha256(
                        f"{tool_name}:{query}:{len(trace)}".encode()
                    ).hexdigest()
                    self._ledger.record_tool(
                        run_id=retrieving.run_id,
                        call_id=call_id,
                        tool_name=tool_name,
                        query=query,
                        status=ToolExecutionStatus.ERROR.value,
                        evidence_count=0,
                        error=f"{type(exc).__name__}: {exc}",
                    )
                    trace.append(
                        ToolExecution(
                            tool_name=tool_name,
                            query=query,
                            status=ToolExecutionStatus.ERROR,
                            evidence_count=0,
                            error=f"{type(exc).__name__}: {exc}",
                        )
                    )
                    continue
                call_id = hashlib.sha256(
                    f"{tool_name}:{query}:{len(trace)}".encode()
                ).hexdigest()
                execution_status = (
                    ToolExecutionStatus.OK if items else ToolExecutionStatus.EMPTY
                )
                self._ledger.record_tool(
                    run_id=retrieving.run_id,
                    call_id=call_id,
                    tool_name=tool_name,
                    query=query,
                    status=execution_status.value,
                    evidence_count=len(items),
                )
                trace.append(
                    ToolExecution(
                        tool_name=tool_name,
                        query=query,
                        status=execution_status,
                        evidence_count=len(items),
                    )
                )
                for item in items:
                    evidence.setdefault(item.evidence_id, item)
            if budget_exhausted:
                break
        assessed = self._updated(
            retrieving,
            stage=WorkflowStage.ASSESSING,
            evidence=tuple(
                sorted(evidence.values(), key=lambda item: (-item.score, item.evidence_id))
            ),
            tool_executions=tuple(trace),
        )
        self._save(assessed)
        return assessed

    def _generate(self, state: DiagnosisWorkflowState) -> DiagnosisWorkflowState:
        output = self._model_output(
            state.run_id,
            "generate_diagnosis",
            {
                "question": state.command.question,
                "evidence": [item.model_dump(mode="json") for item in state.evidence],
                "conversation": [item.model_dump(mode="json") for item in state.memory_context],
            },
            CandidateDiagnosis,
        )
        assert isinstance(output, CandidateDiagnosis)
        generated = self._updated(
            state,
            stage=WorkflowStage.GENERATING,
            candidate=output,
        )
        self._save(generated)
        return generated

    def _verify(self, state: DiagnosisWorkflowState) -> DiagnosisWorkflowState:
        if state.candidate is None:  # pragma: no cover - guarded by stage transition
            raise RuntimeError("candidate is required before verification")
        output = self._model_output(
            state.run_id,
            "verify_diagnosis",
            {
                "question": state.command.question,
                "candidate": state.candidate.model_dump(mode="json"),
                "evidence": [item.model_dump(mode="json") for item in state.evidence],
            },
            VerificationDecision,
        )
        assert isinstance(output, VerificationDecision)
        verified = self._updated(
            state,
            stage=WorkflowStage.VERIFYING,
            verification=output,
        )
        self._save(verified)
        return verified

    @staticmethod
    def _approved_report(state: DiagnosisWorkflowState) -> DiagnosisReport | None:
        if state.candidate is None or state.verification is None:
            return None
        available = {item.evidence_id for item in state.evidence}
        claims = []
        for index in state.verification.supported_claim_indexes:
            if index >= len(state.candidate.claims):
                continue
            candidate = state.candidate.claims[index]
            if not set(candidate.evidence_ids) <= available:
                continue
            claims.append(
                EvidenceClaim(
                    statement=candidate.statement,
                    evidence_ids=candidate.evidence_ids,
                    confidence=candidate.confidence,
                )
            )
        if not claims:
            return None
        return DiagnosisReport(
            run_id=state.run_id,
            session_id=state.command.session_id,
            snapshot_id=state.snapshot_id,
            status=DiagnosisReportStatus.EVIDENCE_READY,
            question=state.command.question,
            summary=state.candidate.summary,
            claims=tuple(claims),
            evidence=state.evidence,
            tool_executions=state.tool_executions,
            attempted_queries=tuple(query for plan in state.plans for query in plan.search_queries),
            workflow_version=state.workflow_version,
        )

    def _complete(
        self,
        state: DiagnosisWorkflowState,
        report: DiagnosisReport,
    ) -> DiagnosisReport:
        completed = self._updated(
            state,
            stage=WorkflowStage.EVIDENCE_READY,
            report=report,
        )
        self._memory.append(
            session_id=state.command.session_id,
            message_id=self._memory_message_id(state.run_id, MessageRole.ASSISTANT),
            role=MessageRole.ASSISTANT,
            content=self._memory_content(report),
        )
        self._save(completed)
        return report

    def _refuse(self, state: DiagnosisWorkflowState) -> DiagnosisReport:
        issues = state.verification.issues if state.verification is not None else ()
        reason = "; ".join(issues) or "No selected tool returned sufficient verified evidence."
        report = DiagnosisReport(
            run_id=state.run_id,
            session_id=state.command.session_id,
            snapshot_id=state.snapshot_id,
            status=DiagnosisReportStatus.INSUFFICIENT_EVIDENCE,
            question=state.command.question,
            tool_executions=state.tool_executions,
            attempted_queries=tuple(query for plan in state.plans for query in plan.search_queries),
            refusal_reason=reason,
            workflow_version=state.workflow_version,
        )
        refused = self._updated(
            state,
            stage=WorkflowStage.INSUFFICIENT_EVIDENCE,
            report=report,
        )
        self._memory.append(
            session_id=state.command.session_id,
            message_id=self._memory_message_id(state.run_id, MessageRole.ASSISTANT),
            role=MessageRole.ASSISTANT,
            content=self._memory_content(report),
        )
        self._save(refused)
        return report
