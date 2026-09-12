from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from aerodiagnosis.adapters.persistence import SQLiteCaseStore, SQLiteKnowledgeEnhancementStore
from aerodiagnosis.application import KnowledgeEnhancedDiagnosis


class CoordinatingModel:
    @property
    def identity(self) -> str:
        return "controlled-root-cause-model"

    def complete_json(self, task: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        if task == "coordinate_root_cause":
            return {
                "ordered_hypothesis_ids": [
                    item["hypothesis_id"] for item in payload["allowed_hypotheses"]
                ],
                "next_action_id": payload["allowed_actions"][0]["action_id"],
                "rationale": "Choose an available inspection that distinguishes one candidate.",
            }
        if task == "write_root_cause_report":
            return {
                "summary": "The initial component result was enhanced with inspection evidence.",
                "root_cause_analysis": (
                    "One candidate was weakened; remaining uncertainty is retained."
                ),
                "maintenance_support": [payload["allowed_maintenance_support"][0]],
                "limitations": list(payload["mandatory_limitations"]),
            }
        raise AssertionError(task)


def test_model_result_triggers_persistent_root_cause_loop(tmp_path: Path) -> None:
    store = SQLiteKnowledgeEnhancementStore(tmp_path / "runtime.db")
    case_store = SQLiteCaseStore(tmp_path / "runtime.db")
    service = KnowledgeEnhancedDiagnosis(store, case_store)
    model = CoordinatingModel()

    plugins = service.list_models()
    assert len(plugins) == 1
    assert plugins[0].manifest.teaching_demo is True

    session = service.start(
        display_name="fan-demo.csv",
        csv_content=("fan_vibration,pressure_ratio,egt,n1\n0.55,1.70,610,91\n0.60,1.68,612,91\n"),
        model_id="teaching-linear-gaspath",
        language_model=model,
    )

    assert session.model_result.predictions[0].component == "fan"
    assert session.next_action is not None
    assert session.trace[0].agent_role == "trigger_router"
    assert {item.agent_role for item in session.trace} >= {
        "data_analysis_agent",
        "model_diagnosis_agent",
        "knowledge_retrieval_agent",
        "root_cause_agent",
        "inspection_planner_agent",
    }

    updated = service.observe(
        session.rca_id,
        action_id=session.next_action.action_id,
        outcome="absent",
        notes="No visible contamination was found.",
        language_model=model,
    )

    assert len(updated.observations) == 1
    assert updated.next_action is not None
    assert updated.next_action.action_id != session.next_action.action_id
    assert store.get_session(session.rca_id) == updated

    completed = service.finalize(session.rca_id, model)
    assert completed.status == "completed"
    assert completed.report is not None
    assert completed.report.limitations
    assert completed.case_draft is not None
    assert completed.case_draft.status == "draft"
    assert completed.case_draft.source_rca_id == session.rca_id

    published = service.publish_case(session.rca_id)
    assert published.case_draft is not None
    assert published.case_draft.status == "published"
    assert case_store.count() == 1
    saved_case = case_store.list_cases()[0]
    assert saved_case.attributes["source_rca_id"] == session.rca_id
    assert saved_case.attributes["leading_root_cause"]

    repeated = service.publish_case(session.rca_id)
    assert repeated == published
    assert case_store.count() == 1
    assert service.list_sessions()[0].rca_id == session.rca_id


def test_case_cannot_be_published_before_report_is_completed(tmp_path: Path) -> None:
    database = tmp_path / "runtime.db"
    service = KnowledgeEnhancedDiagnosis(
        SQLiteKnowledgeEnhancementStore(database), SQLiteCaseStore(database)
    )
    session = service.start(
        display_name="fan-demo.csv",
        csv_content="fan_vibration,pressure_ratio,egt,n1\n0.55,1.70,610,91\n",
        model_id="teaching-linear-gaspath",
        language_model=CoordinatingModel(),
    )

    with pytest.raises(ValueError, match="only completed"):
        service.publish_case(session.rca_id)


def test_registered_model_rejects_incompatible_or_out_of_domain_data(tmp_path: Path) -> None:
    service = KnowledgeEnhancedDiagnosis(SQLiteKnowledgeEnhancementStore(tmp_path / "runtime.db"))
    model = CoordinatingModel()

    with pytest.raises(ValueError, match="missing model features"):
        service.start(
            display_name="wrong.csv",
            csv_content="egt,n1\n600,90\n",
            model_id="teaching-linear-gaspath",
            language_model=model,
        )

    with pytest.raises(ValueError, match="outside the selected model"):
        service.start(
            display_name="ood.csv",
            csv_content=("fan_vibration,pressure_ratio,egt,n1\n100,1.7,600,90\n"),
            model_id="teaching-linear-gaspath",
            language_model=model,
        )
