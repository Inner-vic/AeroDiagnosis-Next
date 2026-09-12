"""Model-grounded, knowledge-enhanced interactive root-cause workflow."""

# ruff: noqa: RUF001

from __future__ import annotations

import csv
import hashlib
import io
import math
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from pydantic import ValidationError

from aerodiagnosis.domain import (
    AgentTraceEvent,
    ClassSpecification,
    ComponentPrediction,
    DatasetProfile,
    EngineerObservation,
    FeatureSpecification,
    InspectionProposal,
    KnowledgeEnhancedReport,
    ModelInferenceResult,
    ModelPluginManifest,
    ModelPluginRecord,
    PluginStatus,
    RootCauseCoordination,
    RootCauseHypothesis,
    RootCauseSession,
    RootCauseSessionStatus,
)
from aerodiagnosis.ports import KnowledgeEnhancementStore, LanguageModel, LanguageModelError


@dataclass(frozen=True, slots=True)
class _RootCauseKnowledge:
    hypothesis_id: str
    label: str
    component: str
    description: str
    action_id: str
    action_title: str
    question: str
    purpose: str
    maintenance_reference: str


_SOURCE = "教学合成根因与检查动作库 v1；实际使用前必须依据适用维修手册复核"
_KNOWLEDGE = (
    _RootCauseKnowledge(
        "fan_fouling",
        "风扇叶片积垢",
        "fan",
        "表面沉积可能改变气动外形并引起性能偏离。",
        "inspect_fan_contamination",
        "检查风扇表面污染与沉积",
        "是否观察到风扇叶片表面存在明显污染、沉积或附着物？",
        "用于区分表面积垢与结构性损伤。",
        "若确认存在污染，应依据适用手册评估清洁、复查和恢复性测试。",
    ),
    _RootCauseKnowledge(
        "fan_erosion",
        "风扇叶片侵蚀",
        "fan",
        "前缘或叶尖侵蚀可能造成效率下降和气路参数变化。",
        "inspect_fan_leading_edge",
        "检查叶片前缘与叶尖侵蚀",
        "叶片前缘或叶尖是否存在圆钝、材料损失或不均匀磨损？",
        "用于确认长期侵蚀类根因。",
        "记录侵蚀位置和程度，并依据适用手册限值决定后续处置。",
    ),
    _RootCauseKnowledge(
        "fan_fod",
        "风扇外物损伤",
        "fan",
        "缺口、凹坑或变形可能指向外物撞击。",
        "inspect_fan_fod",
        "检查外物损伤迹象",
        "是否发现叶片缺口、凹坑、裂纹、卷边或局部变形？",
        "用于识别外物撞击导致的结构性异常。",
        "若发现损伤，应停止自动推理并交由具备资质人员按手册评估。",
    ),
    _RootCauseKnowledge(
        "fan_sensor_chain",
        "风扇相关传感链异常",
        "fan",
        "孤立参数异常也可能来自转速、压力或采集链路。",
        "cross_check_fan_sensors",
        "交叉核验风扇相关传感器",
        "相关转速、压力和振动信号是否出现不同步、跳变或单通道异常？",
        "用于区分真实部件异常与测量链异常。",
        "复核传感器、接线与采集通道，不以单一异常通道直接判定部件故障。",
    ),
    _RootCauseKnowledge(
        "compressor_fouling",
        "压气机积垢",
        "compressor",
        "积垢可能造成流量能力和效率下降。",
        "compare_compressor_trend",
        "比较压气机性能趋势",
        "在相近工况下，压比与排气温度是否呈持续、缓慢的协同偏移？",
        "用于识别渐进式压气机性能退化。",
        "依据适用手册复核趋势并评估是否需要进一步孔探或性能恢复措施。",
    ),
    _RootCauseKnowledge(
        "compressor_bleed",
        "引气或可调机构异常",
        "compressor",
        "引气和可调机构状态可能改变压气机工作点。",
        "check_bleed_and_vanes",
        "检查引气与可调机构状态",
        "是否存在引气异常、可调静子位置异常或相关控制告警？",
        "用于区分本体退化与控制/引气系统影响。",
        "按控制与引气系统手册完成状态核验，不直接更换部件。",
    ),
    _RootCauseKnowledge(
        "compressor_erosion",
        "压气机叶片侵蚀或损伤",
        "compressor",
        "内部叶片退化可能造成压比和温度特征变化。",
        "inspect_compressor_path",
        "检查压气机气路可见损伤",
        "已有孔探或检查结果是否显示叶片侵蚀、缺口、裂纹或异常间隙？",
        "用于验证压气机内部结构性根因。",
        "需要具备资质的人员依据适用孔探与维修标准判读。",
    ),
    _RootCauseKnowledge(
        "compressor_sensor_chain",
        "压气机参数测量异常",
        "compressor",
        "压力或温度测量偏差可能模拟性能下降。",
        "cross_check_compressor_sensors",
        "交叉核验压气机测量链",
        "压气机相关压力、温度通道与冗余或邻近参数是否一致？",
        "用于排除单通道测量偏差。",
        "复核测量链和数据质量，再决定是否进入部件检查。",
    ),
    _RootCauseKnowledge(
        "turbine_hot_section",
        "涡轮热端退化",
        "turbine",
        "热端效率变化可能导致排气温度和性能趋势偏移。",
        "review_hot_section_trend",
        "复核热端参数趋势",
        "在相同工况下，排气温度与燃油、转速参数是否出现持续协同恶化？",
        "用于识别渐进式热端退化。",
        "依据适用手册进行趋势确认并决定是否需要进一步检查。",
    ),
    _RootCauseKnowledge(
        "turbine_cooling",
        "涡轮冷却流异常",
        "turbine",
        "冷却流异常可能影响热端温度分布。",
        "check_turbine_cooling",
        "检查冷却流相关迹象",
        "是否存在冷却气路异常、堵塞迹象或相关控制告警？",
        "用于区分热端本体退化与冷却条件异常。",
        "按适用手册核验冷却气路，不根据本系统直接实施维修。",
    ),
    _RootCauseKnowledge(
        "turbine_blade_damage",
        "涡轮叶片损伤",
        "turbine",
        "结构损伤可能伴随温度、振动和效率异常。",
        "inspect_turbine_blades",
        "检查涡轮叶片损伤证据",
        "孔探、振动或其他检查是否显示涡轮叶片裂纹、烧蚀或缺损？",
        "用于确认高风险结构性根因。",
        "若发现损伤，应立即转交具备资质人员按适用手册评估。",
    ),
    _RootCauseKnowledge(
        "turbine_sensor_chain",
        "排气温度测量链异常",
        "turbine",
        "温度探头或采集异常可能产生局部或整体偏差。",
        "cross_check_egt_sensors",
        "核验排气温度测量链",
        "多个温度通道是否一致，是否存在单点跳变、固定偏置或异常离散？",
        "用于排除测量链导致的假异常。",
        "复核温度探头与采集链，保留原始数据供人工判断。",
    ),
)


def teaching_demo_manifest() -> ModelPluginManifest:
    return ModelPluginManifest(
        model_id="teaching-linear-gaspath",
        version="1.0.0",
        display_name="教学演示气路部件分类器",
        dataset_id="teaching-gaspath-a",
        features=(
            FeatureSpecification(name="fan_vibration", unit="g", mean=0.2, scale=0.1),
            FeatureSpecification(name="pressure_ratio", mean=1.7, scale=0.15),
            FeatureSpecification(name="egt", unit="C", mean=600.0, scale=40.0),
            FeatureSpecification(name="n1", unit="%", mean=90.0, scale=4.0),
        ),
        classes=(
            ClassSpecification(
                code="fan_anomaly",
                label="风扇部件异常",
                component="fan",
                weights=(2.2, -0.2, 0.2, 0.3),
            ),
            ClassSpecification(
                code="compressor_anomaly",
                label="压气机部件异常",
                component="compressor",
                weights=(0.1, -1.7, 0.8, 0.2),
            ),
            ClassSpecification(
                code="turbine_anomaly",
                label="涡轮部件异常",
                component="turbine",
                weights=(0.1, 0.1, 1.9, 0.1),
            ),
        ),
        ood_threshold=8.0,
        description="声明式线性教学模型，仅用于贯通模型注册、数据路由和知识增强流程；不代表深度学习模型或真实发动机诊断性能。",
        teaching_demo=True,
    )


class KnowledgeEnhancedDiagnosis:
    def __init__(self, store: KnowledgeEnhancementStore) -> None:
        self._store = store
        if store.get_model("teaching-linear-gaspath") is None:
            store.register_model(teaching_demo_manifest())

    def register_model(self, manifest: ModelPluginManifest) -> ModelPluginRecord:
        return self._store.register_model(manifest)

    def list_models(self) -> tuple[ModelPluginRecord, ...]:
        return self._store.list_models()

    def list_sessions(self, *, limit: int = 50) -> tuple[RootCauseSession, ...]:
        return self._store.list_sessions(limit=limit)

    def get(self, rca_id: str) -> RootCauseSession:
        session = self._store.get_session(rca_id)
        if session is None:
            raise KeyError(f"root-cause session not found: {rca_id}")
        return session

    def start(
        self, *, display_name: str, csv_content: str, model_id: str, language_model: LanguageModel
    ) -> RootCauseSession:
        record = self._store.get_model(model_id)
        if record is None or record.status is not PluginStatus.ACTIVE:
            raise ValueError("selected model plugin is not active")
        profile, values = self._profile_csv(display_name, csv_content, record.manifest)
        result = self._infer(record.manifest, profile, values)
        if result.applicability != "in_domain":
            raise ValueError("uploaded data is outside the selected model applicability boundary")
        now = datetime.now(UTC).isoformat()
        component = result.predictions[0].component
        hypotheses = self._initial_hypotheses(component)
        actions = self._remaining_actions(component, ())
        coordination = self._coordinate(language_model, result, hypotheses, actions, ())
        hypotheses = self._ordered_hypotheses(hypotheses, coordination)
        next_action = self._proposal(coordination.next_action_id, actions)
        trace = (
            self._event(1, "trigger_router", "workflow_routed", "数据上传触发知识增强根因分析。"),
            self._event(
                2, "data_analysis_agent", "dataset_profiled", self._profile_summary(profile)
            ),
            self._event(3, "model_diagnosis_agent", "model_inferred", self._model_summary(result)),
            self._event(
                4,
                "knowledge_retrieval_agent",
                "root_causes_retrieved",
                f"从版本化教学知识包召回 {len(hypotheses)} 个候选根因。",
            ),
            self._event(5, "root_cause_agent", "hypotheses_ranked", coordination.rationale),
            self._event(
                6,
                "inspection_planner_agent",
                "engineer_input_requested",
                next_action.question if next_action else "当前没有可执行的下一项检查。",
                status="waiting" if next_action else "completed",
            ),
        )
        session = RootCauseSession(
            rca_id=str(uuid.uuid4()),
            status=RootCauseSessionStatus.INVESTIGATING,
            dataset=profile,
            model_result=result,
            hypotheses=hypotheses,
            observations=(),
            next_action=next_action,
            trace=trace,
            created_at=now,
            updated_at=now,
        )
        self._store.save_session(session)
        return session

    def observe(
        self,
        rca_id: str,
        *,
        action_id: str,
        outcome: str,
        notes: str,
        language_model: LanguageModel,
    ) -> RootCauseSession:
        session = self.get(rca_id)
        if session.status is RootCauseSessionStatus.COMPLETED:
            raise ValueError("completed root-cause sessions cannot accept observations")
        if session.next_action is None or session.next_action.action_id != action_id:
            raise ValueError("observation must answer the current inspection action")
        now = datetime.now(UTC).isoformat()
        observation = EngineerObservation(
            observation_id=str(uuid.uuid4()),
            action_id=action_id,
            outcome=outcome,
            notes=notes,
            created_at=now,
        )
        observations = (*session.observations, observation)
        hypotheses = self._update_hypotheses(session.hypotheses, observation)
        component = session.model_result.predictions[0].component
        remaining = self._remaining_actions(
            component, tuple(item.action_id for item in observations)
        )
        ready = (
            len(observations) >= 4
            or max(item.score for item in hypotheses) >= 0.72
            or not remaining
        )
        trace = (
            *session.trace,
            self._event(
                len(session.trace) + 1,
                "engineer",
                "observation_recorded",
                f"{action_id}={outcome}；{notes or '未附加说明'}",
            ),
            self._event(
                len(session.trace) + 2,
                "state_update_engine",
                "hypotheses_updated",
                f"已依据结构化观察更新 {len(hypotheses)} 个候选根因。",
            ),
        )
        next_action: InspectionProposal | None = None
        status = RootCauseSessionStatus.READY_FOR_REPORT if ready else session.status
        if ready:
            trace = (
                *trace,
                self._event(
                    len(trace) + 1,
                    "verifier_agent",
                    "report_gate_ready",
                    "当前证据达到报告生成条件；未确认项仍将在报告中保留。",
                ),
            )
        else:
            coordination = self._coordinate(
                language_model, session.model_result, hypotheses, remaining, observations
            )
            hypotheses = self._ordered_hypotheses(hypotheses, coordination)
            next_action = self._proposal(coordination.next_action_id, remaining)
            trace = (
                *trace,
                self._event(
                    len(trace) + 1, "root_cause_agent", "loop_continued", coordination.rationale
                ),
                self._event(
                    len(trace) + 2,
                    "inspection_planner_agent",
                    "engineer_input_requested",
                    next_action.question if next_action else "当前没有可执行的下一项检查。",
                    status="waiting" if next_action else "completed",
                ),
            )
        updated = session.model_copy(
            update={
                "status": status,
                "hypotheses": hypotheses,
                "observations": observations,
                "next_action": next_action,
                "trace": trace,
                "updated_at": now,
            }
        )
        self._store.save_session(updated)
        return updated

    def finalize(self, rca_id: str, language_model: LanguageModel) -> RootCauseSession:
        session = self.get(rca_id)
        if session.status is RootCauseSessionStatus.COMPLETED:
            return session
        knowledge = self._component_knowledge(session.model_result.predictions[0].component)
        raw = language_model.complete_json(
            "write_root_cause_report",
            {
                "instruction": (
                    "Use only supplied results and preserve uncertainty. "
                    "Never present teaching knowledge as a real manual."
                ),
                "model_result": session.model_result.model_dump(mode="json"),
                "hypotheses": [item.model_dump(mode="json") for item in session.hypotheses],
                "observations": [item.model_dump(mode="json") for item in session.observations],
                "allowed_maintenance_support": [item.maintenance_reference for item in knowledge],
                "mandatory_limitations": [
                    "当前模型和根因知识为教学演示，不代表真实发动机维修结论。",
                    "任何检查与维修参考必须由具备资质人员结合适用型号手册复核。",
                ],
            },
        )
        try:
            report = KnowledgeEnhancedReport.model_validate(raw)
        except ValidationError as exc:
            raise LanguageModelError("invalid structured output for root-cause report") from exc
        allowed_support = {item.maintenance_reference for item in knowledge}
        if not set(report.maintenance_support).issubset(allowed_support):
            raise LanguageModelError(
                "root-cause report introduced maintenance guidance outside supplied knowledge"
            )
        mandatory_limitations = {
            "当前模型和根因知识为教学演示，不代表真实发动机维修结论。",
            "任何检查与维修参考必须由具备资质人员结合适用型号手册复核。",
        }
        if not mandatory_limitations.issubset(set(report.limitations)):
            raise LanguageModelError("root-cause report omitted mandatory safety limitations")
        now = datetime.now(UTC).isoformat()
        trace = (
            *session.trace,
            self._event(
                len(session.trace) + 1,
                "report_agent",
                "report_generated",
                "已从冻结的模型结果、知识和工程师观察生成知识增强报告。",
                status="completed",
            ),
        )
        updated = session.model_copy(
            update={
                "status": RootCauseSessionStatus.COMPLETED,
                "next_action": None,
                "trace": trace,
                "report": report,
                "updated_at": now,
            }
        )
        self._store.save_session(updated)
        return updated

    @staticmethod
    def _profile_csv(
        display_name: str, content: str, manifest: ModelPluginManifest
    ) -> tuple[DatasetProfile, dict[str, float]]:
        if not display_name.strip() or not content.strip():
            raise ValueError("diagnostic CSV name and content must not be empty")
        reader = csv.DictReader(io.StringIO(content))
        if not reader.fieldnames:
            raise ValueError("diagnostic CSV requires a header row")
        columns = tuple(name.strip() for name in reader.fieldnames)
        required = tuple(feature.name for feature in manifest.features)
        missing_columns = [name for name in required if name not in columns]
        if missing_columns:
            raise ValueError(
                f"diagnostic CSV is missing model features: {', '.join(missing_columns)}"
            )
        sums = dict.fromkeys(required, 0.0)
        counts = dict.fromkeys(required, 0)
        missing_values = 0
        row_count = 0
        for row in reader:
            row_count += 1
            if row_count > 100_000:
                raise ValueError("diagnostic CSV exceeds the 100000-row local limit")
            for name in required:
                raw = (row.get(name) or "").strip()
                if not raw:
                    missing_values += 1
                    continue
                try:
                    value = float(raw)
                except ValueError as exc:
                    raise ValueError(
                        f"diagnostic feature {name} contains non-numeric data"
                    ) from exc
                if not math.isfinite(value):
                    raise ValueError(f"diagnostic feature {name} must contain finite values")
                sums[name] += value
                counts[name] += 1
        if row_count < 1 or any(counts[name] == 0 for name in required):
            raise ValueError("diagnostic CSV requires at least one value for every model feature")
        quality = max(0.0, 1.0 - missing_values / (row_count * len(required)))
        profile = DatasetProfile(
            display_name=display_name.strip(),
            row_count=row_count,
            columns=columns,
            numeric_columns=required,
            missing_values=missing_values,
            quality_score=quality,
            content_hash=hashlib.sha256(content.encode()).hexdigest(),
        )
        return profile, {name: sums[name] / counts[name] for name in required}

    @staticmethod
    def _infer(
        manifest: ModelPluginManifest, profile: DatasetProfile, values: dict[str, float]
    ) -> ModelInferenceResult:
        standardized = tuple(
            (values[feature.name] - feature.mean) / feature.scale for feature in manifest.features
        )
        logits = [
            item.bias
            + sum(weight * value for weight, value in zip(item.weights, standardized, strict=True))
            for item in manifest.classes
        ]
        offset = max(logits)
        exponentials = [math.exp(value - offset) for value in logits]
        total = sum(exponentials)
        predictions = tuple(
            sorted(
                (
                    ComponentPrediction(
                        code=item.code,
                        label=item.label,
                        component=item.component,
                        probability=score / total,
                    )
                    for item, score in zip(manifest.classes, exponentials, strict=True)
                ),
                key=lambda item: (-item.probability, item.code),
            )
        )
        ood_score = max(abs(value) for value in standardized)
        warnings = (
            ("输入超出教学模型训练画像；拒绝知识增强诊断。",)
            if ood_score > manifest.ood_threshold
            else ()
        )
        return ModelInferenceResult(
            model_id=manifest.model_id,
            model_version=manifest.version,
            dataset_id=manifest.dataset_id,
            applicability="out_of_domain" if warnings else "in_domain",
            input_quality=profile.quality_score,
            predictions=predictions[:3],
            ood_score=ood_score,
            abnormal_features=tuple(
                feature.name
                for feature, value in zip(manifest.features, standardized, strict=True)
                if abs(value) >= 1.0
            ),
            warnings=warnings,
        )

    @staticmethod
    def _component_knowledge(component: str) -> tuple[_RootCauseKnowledge, ...]:
        matches = tuple(item for item in _KNOWLEDGE if item.component == component)
        if not matches:
            raise ValueError(f"no root-cause knowledge pack registered for component: {component}")
        return matches

    def _initial_hypotheses(self, component: str) -> tuple[RootCauseHypothesis, ...]:
        knowledge = self._component_knowledge(component)
        score = 1.0 / len(knowledge)
        return tuple(
            RootCauseHypothesis(
                hypothesis_id=item.hypothesis_id,
                label=item.label,
                component=item.component,
                description=item.description,
                score=score,
                status="active",
            )
            for item in knowledge
        )

    def _remaining_actions(
        self, component: str, completed: tuple[str, ...]
    ) -> tuple[_RootCauseKnowledge, ...]:
        return tuple(
            item for item in self._component_knowledge(component) if item.action_id not in completed
        )

    @staticmethod
    def _proposal(
        action_id: str | None, actions: tuple[_RootCauseKnowledge, ...]
    ) -> InspectionProposal | None:
        if action_id is None:
            return None
        selected = next((item for item in actions if item.action_id == action_id), None)
        if selected is None:
            raise LanguageModelError(
                "root-cause coordinator selected an unavailable inspection action"
            )
        return InspectionProposal(
            action_id=selected.action_id,
            title=selected.action_title,
            question=selected.question,
            purpose=selected.purpose,
            source=_SOURCE,
        )

    @staticmethod
    def _coordinate(
        language_model: LanguageModel,
        result: ModelInferenceResult,
        hypotheses: tuple[RootCauseHypothesis, ...],
        actions: tuple[_RootCauseKnowledge, ...],
        observations: tuple[EngineerObservation, ...],
    ) -> RootCauseCoordination:
        raw = language_model.complete_json(
            "coordinate_root_cause",
            {
                "initial_model_result": result.model_dump(mode="json"),
                "allowed_hypotheses": [item.model_dump(mode="json") for item in hypotheses],
                "allowed_actions": [
                    {
                        "action_id": item.action_id,
                        "targets": [item.hypothesis_id],
                        "title": item.action_title,
                        "purpose": item.purpose,
                    }
                    for item in actions
                ],
                "observations": [item.model_dump(mode="json") for item in observations],
                "policy": (
                    "Rank every hypothesis exactly once. Select one available action. "
                    "Treat model output as initial evidence, not final truth."
                ),
            },
        )
        try:
            decision = RootCauseCoordination.model_validate(raw)
        except ValidationError as exc:
            raise LanguageModelError(
                "invalid structured output for root-cause coordination"
            ) from exc
        if set(decision.ordered_hypothesis_ids) != {item.hypothesis_id for item in hypotheses}:
            raise LanguageModelError("root-cause coordinator changed the allowed hypothesis set")
        if decision.next_action_id not in {item.action_id for item in actions}:
            raise LanguageModelError(
                "root-cause coordinator selected an unavailable inspection action"
            )
        return decision

    @staticmethod
    def _ordered_hypotheses(
        hypotheses: tuple[RootCauseHypothesis, ...], coordination: RootCauseCoordination
    ) -> tuple[RootCauseHypothesis, ...]:
        by_id = {item.hypothesis_id: item for item in hypotheses}
        return tuple(by_id[item] for item in coordination.ordered_hypothesis_ids)

    @staticmethod
    def _update_hypotheses(
        hypotheses: tuple[RootCauseHypothesis, ...], observation: EngineerObservation
    ) -> tuple[RootCauseHypothesis, ...]:
        target = next(
            (item.hypothesis_id for item in _KNOWLEDGE if item.action_id == observation.action_id),
            None,
        )
        if target is None:
            raise ValueError("unknown inspection action")
        raw_scores = {}
        for item in hypotheses:
            score = item.score
            if item.hypothesis_id == target:
                if observation.outcome == "present":
                    score *= 2.5
                elif observation.outcome == "absent":
                    score *= 0.2
                elif observation.outcome == "not_checked":
                    score *= 0.95
            raw_scores[item.hypothesis_id] = max(score, 0.001)
        total = sum(raw_scores.values())
        normalized = {key: value / total for key, value in raw_scores.items()}
        highest = max(normalized.values())
        updated = []
        for item in hypotheses:
            score = normalized[item.hypothesis_id]
            supports = (
                (*item.supporting_observations, observation.observation_id)
                if item.hypothesis_id == target and observation.outcome == "present"
                else item.supporting_observations
            )
            contradicts = (
                (*item.contradicting_observations, observation.observation_id)
                if item.hypothesis_id == target and observation.outcome == "absent"
                else item.contradicting_observations
            )
            status = (
                "excluded"
                if score < 0.08
                else "weakened"
                if score < 0.18
                else "confirmed"
                if score == highest and highest >= 0.72
                else "active"
            )
            updated.append(
                item.model_copy(
                    update={
                        "score": score,
                        "status": status,
                        "supporting_observations": supports,
                        "contradicting_observations": contradicts,
                    }
                )
            )
        return tuple(sorted(updated, key=lambda item: (-item.score, item.hypothesis_id)))

    @staticmethod
    def _event(
        sequence: int, role: str, event_type: str, summary: str, *, status: str = "ok"
    ) -> AgentTraceEvent:
        return AgentTraceEvent(
            sequence=sequence,
            agent_role=role,
            event_type=event_type,
            summary=summary,
            status=status,
            created_at=datetime.now(UTC).isoformat(),
        )

    @staticmethod
    def _profile_summary(profile: DatasetProfile) -> str:
        return (
            f"读取 {profile.row_count} 行、{len(profile.columns)} 列；"
            f"质量分 {profile.quality_score:.2f}，缺失值 {profile.missing_values}。"
        )

    @staticmethod
    def _model_summary(result: ModelInferenceResult) -> str:
        top = result.predictions[0]
        return (
            f"{result.model_id}@{result.model_version} 初步定位“{top.label}”"
            f"（{top.probability:.1%}），适用性={result.applicability}。"
        )
