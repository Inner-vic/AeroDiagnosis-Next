# ruff: noqa: RUF001 -- Exact Chinese grammar controls.
import pytest

from aerodiagnosis.application.evidence_semantics import translation_issues
from aerodiagnosis.application.grounded_answer import uncertainty_issues


@pytest.mark.parametrize(
    "source,bad,good",
    [
        ("The fuel pump cavitates.", "燃油泵发生气塞。", "燃油泵发生气蚀。"),
        ("Domestic object damage was observed.", "发现国内物体损伤。", "发现发动机内部物体损伤。"),
        (
            "Either blockage or a sensor error was present.",
            "存在堵塞和传感器错误。",
            "存在堵塞或传感器错误。",
        ),
        (
            "We did not identify failures that would indicate a systemic issue.",
            "未发现故障，表明存在系统性问题。",
            "未发现足以提示系统性问题的故障。",
        ),
    ],
)
def test_source_semantic_checks_distinguish_technical_and_scope_changes(source, bad, good):
    assert translation_issues(source, bad)
    assert not translation_issues(source, good)


def test_domestic_flight_is_not_internal_object_damage():
    assert not translation_issues("Domestic flights were grounded.", "国内航班停飞。")


@pytest.mark.parametrize(
    "claim,source",
    [
        ("裂纹起始区损伤限制了检查范围。", "裂纹起始区损伤限制了检查范围，因此无法确定裂纹起因。"),
        ("控制软件包含超速保护功能。", "控制软件包含超速保护功能，在转子可能超速之前关闭发动机。"),
        ("发动机非常可靠。", "发动机非常可靠，但罕见的内部失效仍可能发生。"),
        ("无法在本次检查范围内确定腐蚀的贡献。", "腐蚀的贡献无法在本次检查范围内确定。"),
    ],
)
def test_unrelated_modal_and_separated_unknown_phrasing_do_not_reject_fact(claim, source):
    assert not uncertainty_issues(claim, [source])


def test_dependent_clause_retains_parent_uncertainty_and_condition():
    assert uncertainty_issues(
        "该缺口导致裂纹扩展。", ["缺口可能来自制造过程，该缺口导致裂纹扩展。"]
    )
    assert uncertainty_issues("缺口已证实导致断裂。", ["缺口可能导致断裂。"])


def test_alternative_causes_cannot_be_distributed_to_simultaneous_findings():
    source = "油压超限表明循环系统故障、低油位或者指示系统问题之一。"
    assert uncertainty_issues("油压超限表明循环系统存在故障。", [source])
    assert not uncertainty_issues("油压超限可能表明循环系统存在故障。", [source])


def test_either_reworked_or_new_can_be_translated_with_yaome():
    source = "The hanger must be either new or reworked."
    assert not translation_issues(source, "护罩吊架要么使用新的、要么使用翻修过的。")
    assert translation_issues(source, "护罩吊架使用新的和翻修过的。")


def test_contained_failure_cannot_be_reversed():
    assert translation_issues(
        "The engine experienced a contained failure.", "发动机发生非包容性失效。"
    )
    assert not translation_issues(
        "The engine experienced a contained failure.", "发动机发生包容性失效。"
    )
    assert not translation_issues("An uncontained failure occurred.", "发生非包容性失效。")
