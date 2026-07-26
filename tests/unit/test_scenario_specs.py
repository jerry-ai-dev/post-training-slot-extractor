import pytest

from slot_extractor.data.generator import GenerationRequest, build_generation_messages
from slot_extractor.data.scenario_specs import (
    SCENARIOS,
    scenario_dpo_targets,
    scenarios_by_category,
)


def test_scenario_matrix_has_five_scenarios_per_category() -> None:
    groups = scenarios_by_category()
    assert set(groups) == {"追问", "工具调用", "最终 JSON", "确认", "无关"}
    assert all(len(scenarios) == 5 for scenarios in groups.values())
    assert len(SCENARIOS) == 25


@pytest.mark.parametrize(
    "scenario",
    [
        "final_unavailable",
        "final_not_found",
        "final_no_match",
        "confirm_reject",
        "acknowledge_unavailable",
        "tool_replace_technician",
    ],
)
def test_generation_prompt_contains_scenario_contract(scenario: str) -> None:
    spec = SCENARIOS[scenario]
    messages = build_generation_messages(GenerationRequest(spec.category, 1, scenario))
    text = "\n".join(str(message["content"]) for message in messages)
    assert scenario in text
    assert spec.instruction in text


def test_scenario_dpo_routing_handles_negative_confirmations() -> None:
    assert scenario_dpo_targets("confirm_accept") == ("P5",)
    assert scenario_dpo_targets("confirm_reject") == ()
    assert scenario_dpo_targets("acknowledge_unavailable") == ()
    assert scenario_dpo_targets("final_unavailable") == ("P4", "P2P3")
