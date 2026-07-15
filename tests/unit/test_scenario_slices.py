from slot_extractor.evaluation.scenarios import aggregate_scenario_slices, scenario_labels
from slot_extractor.schemas.sample import Sample


def _sample(**updates: object) -> Sample:
    expected = {
        "action": "final",
        "info_complete": True,
        "unrelated": False,
        "confirmation": False,
    }
    expected.update(updates.pop("expected", {}))
    input_obj = {"history": []}
    input_obj.update(updates.pop("input", {}))
    return Sample(
        id=str(updates.pop("id", "case")),
        layer=updates.pop("layer", "final"),
        input=input_obj,
        expected=expected,
        assertions=[],
        tags=[],
    )


def test_scenario_labels_are_non_scoring_slices() -> None:
    sample = _sample(
        layer="multi_turn",
        input={
            "history": [
                {"role": "user", "content": "预约"},
                {"role": "tool", "tool_call_id": "call-1", "content": "{}"},
            ]
        },
        expected={"info_complete": False, "missing_info": ["duration_minutes"], "unrelated": True, "confirmation": True},
    )

    assert scenario_labels(sample) == {
        "confirmation",
        "missing_information",
        "multi_turn",
        "tool_result",
        "unrelated",
    }


def test_aggregate_scenario_slices_averages_task_scores() -> None:
    samples = [_sample(id="a", expected={"unrelated": True}), _sample(id="b", expected={"unrelated": True})]

    slices = aggregate_scenario_slices(samples, {"a": 1.0, "b": 0.5})

    assert slices["unrelated"] == {"count": 2, "task_correctness": 0.75}
