import pytest

from scripts.eval.diff_runs import diff_predictions


def _row(case_id: str, effective: bool, task: float, *, tag: str = "tool_call") -> dict:
    return {
        "id": case_id,
        "run_id": "run",
        "tags": [tag],
        "model_output": "{}",
        "effective_pass": effective,
        "failure_reasons": [] if effective else ["task_correctness"],
        "dimensions": {
            "protocol": {"score": 1.0},
            "task_correctness": {"score": task},
        },
    }


def test_diff_classifies_flips_and_net_change() -> None:
    left = [_row("case-001", True, 0.5), _row("case-002", False, 0.5), _row("case-003", True, 0.5)]
    right = [_row("case-001", True, 0.75), _row("case-002", True, 0.75), _row("case-003", False, 0.25)]
    result = diff_predictions(left, right, left_run_id="left", right_run_id="right")
    assert result["flipped_positive"] == ["case-002"]
    assert result["flipped_negative"] == ["case-003"]
    assert result["net_effective_pass"] == 0
    assert result["scenario_delta"]["tool_call"] == pytest.approx(1 / 12)


def test_diff_rejects_mismatched_case_sets() -> None:
    left = [_row("case-001", True, 1.0), _row("case-002", False, 0.0)]
    with pytest.raises(ValueError, match="sample id sets differ"):
        diff_predictions(left, left[:-1], left_run_id="left", right_run_id="right")

