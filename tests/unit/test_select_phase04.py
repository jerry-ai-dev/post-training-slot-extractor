from scripts.eval.select_phase04 import select_winner


def card(
    run_id: str,
    *,
    effective: int,
    task: float,
    protocol: float,
    params: float,
    parent: str | None = None,
) -> dict:
    return {
        "run_id": run_id,
        "status": "evaluated",
        "effective_pass": {"numerator": effective, "denominator": 51},
        "aggregate_dimensions": {
            "task_correctness": {"score": task},
            "protocol": {"score": protocol},
        },
        "parameter_billions": params,
        "parent_run_id": parent,
    }


def test_dpo_protocol_regression_over_two_points_is_ineligible() -> None:
    sft = card("sft", effective=25, task=0.80, protocol=0.90, params=1.7)
    dpo = card(
        "dpo", effective=30, task=0.90, protocol=0.879, params=1.7, parent="sft"
    )
    result = select_winner([sft, dpo])
    assert result["winner"] == "sft"
    assert result["runs"]["dpo"]["eligible"] is False


def test_one_case_tie_uses_task_then_smaller_model() -> None:
    result = select_winner(
        [
            card("large", effective=26, task=0.81, protocol=0.95, params=1.7),
            card("small", effective=25, task=0.82, protocol=0.95, params=0.6),
        ]
    )
    assert result["winner"] == "small"


def test_failed_run_remains_visible_but_ineligible() -> None:
    failed = card("failed", effective=0, task=0.0, protocol=0.0, params=0.6)
    failed["status"] = "failed"
    result = select_winner([failed, card("ok", effective=1, task=0.1, protocol=1.0, params=0.6)])
    assert result["runs"]["failed"]["eligible"] is False
