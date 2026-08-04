import json
from pathlib import Path

from scripts.eval.collect_analysis import build_parser
from scripts.eval.phase04_artifacts import write_phase04_artifacts


def _analysis_payload() -> dict:
    records = []
    for index in range(51):
        protocol = 1.0 if index != 1 else 0.0
        task = 1.0 if index != 2 else 0.5
        records.append(
            {
                "id": f"case-{index:03d}",
                "output_kind": "tool_call" if index < 15 else "final",
                "conversation_kind": "single_turn",
                "tags": ["tool_call"] if index < 15 else ["confirmation"],
                "input": {"user_input": "test"},
                "expected": {"action": "final"},
                "model_output": "{}",
                "dimensions": {
                    "protocol": {"score": protocol, "passed": protocol == 1.0, "detail": ""},
                    "task_correctness": {
                        "score": task,
                        "passed": task == 1.0,
                        "detail": "{}",
                    },
                },
                "timing": {"total_ms": 10.0, "first_token_ms": 5.0, "tokens_per_s": 20.0},
            }
        )
    return {
        "model": "qwen3-0.6b-sft",
        "backend_config": "config.yaml",
        "cases_path": "data/eval/test.jsonl",
        "n": 51,
        "aggregate_dimensions": {
            "protocol": {"score": 50 / 51},
            "task_correctness": {"score": 50.5 / 51},
        },
        "aggregate_timing": {"count": 51, "total_ms_mean": 10.0, "total_ms_p95": 10.0},
        "scenario_slices": {
            name: {"count": 1, "task_correctness": 1.0}
            for name in (
                "confirmation",
                "missing_information",
                "multi_turn",
                "tool_call",
                "tool_result",
                "unrelated",
            )
        },
        "records": records,
    }


def test_write_phase04_artifacts_has_effective_pass_and_slices(tmp_path: Path) -> None:
    write_phase04_artifacts(
        _analysis_payload(),
        tmp_path,
        run_id="qwen3-0.6b-sft",
        evaluation_environment={
            "backend": "llamafactory_huggingface",
            "device": "cpu",
            "latency_comparable_to_m0": False,
        },
    )
    rows = [
        json.loads(line)
        for line in (tmp_path / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert len(rows) == 51
    assert set(rows[0]) >= {
        "id",
        "input",
        "expected",
        "model_output",
        "dimensions",
        "effective_pass",
        "failure_reasons",
        "timing",
    }
    assert rows[1]["failure_reasons"] == ["protocol"]
    assert rows[2]["failure_reasons"] == ["task_correctness"]
    assert rows[0]["scenario_labels"] == ["tool_call"]
    card = json.loads((tmp_path / "scorecard.json").read_text(encoding="utf-8"))
    assert card["effective_pass"] == {"numerator": 49, "denominator": 51, "rate": 49 / 51}
    assert card["evaluation_environment"] == {
        "backend": "llamafactory_huggingface",
        "device": "cpu",
        "latency_comparable_to_m0": False,
    }
    assert set(card["scenario_slices"]) == {
        "confirmation",
        "missing_information",
        "multi_turn",
        "tool_call",
        "tool_result",
        "unrelated",
    }


def test_collect_analysis_accepts_phase04_run_output_mode(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--backend-config",
            "backend.yaml",
            "--cases",
            "cases.jsonl",
            "--run-id",
            "qwen3-0.6b-sft",
            "--run-dir",
            str(tmp_path),
            "--evaluation-backend",
            "llamafactory_huggingface",
            "--evaluation-device",
            "cpu",
        ]
    )
    assert args.out is None
    assert args.run_id == "qwen3-0.6b-sft"
    assert args.evaluation_backend == "llamafactory_huggingface"
    assert args.evaluation_device == "cpu"
