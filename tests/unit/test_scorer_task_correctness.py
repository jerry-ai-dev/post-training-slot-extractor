import json

from slot_extractor.evaluation.scorers.task_correctness import TaskCorrectnessScorer
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.schemas.sample import ReplyExpectations, Sample


def _result(output: dict) -> GenerationResult:
    return GenerationResult(
        text=json.dumps(output, ensure_ascii=False),
        model="mock",
        prefill_ms=None,
        first_token_ms=None,
        total_ms=1,
    )


def _final_sample(expected: dict, expectations: ReplyExpectations) -> Sample:
    return Sample(
        id="final-case",
        layer="final",
        input={"current_time": "2026-06-08 10:00"},
        expected=expected,
        assertions=[],
        tags=[],
        reply_expectations=expectations,
    )


def _expected_final() -> dict:
    return {
        "action": "final",
        "gender": "female",
        "start_time": "2026-06-09 14:00",
        "duration_minutes": 60,
        "preferences": ["肩颈"],
        "technician_name": "王芳",
        "technician_status": "available",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "confirm_available",
        "reply": "王芳技师明天下午2点有空，可以安排60分钟肩颈按摩，您确认吗？",
    }


def test_final_task_score_combines_structured_and_reply_with_70_30_weights() -> None:
    expected = _expected_final()
    actual = dict(expected)
    actual["gender"] = None
    score = TaskCorrectnessScorer().score(
        _final_sample(
            expected,
            ReplyExpectations(
                required_acts=("inform_technician_available", "request_confirmation"),
                forbidden_acts=("claim_booking_success",),
                required_fields=("technician_name", "start_time", "duration_minutes"),
                references=(expected["reply"],),
            ),
        ),
        _result(actual),
    )

    detail = json.loads(score.detail)
    assert detail["structured_score"] == 10 / 11
    assert detail["reply_score"] == 1.0
    assert score.score == 0.7 * (10 / 11) + 0.3
    assert detail["errors"] == ["wrong_field:gender"]


def test_final_task_score_accepts_semantic_preference_alias() -> None:
    expected = _expected_final()
    actual = dict(expected)
    actual["preferences"] = ["颈肩放松"]
    score = TaskCorrectnessScorer().score(
        _final_sample(
            expected,
            ReplyExpectations((), (), (), (expected["reply"],)),
        ),
        _result(actual),
    )

    detail = json.loads(score.detail)
    assert detail["structured_score"] == 1.0


def test_final_task_score_aggregates_reply_type_acts_and_facts() -> None:
    expected = _expected_final()
    actual = dict(expected)
    actual["reply_type"] = "ask_start_time"
    actual["reply"] = "请问您什么时候过来？"
    score = TaskCorrectnessScorer().score(
        _final_sample(
            expected,
            ReplyExpectations(
                required_acts=("inform_technician_available", "request_confirmation"),
                forbidden_acts=("claim_booking_success",),
                required_fields=("technician_name", "start_time", "duration_minutes"),
                references=(expected["reply"],),
            ),
        ),
        _result(actual),
    )

    detail = json.loads(score.detail)
    assert detail["structured_score"] == 1.0
    assert detail["reply_score"] < 1.0
    assert "wrong_reply_type" in detail["errors"]
    assert any(error.startswith("reply_semantic:") for error in detail["errors"])
    assert any(error.startswith("reply_faithfulness:") for error in detail["errors"])


def test_tool_task_score_checks_action_tool_and_all_arguments() -> None:
    expected = {
        "action": "tool_call",
        "tool_name": "find_technicians",
        "arguments": {
            "technician_name": "王芳",
            "start_time": "2026-06-09 14:00",
            "duration_minutes": 60,
            "gender": None,
            "preferences": ["肩颈"],
        },
    }
    actual = {
        "action": "tool_call",
        "tool_name": "find_technicians",
        "arguments": {**expected["arguments"], "gender": "male"},
    }
    sample = Sample("tool", "tool_call", {}, expected, [], [])

    score = TaskCorrectnessScorer().score(sample, _result(actual))

    detail = json.loads(score.detail)
    assert score.score == 6 / 7
    assert detail["errors"] == ["wrong_argument:gender"]
