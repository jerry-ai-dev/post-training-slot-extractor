import pytest

from slot_extractor.schemas.output import (
    OutputValidationError,
    parse_model_json,
    validate_final_output,
    validate_tool_call_output,
)

VALID_FINAL = {
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
    "reply": "王芳技师明天下午2点有空，可以为您安排60分钟肩颈按摩，您确认吗？",
}


def test_parse_model_json_rejects_markdown_fence() -> None:
    with pytest.raises(OutputValidationError, match="must be raw JSON"):
        parse_model_json('```json\n{"action":"final"}\n```')


def test_validate_final_output_accepts_typed_schema() -> None:
    validate_final_output(VALID_FINAL)


def test_validate_final_output_accepts_null_unknowns() -> None:
    validate_final_output(
        {
            **VALID_FINAL,
            "gender": None,
            "start_time": None,
            "duration_minutes": None,
            "preferences": [],
            "technician_name": None,
            "technician_status": "not_checked",
            "info_complete": False,
            "missing_info": ["start_time", "duration_minutes"],
        }
    )


def test_validate_final_output_requires_reply_fields() -> None:
    missing_reply = dict(VALID_FINAL)
    missing_reply.pop("reply")

    with pytest.raises(OutputValidationError, match="schema fields"):
        validate_final_output(missing_reply)


def test_validate_final_output_allows_null_reply_for_handoff() -> None:
    validate_final_output(
        {
            **VALID_FINAL,
            "gender": None,
            "start_time": None,
            "duration_minutes": None,
            "preferences": [],
            "technician_name": None,
            "technician_status": "not_checked",
            "confirmation": False,
            "info_complete": False,
            "unrelated": True,
            "missing_info": [],
            "reply_type": "handoff",
            "reply": None,
        }
    )


def test_validate_final_output_rejects_null_reply_outside_handoff() -> None:
    with pytest.raises(OutputValidationError, match="reply"):
        validate_final_output({**VALID_FINAL, "reply": None})


def test_validate_final_output_rejects_unknown_reply_type() -> None:
    with pytest.raises(OutputValidationError, match="reply_type"):
        validate_final_output({**VALID_FINAL, "reply_type": "unknown"})


def test_validate_tool_call_output_rejects_reply_fields() -> None:
    with pytest.raises(OutputValidationError, match="schema fields"):
        validate_tool_call_output(
            {
                "action": "tool_call",
                "tool_name": "find_technicians",
                "arguments": {
                    "technician_name": None,
                    "start_time": "2026-06-09 14:00",
                    "duration_minutes": 60,
                    "gender": None,
                    "preferences": [],
                },
                "reply_type": None,
                "reply": None,
            }
        )


@pytest.mark.parametrize(
    ("updates", "error"),
    [
        ({"duration_minutes": "60分钟"}, "duration_minutes"),
        ({"duration_minutes": 0}, "duration_minutes"),
        ({"gender": "女"}, "gender"),
        ({"start_time": "明天下午两点"}, "start_time"),
        ({"preferences": "肩颈"}, "preferences"),
        ({"preferences": ["肩颈", 1]}, "preferences"),
        ({"technician_name": 1}, "technician_name"),
    ],
)
def test_validate_final_output_rejects_noncanonical_values(updates: dict, error: str) -> None:
    with pytest.raises(OutputValidationError, match=error):
        validate_final_output({**VALID_FINAL, **updates})


def test_validate_final_output_rejects_legacy_fields() -> None:
    legacy = dict(VALID_FINAL)
    legacy["duration"] = legacy.pop("duration_minutes")
    legacy["preference"] = legacy.pop("preferences")

    with pytest.raises(OutputValidationError, match="schema fields"):
        validate_final_output(legacy)


def test_validate_tool_call_output_accepts_typed_arguments() -> None:
    validate_tool_call_output(
        {
            "action": "tool_call",
            "tool_name": "find_technicians",
            "arguments": {
                "technician_name": None,
                "start_time": "2026-06-09 14:00",
                "duration_minutes": 60,
                "gender": None,
                "preferences": ["肩颈"],
            },
        }
    )


def test_validate_final_output_rejects_unknown_technician_status() -> None:
    with pytest.raises(OutputValidationError, match="technician_status"):
        validate_final_output({**VALID_FINAL, "technician_status": "maybe"})


def test_validate_final_output_rejects_removed_selection_required_status() -> None:
    with pytest.raises(OutputValidationError, match="technician_status"):
        validate_final_output({**VALID_FINAL, "technician_status": "selection_required"})
