import json

from slot_extractor.schemas.dataset_contract import validate_sample_against_contract
from slot_extractor.schemas.sample import ReplyExpectations, Sample

CONTRACT = {
    "version": "2.1",
    "business_hours": {"open": "09:00", "close": "21:00"},
    "fields": {
        "gender": {"allowed_values": ["female", "male", None]},
        "technician_status": {
            "allowed_values": [
                "not_checked",
                "available",
                "unavailable",
                "not_found",
                "no_match",
            ]
        },
        "reply_type": {
            "allowed_values": [
                "handoff",
                "ask_start_time",
                "ask_duration",
                "ask_start_time_and_duration",
                "confirm_available",
                "inform_unavailable",
                "inform_not_found",
                "inform_no_match",
                "booking_authorized",
                "acknowledge_result",
                "appointment_paused",
            ]
        },
    },
    "completion": {
        "allowed_missing_slots": ["start_time", "duration_minutes"],
        "missing_slot_order": ["start_time", "duration_minutes"],
    },
    "reply": {
        "allowed_acts": [
            "ask_for_start_time",
            "ask_for_duration",
            "inform_technician_available",
            "request_confirmation",
            "inform_technician_unavailable",
            "inform_technician_not_found",
            "inform_no_match",
            "acknowledge_booking_authorization",
            "acknowledge_result",
            "acknowledge_pause",
            "claim_booking_success",
        ],
        "allowed_required_fields": [
            "technician_name",
            "start_time",
            "duration_minutes",
            "preferences",
        ],
    },
    "tools": {
        "find_technicians": {
            "arguments": [
                "technician_name",
                "start_time",
                "duration_minutes",
                "gender",
                "preferences",
            ]
        }
    },
}


def _expectations(
    required_acts: tuple[str, ...] = ("inform_technician_available", "request_confirmation"),
    forbidden_acts: tuple[str, ...] = ("claim_booking_success",),
    required_fields: tuple[str, ...] = (
        "technician_name",
        "start_time",
        "duration_minutes",
    ),
) -> ReplyExpectations:
    return ReplyExpectations(
        required_acts=required_acts,
        forbidden_acts=forbidden_acts,
        required_fields=required_fields,
        references=("王芳技师明天下午2点有空，可以安排60分钟，您确认吗？",),
    )


def _final(**overrides: object) -> dict:
    value = {
        "action": "final",
        "gender": "female",
        "start_time": "2026-06-09 14:00",
        "duration_minutes": 60,
        "preferences": [],
        "technician_name": "王芳",
        "technician_status": "available",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "confirm_available",
        "reply": "王芳技师明天下午2点有空，可以安排60分钟，您确认吗？",
    }
    value.update(overrides)
    return value


def _state(**overrides: object) -> dict:
    value = {
        "gender": None,
        "start_time": "2026-06-09 14:00",
        "duration_minutes": 60,
        "preferences": [],
        "technician_name": "王芳",
        "technician_status": "not_checked",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "last_reply_type": None,
    }
    value.update(overrides)
    return value


def _tool_history(status: str = "available") -> list[dict]:
    arguments = {
            "technician_name": "王芳",
            "start_time": "2026-06-09 14:00",
            "duration_minutes": 60,
            "gender": None,
            "preferences": [],
    }
    result = {
            "mode": "specific",
            "status": status,
            "requested_technician": "王芳",
            "technician": {"name": "王芳", "gender": "female"},
    }
    return [
        {"role": "user", "content": "明天下午两点，60分钟，找王芳"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "find_technicians",
                        "arguments": json.dumps(arguments, ensure_ascii=False, separators=(",", ":")),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps(result, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _checked_history(status: str = "available") -> list[dict]:
    reply = {
        "available": "王芳技师明天下午2点有空，可以安排60分钟，您确认吗？",
        "unavailable": "王芳技师明天下午2点没有空，您想调整吗？",
    }[status]
    return [*_tool_history(status), {"role": "assistant", "content": reply}]


def _sample(
    expected: dict,
    *,
    current_state: dict | None = None,
    history: list[dict] | None = None,
    user_input: str | None = None,
    expectations: ReplyExpectations | None = None,
) -> Sample:
    return Sample(
        id="case",
        layer="tool_call" if expected["action"] == "tool_call" else "final",
        input={
            "history": history or [],
            "current_state": current_state,
            "user_input": user_input,
            "current_time": "2026-06-08 10:00",
            "available_tools": ["find_technicians"],
        },
        expected=expected,
        assertions=[],
        tags=[],
        reply_expectations=expectations,
    )


def test_available_final_accepts_matching_tool_history() -> None:
    sample = _sample(
        _final(),
        current_state=_state(),
        history=_tool_history(),
        expectations=_expectations(),
    )
    assert validate_sample_against_contract(sample, CONTRACT) == []


def test_available_final_without_tool_history_is_rejected() -> None:
    errors = validate_sample_against_contract(
        _sample(_final(), current_state=_state(), expectations=_expectations()),
        CONTRACT,
    )
    assert any("confirm_available requires matching latest tool result" in error for error in errors)


def test_paused_plan_uses_pending_current_state_without_latest_tool_result() -> None:
    expected = _final(
        reply_type="appointment_paused",
        reply="好的，暂时不给您预约。",
    )
    sample = _sample(
        expected,
        current_state=_state(
            gender="female",
            technician_status="available",
            last_reply_type="confirm_available",
        ),
        history=_checked_history(),
        user_input="先不了",
        expectations=_expectations(
            required_acts=("acknowledge_pause",),
            required_fields=(),
        ),
    )
    assert validate_sample_against_contract(sample, CONTRACT) == []


def test_confirmation_requires_matching_pending_state() -> None:
    expected = _final(
        confirmation=True,
        reply_type="booking_authorized",
        reply="好的，正在为您办理预约。",
    )
    errors = validate_sample_against_contract(
        _sample(
            expected,
            current_state=_state(
                duration_minutes=90,
                technician_status="available",
                last_reply_type="confirm_available",
            ),
            user_input="确认",
            expectations=_expectations(
                required_acts=("acknowledge_booking_authorization",),
                required_fields=(),
            ),
        ),
        CONTRACT,
    )
    assert any("current_state must match confirmed plan" in error for error in errors)


def test_missing_time_requires_ask_start_time_reply_type() -> None:
    expected = _final(
        gender=None,
        start_time=None,
        technician_name=None,
        technician_status="not_checked",
        info_complete=False,
        missing_info=["start_time"],
        reply_type="ask_duration",
        reply="请问您想按摩多长时间呢？",
    )
    errors = validate_sample_against_contract(
        _sample(
            expected,
            user_input="想按摩60分钟",
            expectations=_expectations(
                required_acts=("ask_for_start_time",),
                required_fields=(),
            ),
        ),
        CONTRACT,
    )
    assert any("missing_info requires reply_type='ask_start_time'" in error for error in errors)


def test_handoff_requires_null_reply() -> None:
    expected = _final(
        gender=None,
        start_time=None,
        duration_minutes=None,
        technician_name=None,
        technician_status="not_checked",
        info_complete=False,
        unrelated=True,
        missing_info=[],
        reply_type="handoff",
        reply=None,
    )
    sample = _sample(
        expected,
        user_input="今天天气怎么样",
        expectations=_expectations(required_acts=(), forbidden_acts=(), required_fields=()),
    )
    assert validate_sample_against_contract(sample, CONTRACT) == []


def test_tool_call_rejects_unknown_duration() -> None:
    expected = {
        "action": "tool_call",
        "tool_name": "find_technicians",
        "arguments": {
            "technician_name": "王芳",
            "start_time": "2026-06-09 14:00",
            "duration_minutes": None,
            "gender": None,
            "preferences": [],
        },
    }
    errors = validate_sample_against_contract(
        _sample(expected, user_input="明天下午两点找王芳，时长没定"),
        CONTRACT,
    )
    assert any("duration_minutes" in error for error in errors)


def test_every_turn_requires_full_runtime_input_keys() -> None:
    sample = _sample(
        _final(),
        current_state=_state(),
        history=_tool_history(),
        expectations=_expectations(),
    )
    sample.input.pop("available_tools")
    errors = validate_sample_against_contract(sample, CONTRACT)
    assert any("input keys" in error for error in errors)


def test_reply_expectations_reject_unknown_act_and_field() -> None:
    sample = _sample(
        _final(),
        current_state=_state(),
        history=_tool_history(),
        expectations=_expectations(
            required_acts=("unknown_act",),
            required_fields=("unknown_field",),
        ),
    )
    errors = validate_sample_against_contract(sample, CONTRACT)
    assert any("unsupported reply act" in error for error in errors)
    assert any("unsupported required reply field" in error for error in errors)
