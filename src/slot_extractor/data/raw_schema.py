from __future__ import annotations

from copy import deepcopy
from typing import Any

TIME_PATTERN = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$"
GENDERS = ["female", "male", None]
STATUSES = ["not_checked", "available", "unavailable", "not_found", "no_match"]
REPLY_TYPES = [
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


def _closed(properties: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": properties,
        "required": list(properties),
        "additionalProperties": False,
    }


def _nullable_string() -> dict[str, Any]:
    return {"type": ["string", "null"]}


def _arguments() -> dict[str, Any]:
    return _closed(
        {
            "technician_name": _nullable_string(),
            "start_time": {"type": "string", "pattern": TIME_PATTERN},
            "duration_minutes": {"type": "integer", "minimum": 1},
            "gender_preference": {"type": ["string", "null"], "enum": GENDERS},
            "preferences": {"type": "array", "items": {"type": "string"}},
        }
    )


def _final() -> dict[str, Any]:
    return _closed(
        {
            "action": {"type": "string", "const": "final"},
            "gender_preference": {"type": ["string", "null"], "enum": GENDERS},
            "technician_gender": {"type": ["string", "null"], "enum": GENDERS},
            "start_time": {
                "anyOf": [
                    {"type": "string", "pattern": TIME_PATTERN},
                    {"type": "null"},
                ]
            },
            "duration_minutes": {
                "anyOf": [
                    {"type": "integer", "minimum": 1},
                    {"type": "null"},
                ]
            },
            "preferences": {"type": "array", "items": {"type": "string"}},
            "technician_name": _nullable_string(),
            "technician_status": {"type": "string", "enum": STATUSES},
            "confirmation": {"type": "boolean"},
            "info_complete": {"type": "boolean"},
            "unrelated": {"type": "boolean"},
            "missing_info": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["start_time", "duration_minutes"],
                },
            },
            "reply_type": {"type": "string", "enum": REPLY_TYPES},
            "reply": _nullable_string(),
        }
    )


def _tool_call() -> dict[str, Any]:
    return _closed(
        {
            "action": {"type": "string", "const": "tool_call"},
            "tool_name": {"type": "string", "const": "find_technicians"},
            "arguments": {"$ref": "#/$defs/arguments"},
        }
    )


def _history_item() -> dict[str, Any]:
    natural_user = _closed(
        {
            "role": {"type": "string", "const": "user"},
            "content": {"type": "string", "minLength": 1},
        }
    )
    natural_assistant = _closed(
        {
            "role": {"type": "string", "const": "assistant"},
            "content": {"type": "string", "minLength": 1},
        }
    )
    function = _closed(
        {
            "name": {"type": "string", "const": "find_technicians"},
            "arguments": {"type": "string", "minLength": 2},
        }
    )
    call = _closed(
        {
            "id": {"type": "string", "minLength": 1},
            "type": {"type": "string", "const": "function"},
            "function": function,
        }
    )
    tool_assistant = _closed(
        {
            "role": {"type": "string", "const": "assistant"},
            "content": {"type": "null"},
            "tool_calls": {
                "type": "array",
                "items": call,
                "minItems": 1,
                "maxItems": 1,
            },
        }
    )
    tool_result = _closed(
        {
            "role": {"type": "string", "const": "tool"},
            "tool_call_id": {"type": "string", "minLength": 1},
            "content": {"type": "string", "minLength": 2},
        }
    )
    return {"anyOf": [natural_user, natural_assistant, tool_assistant, tool_result]}


def _state() -> dict[str, Any]:
    properties = {
        "start_time": _final()["properties"]["start_time"],
        "duration_minutes": _final()["properties"]["duration_minutes"],
        "preferences": {"type": "array", "items": {"type": "string"}},
        "technician_name": _nullable_string(),
        "technician_status": {"type": "string", "enum": STATUSES},
        "confirmation": {"type": "boolean"},
        "info_complete": {"type": "boolean"},
        "unrelated": {"type": "boolean"},
        "missing_info": {
            "type": "array",
            "items": {
                "type": "string",
                "enum": ["start_time", "duration_minutes"],
            },
        },
        "last_reply_type": {"type": "string", "enum": REPLY_TYPES},
        "gender_preference": {"type": ["string", "null"], "enum": GENDERS},
        "technician_gender": {"type": ["string", "null"], "enum": GENDERS},
    }
    return {"anyOf": [{"type": "null"}, _closed(properties)]}


def raw_response_schema() -> dict[str, Any]:
    schema = _closed(
        {
            "id": {"type": "string", "minLength": 1},
            "output_kind": {"type": "string", "enum": ["final", "tool_call"]},
            "conversation_kind": {
                "type": "string",
                "enum": ["single_turn", "multi_turn"],
            },
            "tags": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": [
                        "追问",
                        "工具调用",
                        "最终 JSON",
                        "确认",
                        "无关",
                        "相对时间",
                        "多义短词",
                        "多轮改口",
                        "易混边界",
                        "幻觉陷阱",
                    ],
                },
                "minItems": 1,
            },
            "input": {"$ref": "#/$defs/input"},
            "expected": {"$ref": "#/$defs/expected"},
            "dpo_targets": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["P4", "P6", "P5", "P7", "P2P3"],
                },
            },
        }
    )
    schema["$defs"] = {
        "arguments": _arguments(),
        "expected": {"anyOf": [_final(), _tool_call()]},
        "history_item": _history_item(),
        "input": _closed(
            {
                "history": {
                    "type": "array",
                    "items": {"$ref": "#/$defs/history_item"},
                },
                "user_input": _nullable_string(),
                "current_time": {"type": "string", "pattern": TIME_PATTERN},
                "current_state": _state(),
                "available_tools": {
                    "type": "array",
                    "items": {"type": "string", "const": "find_technicians"},
                },
            }
        ),
    }
    return deepcopy(schema)
