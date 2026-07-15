# src/slot_extractor/schemas/output.py
from __future__ import annotations

import json
import re
from typing import Any

FINAL_FIELDS = {
    "action",
    "gender",
    "start_time",
    "duration_minutes",
    "preferences",
    "technician_name",
    "technician_status",
    "confirmation",
    "info_complete",
    "unrelated",
    "missing_info",
    "reply_type",
    "reply",
}
TOOL_CALL_FIELDS = {"action", "tool_name", "arguments"}
TIME_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}$")
TOOL_ARGUMENT_FIELDS = {
    "technician_name",
    "start_time",
    "duration_minutes",
    "gender",
    "preferences",
}
MISSING_INFO_FIELDS = ("start_time", "duration_minutes")
TECHNICIAN_STATUSES = {
    "not_checked",
    "available",
    "unavailable",
    "not_found",
    "no_match",
}
REPLY_TYPES = {
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
}


class OutputValidationError(ValueError):
    pass


def parse_model_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```") or stripped.endswith("```"):
        raise OutputValidationError("model output must be raw JSON, not Markdown fenced JSON")
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise OutputValidationError(f"model output is not valid JSON: {exc.msg}") from exc
    if not isinstance(value, dict):
        raise OutputValidationError("model output must be a JSON object")
    return value


def _ensure_exact_fields(data: dict[str, Any], expected_fields: set[str]) -> None:
    actual_fields = set(data)
    if actual_fields != expected_fields:
        raise OutputValidationError(
            f"schema fields mismatch: missing={sorted(expected_fields - actual_fields)}, "
            f"extra={sorted(actual_fields - expected_fields)}"
        )


def validate_final_output(data: dict[str, Any]) -> None:
    _ensure_exact_fields(data, FINAL_FIELDS)
    if data["action"] != "final":
        raise OutputValidationError("final output must set action='final'")
    if data["gender"] not in {"female", "male", None}:
        raise OutputValidationError("gender must be 'female', 'male', or null")
    if data["start_time"] is not None and (
        not isinstance(data["start_time"], str) or not TIME_PATTERN.fullmatch(data["start_time"])
    ):
        raise OutputValidationError("start_time must be null or 'YYYY-MM-DD HH:MM'")
    duration = data["duration_minutes"]
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0
    ):
        raise OutputValidationError("duration_minutes must be a positive integer or null")
    preferences = data["preferences"]
    if not isinstance(preferences, list) or not all(
        isinstance(item, str) and bool(item.strip()) for item in preferences
    ):
        raise OutputValidationError("preferences must be a list of non-empty strings")
    if len(set(preferences)) != len(preferences):
        raise OutputValidationError("preferences must not contain duplicates")
    technician_name = data["technician_name"]
    if technician_name is not None and (
        not isinstance(technician_name, str) or not technician_name.strip()
    ):
        raise OutputValidationError("technician_name must be a non-empty string or null")
    if not isinstance(data["technician_status"], str):
        raise OutputValidationError("technician_status must be a string")
    if data["technician_status"] not in TECHNICIAN_STATUSES:
        raise OutputValidationError("technician_status has an unsupported value")
    for key in ["confirmation", "info_complete", "unrelated"]:
        if not isinstance(data[key], bool):
            raise OutputValidationError(f"{key} must be a boolean")
    if not isinstance(data["missing_info"], list) or not all(
        isinstance(item, str) for item in data["missing_info"]
    ):
        raise OutputValidationError("missing_info must be a list of strings")
    invalid_missing = set(data["missing_info"]) - set(MISSING_INFO_FIELDS)
    if invalid_missing:
        raise OutputValidationError("missing_info contains unsupported fields")
    ordered_missing = [field for field in MISSING_INFO_FIELDS if field in data["missing_info"]]
    if data["missing_info"] != ordered_missing:
        raise OutputValidationError("missing_info must use canonical order without duplicates")
    reply_type = data["reply_type"]
    if not isinstance(reply_type, str) or reply_type not in REPLY_TYPES:
        raise OutputValidationError("reply_type has an unsupported value")
    reply = data["reply"]
    if reply_type == "handoff":
        if reply is not None:
            raise OutputValidationError("handoff reply must be null")
    elif not isinstance(reply, str) or not reply.strip():
        raise OutputValidationError("reply must be a non-empty string outside handoff")
    elif reply.strip().startswith("```") or (
        reply.strip().startswith("{") and reply.strip().endswith("}")
    ):
        raise OutputValidationError("reply must be plain text")


def validate_tool_call_output(data: dict[str, Any]) -> None:
    _ensure_exact_fields(data, TOOL_CALL_FIELDS)
    if data["action"] != "tool_call":
        raise OutputValidationError("tool output must set action='tool_call'")
    if not isinstance(data["tool_name"], str) or not data["tool_name"]:
        raise OutputValidationError("tool_name must be a non-empty string")
    if not isinstance(data["arguments"], dict):
        raise OutputValidationError("arguments must be an object")
    arguments = data["arguments"]
    _ensure_exact_fields(arguments, TOOL_ARGUMENT_FIELDS)
    if arguments["technician_name"] is not None and (
        not isinstance(arguments["technician_name"], str)
        or not arguments["technician_name"].strip()
    ):
        raise OutputValidationError("technician_name must be a non-empty string or null")
    if not isinstance(arguments["start_time"], str) or not TIME_PATTERN.fullmatch(
        arguments["start_time"]
    ):
        raise OutputValidationError("tool start_time must use 'YYYY-MM-DD HH:MM'")
    duration = arguments["duration_minutes"]
    if isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0:
        raise OutputValidationError("tool duration_minutes must be a positive integer")
    if arguments["gender"] not in {"female", "male", None}:
        raise OutputValidationError("tool gender must be 'female', 'male', or null")
    preferences = arguments["preferences"]
    if not isinstance(preferences, list) or not all(
        isinstance(item, str) and bool(item.strip()) for item in preferences
    ):
        raise OutputValidationError("tool preferences must be a list of non-empty strings")
