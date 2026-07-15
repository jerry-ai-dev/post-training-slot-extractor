# src/slot_extractor/schemas/output.py
from __future__ import annotations

import json
import re
from typing import Any

FINAL_FIELDS = {
    "action",
    "gender",
    "start_time",
    "duration",
    "project",
    "preference",
    "technician_name",
    "confirmation",
    "info_complete",
    "unrelated",
    "missing_info",
}
TOOL_CALL_FIELDS = {"action", "tool_name", "arguments"}
TIME_PATTERN = re.compile(r"^(未知|\d{4}-\d{2}-\d{2} \d{2}:\d{2})$")


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
    for key in ["gender", "start_time", "duration", "project", "preference", "technician_name"]:
        if not isinstance(data[key], str):
            raise OutputValidationError(f"{key} must be a string")
    if not TIME_PATTERN.match(data["start_time"]):
        raise OutputValidationError("start_time must be '未知' or 'YYYY-MM-DD HH:MM'")
    for key in ["confirmation", "info_complete", "unrelated"]:
        if not isinstance(data[key], bool):
            raise OutputValidationError(f"{key} must be a boolean")
    if not isinstance(data["missing_info"], list) or not all(
        isinstance(item, str) for item in data["missing_info"]
    ):
        raise OutputValidationError("missing_info must be a list of strings")


def validate_tool_call_output(data: dict[str, Any]) -> None:
    _ensure_exact_fields(data, TOOL_CALL_FIELDS)
    if data["action"] != "tool_call":
        raise OutputValidationError("tool output must set action='tool_call'")
    if not isinstance(data["tool_name"], str) or not data["tool_name"]:
        raise OutputValidationError("tool_name must be a non-empty string")
    if not isinstance(data["arguments"], dict):
        raise OutputValidationError("arguments must be an object")
