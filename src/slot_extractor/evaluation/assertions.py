# src/slot_extractor/evaluation/assertions.py
from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from slot_extractor.schemas.output import FINAL_FIELDS, TOOL_CALL_FIELDS
from slot_extractor.schemas.sample import Sample

DIMENSION_OF = {
    "field": "field_extraction",
    "args": "field_extraction",
    "tool_name": "tool_call",
    "action": "intent",
    "unrelated": "intent",
    "confirmation": "intent",
    "no_field_outside_schema": "instruction",
    "no_hallucinated_entity": "hallucination",
    "not_a_tool_call": "restraint",
    "within_business_hours": "restraint",
}


@dataclass(frozen=True)
class AssertionResult:
    expression: str
    passed: bool
    dimension: str
    detail: str


def _json_object(content: Any) -> dict[str, Any] | None:
    if not isinstance(content, str):
        return None
    try:
        value = json.loads(content)
    except json.JSONDecodeError:
        return None
    return value if isinstance(value, dict) else None


def _user_evidence(sample: Sample) -> list[str]:
    messages = [
        str(turn.get("content", ""))
        for turn in sample.input.get("history", [])
        if turn.get("role") == "user"
    ]
    user_input = sample.input.get("user_input")
    if isinstance(user_input, str):
        messages.append(user_input)
    current_state = sample.input.get("current_state")
    if isinstance(current_state, dict) and isinstance(
        current_state.get("technician_name"), str
    ):
        messages.append(current_state["technician_name"])
    return messages


def _latest_tool_result(sample: Sample) -> dict[str, Any] | None:
    for turn in reversed(sample.input.get("history", [])):
        if turn.get("role") == "tool":
            return _json_object(turn.get("content"))
    return None


def _available_tool_technicians(sample: Sample) -> set[str]:
    result = _latest_tool_result(sample)
    if not result:
        return set()
    if result.get("mode") == "specific" and result.get("status") == "available":
        technician = result.get("technician")
        if isinstance(technician, dict) and isinstance(technician.get("name"), str):
            return {technician["name"]}
    if result.get("mode") == "search" and result.get("status") == "matched":
        candidates = result.get("candidates")
        if isinstance(candidates, list):
            return {
                candidate["name"]
                for candidate in candidates
                if isinstance(candidate, dict) and isinstance(candidate.get("name"), str)
            }
    return set()


def _requested_tool_technicians(sample: Sample) -> set[str]:
    result = _latest_tool_result(sample)
    if not result or result.get("mode") != "specific":
        return set()
    requested = result.get("requested_technician")
    return {requested} if isinstance(requested, str) and requested else set()


def _current_state_technicians(sample: Sample) -> set[str]:
    current_state = sample.input.get("current_state")
    if not isinstance(current_state, dict):
        return set()
    name = current_state.get("technician_name")
    return {name} if isinstance(name, str) and name else set()


def checked_technicians(output: dict[str, Any]) -> list[str]:
    if output.get("action") == "tool_call":
        arguments = output.get("arguments")
        value = arguments.get("technician_name") if isinstance(arguments, dict) else None
    else:
        value = output.get("technician_name")
    return [value] if isinstance(value, str) and value else []


def hallucinated_entities(output: dict[str, Any], sample: Sample) -> list[str]:
    checked = checked_technicians(output)
    if not checked:
        return []

    if output.get("action") == "tool_call" or output.get("technician_status") == "not_checked":
        user_messages = _user_evidence(sample)
        return [name for name in checked if not any(name in message for message in user_messages)]

    if output.get("technician_status") == "available":
        allowed = _available_tool_technicians(sample) | _current_state_technicians(sample)
        return [name for name in checked if name not in allowed]
    if output.get("technician_status") in {"unavailable", "not_found"}:
        allowed = _requested_tool_technicians(sample) | _current_state_technicians(sample)
        return [name for name in checked if name not in allowed]

    return checked


def is_tool_call(output: dict[str, Any]) -> bool:
    return output.get("action") == "tool_call"


def within_business_hours(start_time: str, open_h: int = 9, close_h: int = 21) -> bool:
    if not isinstance(start_time, str):
        return False
    try:
        hour = int(start_time.split(" ")[1].split(":")[0])
    except (IndexError, ValueError):
        return False
    return open_h <= hour < close_h


def _no_field_outside_schema(output: dict[str, Any]) -> bool:
    fields = set(output)
    return fields == FINAL_FIELDS or fields == TOOL_CALL_FIELDS


def _format_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def evaluate_assertion(expression: str, output: dict[str, Any], sample: Sample) -> AssertionResult:
    expr = expression.strip()

    if expr == "no_field_outside_schema":
        ok = _no_field_outside_schema(output)
        return AssertionResult(expr, ok, "instruction", f"no_field_outside_schema={ok}")
    if expr == "no_hallucinated_entity":
        bad = hallucinated_entities(output, sample)
        return AssertionResult(expr, not bad, "hallucination", f"hallucinated={bad}")
    if expr == "not_a_tool_call":
        ok = not is_tool_call(output)
        return AssertionResult(expr, ok, "restraint", f"is_tool_call={is_tool_call(output)}")
    if expr == "within_business_hours":
        ok = within_business_hours(str(output.get("start_time", "")))
        return AssertionResult(expr, ok, "restraint", f"start_time={output.get('start_time')}")

    if "==" not in expr:
        return AssertionResult(
            expr, False, "instruction", "malformed: expected '==' or named predicate"
        )

    lhs, rhs = (part.strip() for part in expr.split("==", 1))
    if lhs.startswith("args."):
        key = lhs[len("args.") :]
        actual = _format_value(output.get("arguments", {}).get(key))
        return AssertionResult(
            expr, actual == rhs, "field_extraction", f"args.{key}={actual} expected={rhs}"
        )

    dimension = DIMENSION_OF.get(lhs, "field_extraction")
    value = output.get(lhs)
    actual = _format_value(value)
    return AssertionResult(expr, actual == rhs, dimension, f"{lhs}={actual} expected={rhs}")
