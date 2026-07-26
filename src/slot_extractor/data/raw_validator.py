from __future__ import annotations

import json
from typing import Any

from slot_extractor.data.raw_sample import RawSample
from slot_extractor.schemas.output import validate_final_output, validate_tool_call_output


class RawValidationError(ValueError):
    pass


def _tool_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        names = {value["name"]} if isinstance(value.get("name"), str) else set()
        return names | set().union(*(_tool_names(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_tool_names(item) for item in value), set())
    return set()


def validate_raw_sample(sample: RawSample) -> None:
    try:
        if sample.output_kind == "final":
            validate_final_output(sample.expected)
            _validate_final_consistency(sample)
        else:
            validate_tool_call_output(sample.expected)
        expected_kind = "multi_turn" if sample.input.get("history") else "single_turn"
        if sample.conversation_kind != expected_kind:
            raise RawValidationError(f"{sample.id} conversation_kind must be {expected_kind}")
    except RawValidationError:
        raise
    except ValueError as exc:
        raise RawValidationError(str(exc)) from exc


def _validate_final_consistency(sample: RawSample) -> None:
    output = sample.expected
    missing = [field for field in ("start_time", "duration_minutes") if output[field] is None]
    expected_missing = [] if output["unrelated"] else missing
    if output["missing_info"] != expected_missing:
        raise RawValidationError(f"{sample.id} missing_info is inconsistent")
    if output["info_complete"] != (not missing):
        raise RawValidationError(f"{sample.id} info_complete is inconsistent")
    if output["unrelated"]:
        defaults = {
            "gender_preference": None,
            "technician_gender": None,
            "start_time": None,
            "duration_minutes": None,
            "preferences": [],
            "technician_name": None,
            "technician_status": "not_checked",
            "confirmation": False,
            "info_complete": False,
            "missing_info": [],
            "reply_type": "handoff",
            "reply": None,
        }
        if any(output[key] != value for key, value in defaults.items()):
            raise RawValidationError(f"{sample.id} unrelated final is inconsistent")
    if output["confirmation"] and (not output["info_complete"] or output["missing_info"]):
        raise RawValidationError(f"{sample.id} confirmation requires complete information")
    if (
        output["technician_status"] in {"not_found", "no_match"}
        and output["technician_name"] is not None
    ):
        raise RawValidationError(f"{sample.id} technician_name must be null for status")
    tool_names: set[str] = set()
    for turn in reversed(sample.input.get("history", [])):
        if turn.get("role") == "tool":
            tool_names = _tool_names(json.loads(turn["content"]))
            break
    if tool_names and output["technician_name"] not in tool_names | {None}:
        raise RawValidationError(f"{sample.id} technician_name is outside tool result")
