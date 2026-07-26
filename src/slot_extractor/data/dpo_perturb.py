from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from copy import deepcopy
from datetime import datetime, timedelta
from typing import Any

from slot_extractor.data.fake_names import FAKE_NAMES
from slot_extractor.data.raw_sample import RawSample
from slot_extractor.data.sft_render import compact_json, render_sft
from slot_extractor.schemas.output import validate_final_output, validate_tool_call_output


class PerturbationError(ValueError):
    pass


def _validate_output(output: dict[str, Any]) -> None:
    if output.get("action") == "final":
        validate_final_output(output)
    elif output.get("action") == "tool_call":
        validate_tool_call_output(output)
    else:
        raise PerturbationError("rejected has unsupported action")


def _tool_arguments(raw: RawSample) -> dict[str, Any]:
    if raw.expected["action"] == "tool_call":
        return deepcopy(raw.expected["arguments"])
    output = raw.expected
    return {
        "technician_name": output.get("technician_name"),
        "start_time": output.get("start_time"),
        "duration_minutes": output.get("duration_minutes"),
        "gender_preference": output.get("gender_preference"),
        "preferences": deepcopy(output.get("preferences", [])),
    }


def _final_from_arguments(arguments: dict[str, Any]) -> dict[str, Any]:
    return {
        "action": "final",
        "gender_preference": arguments["gender_preference"],
        "technician_gender": None,
        "start_time": arguments["start_time"],
        "duration_minutes": arguments["duration_minutes"],
        "preferences": deepcopy(arguments["preferences"]),
        "technician_name": arguments["technician_name"] or "李娜",
        "technician_status": "available",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "confirm_available",
        "reply": "已找到可预约技师，请确认。",
    }


def perturb_p4(raw: RawSample) -> dict[str, Any]:
    rejected = deepcopy(raw.expected)
    tool_names: set[str] = set()
    for turn in raw.input.get("history", []):
        if turn.get("role") == "tool":
            tool_names.update(_collect_names(json.loads(turn["content"])))
    start = int(hashlib.sha256(raw.id.encode()).hexdigest(), 16) % len(FAKE_NAMES)
    name = next(
        (
            FAKE_NAMES[(start + offset) % len(FAKE_NAMES)]
            for offset in range(len(FAKE_NAMES))
            if FAKE_NAMES[(start + offset) % len(FAKE_NAMES)] not in tool_names
        ),
        None,
    )
    if name is None:
        raise PerturbationError("fake-name pool exhausted")
    old = rejected.get("technician_name")
    rejected["technician_name"] = name
    if rejected.get("technician_status") in {"not_found", "no_match"}:
        rejected["technician_status"] = "available"
        rejected["reply_type"] = "confirm_available"
        rejected["reply"] = f"{name}可以预约，请确认。"
    elif isinstance(rejected.get("reply"), str) and old:
        rejected["reply"] = rejected["reply"].replace(old, name)
    return rejected


def _collect_names(value: Any) -> set[str]:
    if isinstance(value, dict):
        own = {value["name"]} if isinstance(value.get("name"), str) else set()
        return own | set().union(*(_collect_names(item) for item in value.values()), set())
    if isinstance(value, list):
        return set().union(*(_collect_names(item) for item in value), set())
    return set()


def perturb_p6(raw: RawSample) -> dict[str, Any]:
    if raw.expected["action"] == "tool_call":
        return _final_from_arguments(_tool_arguments(raw))
    current = raw.input["current_time"]
    arguments = _tool_arguments(raw)
    arguments["start_time"] = arguments["start_time"] or current
    arguments["duration_minutes"] = arguments["duration_minutes"] or 60
    return {"action": "tool_call", "tool_name": "find_technicians", "arguments": arguments}


def perturb_p5(raw: RawSample) -> dict[str, Any]:
    rejected = deepcopy(raw.expected)
    if rejected["confirmation"]:
        rejected["confirmation"] = False
        rejected["reply_type"] = "ask_start_time"
        rejected["reply"] = "请再确认预约时间。"
    elif rejected["unrelated"]:
        rejected["unrelated"] = False
        rejected["reply_type"] = "ask_start_time_and_duration"
        rejected["reply"] = "请提供预约时间和时长。"
        rejected["missing_info"] = ["start_time", "duration_minutes"]
    else:
        raise PerturbationError("P5 requires confirmation or unrelated final")
    return rejected


def perturb_p7(raw: RawSample) -> dict[str, Any]:
    arguments = _tool_arguments(raw)
    current = datetime.strptime(raw.input["current_time"], "%Y-%m-%d %H:%M")
    user_input = raw.input.get("user_input", "")
    target_day = current + timedelta(days=1 if "明天" in user_input else 0)
    arguments["start_time"] = arguments["start_time"] or target_day.strftime("%Y-%m-%d 14:00")
    arguments["duration_minutes"] = arguments["duration_minutes"] or 60
    return {"action": "tool_call", "tool_name": "find_technicians", "arguments": arguments}


def perturb_p2p3(raw: RawSample) -> dict[str, Any]:
    rejected = deepcopy(raw.expected)
    container = rejected["arguments"] if rejected["action"] == "tool_call" else rejected
    if container.get("duration_minutes") is not None:
        container["duration_minutes"] += 30
    elif container.get("start_time"):
        value = datetime.strptime(container["start_time"], "%Y-%m-%d %H:%M")
        container["start_time"] = (value + timedelta(days=1)).strftime("%Y-%m-%d %H:%M")
    else:
        raise PerturbationError("P2P3 has no perturbable field")
    return rejected


PERTURBERS: dict[str, Callable[[RawSample], dict[str, Any]]] = {
    "P4": perturb_p4,
    "P6": perturb_p6,
    "P5": perturb_p5,
    "P7": perturb_p7,
    "P2P3": perturb_p2p3,
}


def perturb(raw: RawSample, target: str) -> dict[str, object]:
    if target not in raw.dpo_targets:
        raise PerturbationError(f"{raw.id} does not declare {target}")
    rejected = PERTURBERS[target](raw)
    _validate_output(rejected)
    if rejected == raw.expected:
        raise PerturbationError("rejected equals chosen")
    base = render_sft(raw)
    return {
        "system": base["system"],
        "tools": base["tools"],
        "conversations": base["conversations"][:-1],
        "chosen": {"from": "gpt", "value": compact_json(raw.expected)},
        "rejected": {"from": "gpt", "value": compact_json(rejected)},
    }
