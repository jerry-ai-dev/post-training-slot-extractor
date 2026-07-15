from __future__ import annotations

import json
from datetime import datetime, time
from pathlib import Path
from typing import Any

from slot_extractor.evaluation.assertions import evaluate_assertion
from slot_extractor.schemas.output import (
    OutputValidationError,
    validate_final_output,
    validate_tool_call_output,
)
from slot_extractor.schemas.sample import Sample

_TIME_FORMAT = "%Y-%m-%d %H:%M"
_INPUT_FIELDS = {
    "history",
    "current_state",
    "user_input",
    "current_time",
    "available_tools",
}
_STATE_FIELDS = {
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
    "last_reply_type",
}
_PLAN_FIELDS = (
    "gender",
    "start_time",
    "duration_minutes",
    "preferences",
    "technician_name",
    "technician_status",
)


class DatasetContractError(ValueError):
    pass


def load_dataset_contract(path: str | Path) -> dict[str, Any]:
    contract_path = Path(path)
    try:
        contract = json.loads(contract_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise DatasetContractError(f"{contract_path} is not valid JSON: {exc.msg}") from exc
    if not isinstance(contract, dict):
        raise DatasetContractError(f"{contract_path} must contain a JSON object")
    return contract


def _parse_clock(value: str) -> time:
    try:
        return datetime.strptime(value, "%H:%M").time()
    except ValueError as exc:
        raise DatasetContractError(f"invalid contract clock: {value}") from exc


def _validate_time(sample_id: str, field: str, value: Any, contract: dict[str, Any]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, str):
        return [f"{sample_id}: {field} must be a string or null"]
    try:
        parsed = datetime.strptime(value, _TIME_FORMAT)
    except ValueError:
        return [f"{sample_id}: {field} must use YYYY-MM-DD HH:MM"]
    hours = contract["business_hours"]
    if not (_parse_clock(hours["open"]) <= parsed.time() < _parse_clock(hours["close"])):
        return [f"{sample_id}: {field} {value} is outside business hours"]
    return []


def _validate_values(
    sample_id: str,
    values: dict[str, Any],
    contract: dict[str, Any],
    prefix: str = "",
) -> list[str]:
    errors: list[str] = []
    if values.get("gender") not in contract["fields"]["gender"]["allowed_values"]:
        errors.append(f"{sample_id}: {prefix}gender has unsupported value")
    duration = values.get("duration_minutes")
    if duration is not None and (
        isinstance(duration, bool) or not isinstance(duration, int) or duration <= 0
    ):
        errors.append(f"{sample_id}: {prefix}duration_minutes must be a positive integer or null")
    preferences = values.get("preferences")
    if not isinstance(preferences, list) or not all(
        isinstance(item, str) and item.strip() for item in preferences
    ):
        errors.append(f"{sample_id}: {prefix}preferences must be a string array")
    errors.extend(
        _validate_time(sample_id, f"{prefix}start_time", values.get("start_time"), contract)
    )
    return errors


def _validate_current_state(sample: Sample, contract: dict[str, Any]) -> list[str]:
    state = sample.input.get("current_state")
    if state is None:
        return []
    if not isinstance(state, dict):
        return [f"{sample.id}: current_state must be an object or null"]
    errors: list[str] = []
    if set(state) != _STATE_FIELDS:
        errors.append(
            f"{sample.id}: current_state fields mismatch: "
            f"missing={sorted(_STATE_FIELDS - set(state))}, "
            f"extra={sorted(set(state) - _STATE_FIELDS)}"
        )
        return errors
    errors.extend(_validate_values(sample.id, state, contract, "current_state."))
    statuses = contract["fields"]["technician_status"]["allowed_values"]
    if state.get("technician_status") not in statuses:
        errors.append(f"{sample.id}: current_state.technician_status is unsupported")
    if not isinstance(state.get("confirmation"), bool):
        errors.append(f"{sample.id}: current_state.confirmation must be boolean")
    if not isinstance(state.get("info_complete"), bool):
        errors.append(f"{sample.id}: current_state.info_complete must be boolean")
    if not isinstance(state.get("unrelated"), bool):
        errors.append(f"{sample.id}: current_state.unrelated must be boolean")
    if not isinstance(state.get("missing_info"), list):
        errors.append(f"{sample.id}: current_state.missing_info must be a list")
    reply_types = contract["fields"]["reply_type"]["allowed_values"]
    if state.get("last_reply_type") is not None and state.get("last_reply_type") not in reply_types:
        errors.append(f"{sample.id}: current_state.last_reply_type is unsupported")
    return errors


def _tool_exchanges(history: Any) -> list[dict[str, Any]]:
    if not isinstance(history, list):
        return []
    calls: dict[str, dict[str, Any]] = {}
    exchanges: list[dict[str, Any]] = []
    for turn in history:
        if not isinstance(turn, dict):
            continue
        if turn.get("role") == "assistant":
            tool_calls = turn.get("tool_calls")
            if not isinstance(tool_calls, list):
                continue
            for call in tool_calls:
                function = call.get("function") if isinstance(call, dict) else None
                if not isinstance(function, dict):
                    continue
                try:
                    arguments = json.loads(function.get("arguments", ""))
                except (TypeError, json.JSONDecodeError):
                    continue
                call_id = call.get("id")
                if isinstance(call_id, str) and isinstance(arguments, dict):
                    calls[call_id] = {
                        "tool_name": function.get("name"),
                        "arguments": arguments,
                    }
        elif turn.get("role") == "tool":
            call_id = turn.get("tool_call_id")
            call = calls.get(call_id) if isinstance(call_id, str) else None
            if call is None:
                continue
            try:
                result = json.loads(turn.get("content", ""))
            except (TypeError, json.JSONDecodeError):
                continue
            if isinstance(result, dict):
                exchanges.append({**call, "result": result, "tool_call_id": call_id})
    return exchanges


def _latest_tool_exchange(sample: Sample) -> dict[str, Any] | None:
    exchanges = _tool_exchanges(sample.input.get("history"))
    return exchanges[-1] if exchanges else None


def _validate_tool_history(sample: Sample, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    for index, exchange in enumerate(_tool_exchanges(sample.input.get("history"))):
        tool_name = exchange.get("tool_name")
        if tool_name not in contract["tools"]:
            errors.append(f"{sample.id}: tool history[{index}] uses unsupported tool")
            continue
        arguments = exchange.get("arguments")
        expected_keys = set(contract["tools"][tool_name]["arguments"])
        if not isinstance(arguments, dict) or set(arguments) != expected_keys:
            errors.append(f"{sample.id}: tool history[{index}] arguments fields mismatch")
            continue
        errors.extend(
            _validate_values(sample.id, arguments, contract, f"tool_history[{index}].arguments.")
        )
    return errors


def _validate_tool_result_state(sample: Sample) -> list[str]:
    history = sample.input.get("history")
    if not isinstance(history, list) or not history or history[-1].get("role") != "tool":
        return []
    exchange = _latest_tool_exchange(sample)
    state = sample.input.get("current_state")
    if not isinstance(exchange, dict) or not isinstance(state, dict):
        return [f"{sample.id}: trailing tool result requires current_state"]
    arguments = exchange.get("arguments")
    if not isinstance(arguments, dict):
        return [f"{sample.id}: trailing tool result requires tool arguments"]
    errors: list[str] = []
    for field in ("gender", "start_time", "duration_minutes", "preferences", "technician_name"):
        if state.get(field) != arguments.get(field):
            errors.append(f"{sample.id}: current_state.{field} must match latest tool arguments")
    expected_meta = {
        "technician_status": "not_checked",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
    }
    for field, expected in expected_meta.items():
        if state.get(field) != expected:
            errors.append(f"{sample.id}: current_state.{field} must be {expected!r}")
    return errors


def _validate_input(sample: Sample, contract: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    actual_keys = set(sample.input)
    if actual_keys != _INPUT_FIELDS:
        errors.append(
            f"{sample.id}: input keys mismatch: "
            f"missing={sorted(_INPUT_FIELDS - actual_keys)}, "
            f"extra={sorted(actual_keys - _INPUT_FIELDS)}"
        )
    history = sample.input.get("history")
    if not isinstance(history, list):
        errors.append(f"{sample.id}: input.history must be a list")
        history = []
    for index, turn in enumerate(history):
        if not isinstance(turn, dict) or turn.get("role") not in {"user", "assistant", "tool"}:
            errors.append(f"{sample.id}: history[{index}] has invalid message role")

    current_time = sample.input.get("current_time")
    if not isinstance(current_time, str):
        errors.append(f"{sample.id}: input.current_time must be a string")
    else:
        try:
            datetime.strptime(current_time, _TIME_FORMAT)
        except ValueError:
            errors.append(f"{sample.id}: input.current_time must use YYYY-MM-DD HH:MM")

    expected_tools = list(contract["tools"])
    if sample.input.get("available_tools") != expected_tools:
        errors.append(f"{sample.id}: input.available_tools must equal {expected_tools}")

    user_input = sample.input.get("user_input")
    if user_input is None:
        if not history or history[-1].get("role") != "tool":
            errors.append(f"{sample.id}: input without user_input requires trailing tool result")
    elif not isinstance(user_input, str) or not user_input.strip():
        errors.append(f"{sample.id}: input.user_input must be non-empty text or null")

    errors.extend(_validate_current_state(sample, contract))
    errors.extend(_validate_tool_history(sample, contract))
    errors.extend(_validate_tool_result_state(sample))
    state = sample.input.get("current_state")
    if (
        isinstance(state, dict)
        and state.get("technician_status") != "not_checked"
        and not _tool_exchanges(history)
    ):
        errors.append(f"{sample.id}: checked current_state requires tool history evidence")
    return errors


def _tool_result_supports_final(expected: dict[str, Any], context: dict[str, Any] | None) -> bool:
    if not isinstance(context, dict):
        return False
    result = context.get("result")
    arguments = context.get("arguments")
    if not isinstance(result, dict) or not isinstance(arguments, dict):
        return False
    if any(
        expected.get(field) != arguments.get(field)
        for field in ("start_time", "duration_minutes", "preferences")
    ):
        return False
    status = expected.get("technician_status")
    mode = result.get("mode")
    result_status = result.get("status")
    if mode == "specific":
        if status != result_status:
            return False
        requested = result.get("requested_technician")
        if expected.get("technician_name") != requested:
            return False
        technician = result.get("technician")
        if status == "available":
            return isinstance(technician, dict) and (
                technician.get("name") == expected.get("technician_name")
                and technician.get("gender") == expected.get("gender")
            )
        if status == "unavailable":
            expected_gender = (
                technician.get("gender")
                if isinstance(technician, dict)
                else arguments.get("gender")
            )
            return expected.get("gender") == expected_gender
        return status == "not_found" and expected.get("gender") == arguments.get("gender")
    if mode == "search" and result_status == "no_match":
        return (
            status == "no_match"
            and expected.get("technician_name") is None
            and expected.get("gender") == arguments.get("gender")
        )
    if mode == "search" and result_status == "matched":
        candidates = result.get("candidates")
        if not isinstance(candidates, list) or len(candidates) != 1:
            return False
        candidate = candidates[0]
        return isinstance(candidate, dict) and status == "available" and (
            candidate.get("name") == expected.get("technician_name")
            and candidate.get("gender") == expected.get("gender")
        )
    return False


def _validate_reply_expectations(sample: Sample, contract: dict[str, Any]) -> list[str]:
    expectations = sample.reply_expectations
    if sample.expected.get("action") == "tool_call":
        return [] if expectations is None else [f"{sample.id}: tool_call has reply expectations"]
    if expectations is None:
        return [f"{sample.id}: final sample requires reply_expectations"]
    errors: list[str] = []
    allowed_acts = set(contract["reply"]["allowed_acts"])
    for act in (*expectations.required_acts, *expectations.forbidden_acts):
        if act not in allowed_acts:
            errors.append(f"{sample.id}: unsupported reply act {act!r}")
    allowed_fields = set(contract["reply"]["allowed_required_fields"])
    for field in expectations.required_fields:
        if field not in allowed_fields:
            errors.append(f"{sample.id}: unsupported required reply field {field!r}")
    if sample.expected.get("reply_type") != "handoff" and not expectations.references:
        errors.append(f"{sample.id}: user-facing final requires reply references")
    return errors


def _state_matches_plan(state: Any, expected: dict[str, Any]) -> bool:
    return isinstance(state, dict) and all(
        state.get(field) == expected.get(field) for field in _PLAN_FIELDS
    )


def _validate_final(sample: Sample, contract: dict[str, Any]) -> list[str]:
    expected = sample.expected
    try:
        validate_final_output(expected)
    except OutputValidationError as exc:
        return [f"{sample.id}: {exc}"]
    errors = _validate_values(sample.id, expected, contract)
    status = expected["technician_status"]
    if status not in contract["fields"]["technician_status"]["allowed_values"]:
        errors.append(f"{sample.id}: unsupported technician_status {status!r}")

    missing = expected["missing_info"]
    required_missing = [
        field
        for field in contract["completion"]["missing_slot_order"]
        if expected.get(field) is None
    ]
    if expected["unrelated"]:
        if expected["reply_type"] != "handoff" or expected["reply"] is not None:
            errors.append(f"{sample.id}: unrelated output requires handoff with null reply")
        if expected["confirmation"] or expected["info_complete"] or missing:
            errors.append(f"{sample.id}: unrelated output requires false/false/[] state")
        if status != "not_checked":
            errors.append(f"{sample.id}: unrelated output requires not_checked")
        return errors

    if status == "not_checked":
        if missing != required_missing:
            errors.append(f"{sample.id}: missing_info must match required slot gaps")
        if expected["info_complete"] != (not required_missing):
            errors.append(f"{sample.id}: info_complete conflicts with missing slots")
        reply_by_missing = {
            ("start_time",): "ask_start_time",
            ("duration_minutes",): "ask_duration",
            ("start_time", "duration_minutes"): "ask_start_time_and_duration",
        }
        required_reply = reply_by_missing.get(tuple(missing))
        if required_reply and expected["reply_type"] != required_reply:
            errors.append(
                f"{sample.id}: missing_info requires reply_type={required_reply!r}"
            )
    else:
        if required_missing or not expected["info_complete"] or missing:
            errors.append(f"{sample.id}: checked result requires complete information")
        context = _latest_tool_exchange(sample)
        state = sample.input.get("current_state")
        reply_type = expected["reply_type"]
        immediate_types = {
            "available": "confirm_available",
            "unavailable": "inform_unavailable",
            "not_found": "inform_not_found",
            "no_match": "inform_no_match",
        }
        if reply_type == immediate_types.get(status):
            if not _tool_result_supports_final(expected, context):
                errors.append(f"{sample.id}: {reply_type} requires matching latest tool result")
        elif reply_type == "appointment_paused":
            if status != "available" or not _state_matches_plan(state, expected) or (
                not isinstance(state, dict) or state.get("last_reply_type") != "confirm_available"
            ):
                errors.append(f"{sample.id}: appointment_paused requires pending available state")
        elif reply_type == "booking_authorized":
            if not expected["confirmation"] or status != "available" or not _state_matches_plan(
                state, expected
            ):
                errors.append(f"{sample.id}: current_state must match confirmed plan")
        elif reply_type == "acknowledge_result":
            if not expected["confirmation"] or status not in {
                "unavailable",
                "not_found",
                "no_match",
            } or not _state_matches_plan(state, expected):
                errors.append(f"{sample.id}: acknowledge_result requires matching failed state")
        else:
            errors.append(f"{sample.id}: reply_type does not match technician status")

    if expected["confirmation"] and expected["reply_type"] not in {
        "booking_authorized",
        "acknowledge_result",
    }:
        errors.append(f"{sample.id}: confirmation=true has incompatible reply_type")
    return errors


def _validate_tool_call(sample: Sample, contract: dict[str, Any]) -> list[str]:
    expected = sample.expected
    try:
        validate_tool_call_output(expected)
    except OutputValidationError as exc:
        return [f"{sample.id}: {exc}"]
    errors: list[str] = []
    tool_name = expected["tool_name"]
    if tool_name not in contract["tools"]:
        return [f"{sample.id}: unsupported tool_name {tool_name!r}"]
    arguments = expected["arguments"]
    expected_keys = set(contract["tools"][tool_name]["arguments"])
    if set(arguments) != expected_keys:
        errors.append(f"{sample.id}: tool arguments fields mismatch")
    errors.extend(_validate_values(sample.id, arguments, contract, "arguments."))
    if arguments.get("start_time") is None:
        errors.append(f"{sample.id}: tool call requires start_time")
    if arguments.get("duration_minutes") is None:
        errors.append(f"{sample.id}: tool call requires duration_minutes")
    return errors


def validate_sample_against_contract(sample: Sample, contract: dict[str, Any]) -> list[str]:
    errors = _validate_input(sample, contract)
    expected_action = "tool_call" if sample.layer == "tool_call" else "final"
    if sample.expected.get("action") != expected_action:
        errors.append(
            f"{sample.id}: layer {sample.layer!r} requires action {expected_action!r}"
        )
    if sample.expected.get("action") == "final":
        errors.extend(_validate_final(sample, contract))
    elif sample.expected.get("action") == "tool_call":
        errors.extend(_validate_tool_call(sample, contract))
    else:
        errors.append(f"{sample.id}: expected action is unsupported")
    errors.extend(_validate_reply_expectations(sample, contract))
    for expression in sample.assertions:
        result = evaluate_assertion(expression, sample.expected, sample)
        if not result.passed:
            errors.append(f"{sample.id}: assertion failed: {expression}: {result.detail}")
    return errors


def validate_dataset_against_contract(samples: list[Sample], contract: dict[str, Any]) -> None:
    errors: list[str] = []
    ids: set[str] = set()
    chains: dict[str, dict[int, Sample]] = {}
    for sample in samples:
        if sample.id in ids:
            errors.append(f"duplicate sample id: {sample.id}")
        ids.add(sample.id)
        errors.extend(validate_sample_against_contract(sample, contract))
        if sample.chain_id is not None and sample.step is not None:
            chains.setdefault(sample.chain_id, {})[sample.step] = sample

    for chain_id, steps in chains.items():
        ordered = sorted(steps)
        if ordered != list(range(min(ordered), max(ordered) + 1)):
            errors.append(f"{chain_id}: chain steps must be contiguous")
        for step in ordered:
            sample = steps[step]
            if sample.expected.get("action") != "tool_call":
                continue
            successor = steps.get(step + 1)
            if successor is None:
                errors.append(f"{sample.id}: tool_call requires a successor")
                continue
            context = _latest_tool_exchange(successor)
            if not isinstance(context, dict) or context.get("tool_name") != sample.expected.get(
                "tool_name"
            ) or context.get("arguments") != sample.expected.get("arguments"):
                errors.append(
                    f"{sample.id}: successor history must contain the exact tool exchange"
                )

    if errors:
        raise DatasetContractError("\n".join(errors))
