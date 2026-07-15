# src/slot_extractor/schemas/sample.py
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slot_extractor.utils.jsonl import read_jsonl

Layer = Literal["final", "tool_call", "multi_turn"]


@dataclass(frozen=True)
class ReplyExpectations:
    required_acts: tuple[str, ...]
    forbidden_acts: tuple[str, ...]
    required_fields: tuple[str, ...]
    references: tuple[str, ...]


@dataclass(frozen=True)
class Sample:
    id: str
    layer: Layer
    input: dict[str, Any]
    expected: dict[str, Any]
    assertions: list[str]
    tags: list[str]
    reply_expectations: ReplyExpectations | None = None
    chain_id: str | None = None
    step: int | None = None


def _require_dict(record: dict[str, Any], key: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{record.get('id', '<unknown>')} missing object field: {key}")
    return value


def _require_list_of_str(record: dict[str, Any], key: str) -> list[str]:
    value = record.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{record.get('id', '<unknown>')} missing string-list field: {key}")
    return value


_VALID_ROLES = {"user", "assistant", "tool"}


def _validate_tool_calls(sample_id: str, index: int, value: Any) -> set[str]:
    if not isinstance(value, list) or len(value) != 1:
        raise ValueError(f"{sample_id} history[{index}].tool_calls must contain one call")
    call = value[0]
    if not isinstance(call, dict) or set(call) != {"id", "type", "function"}:
        raise ValueError(f"{sample_id} history[{index}] tool call shape is invalid")
    if not isinstance(call.get("id"), str) or not call["id"]:
        raise ValueError(f"{sample_id} history[{index}] tool call id must be non-empty")
    if call.get("type") != "function":
        raise ValueError(f"{sample_id} history[{index}] tool call type must be function")
    function = call.get("function")
    if not isinstance(function, dict) or set(function) != {"name", "arguments"}:
        raise ValueError(f"{sample_id} history[{index}] tool function shape is invalid")
    if not isinstance(function.get("name"), str) or not function["name"]:
        raise ValueError(f"{sample_id} history[{index}] tool name must be non-empty")
    try:
        arguments = json.loads(function.get("arguments", ""))
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"{sample_id} history[{index}] tool arguments must be a JSON object string"
        ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"{sample_id} history[{index}] tool arguments must decode to an object")
    return {call["id"]}


def _validate_history(sample_id: str, input_obj: dict[str, Any]) -> None:
    """验证用户、助手与工具事件组成的完整消息历史。"""
    history = input_obj.get("history", [])
    if not isinstance(history, list):
        raise ValueError(f"{sample_id} history must be a list of {{role, content}} turns")
    pending_tool_call_ids: set[str] = set()
    for index, turn in enumerate(history):
        if not isinstance(turn, dict):
            raise ValueError(f"{sample_id} history[{index}] must be an object")
        role = turn.get("role")
        if role not in _VALID_ROLES:
            raise ValueError(f"{sample_id} history[{index}].role must be one of {_VALID_ROLES}")
        content = turn.get("content")
        if role == "assistant" and "tool_calls" in turn:
            if content is not None:
                raise ValueError(
                    f"{sample_id} history[{index}] tool-calling assistant content must be null"
                )
            pending_tool_call_ids = _validate_tool_calls(
                sample_id, index, turn.get("tool_calls")
            )
        elif role == "tool":
            if set(turn) != {"role", "tool_call_id", "content"}:
                raise ValueError(f"{sample_id} history[{index}] tool message shape is invalid")
            call_id = turn.get("tool_call_id")
            if call_id not in pending_tool_call_ids:
                raise ValueError(f"{sample_id} history[{index}] tool_call_id has no pending call")
            try:
                result = json.loads(content)
            except (TypeError, json.JSONDecodeError) as exc:
                raise ValueError(
                    f"{sample_id} history[{index}] tool content must be a JSON object string"
                ) from exc
            if not isinstance(result, dict):
                raise ValueError(
                    f"{sample_id} history[{index}] tool content must decode to an object"
                )
            pending_tool_call_ids.remove(call_id)
        else:
            if set(turn) != {"role", "content"}:
                raise ValueError(f"{sample_id} history[{index}] natural message shape is invalid")
            if not isinstance(content, str) or not content.strip():
                raise ValueError(f"{sample_id} history[{index}].content must be non-empty text")
            if role == "assistant":
                try:
                    parsed = json.loads(content)
                except json.JSONDecodeError:
                    parsed = None
                if isinstance(parsed, (dict, list)):
                    raise ValueError(
                        f"{sample_id} history[{index}] assistant content must be natural language"
                    )

    if history and history[0]["role"] != "user":
        raise ValueError(f"{sample_id} history must start with user")
    allowed_next = {
        "user": {"assistant"},
        "assistant": {"user", "tool"},
        "tool": {"assistant"},
    }
    for previous, current in zip(history, history[1:], strict=False):
        if current["role"] not in allowed_next[previous["role"]]:
            raise ValueError(
                f"{sample_id} invalid history transition {previous['role']} -> {current['role']}"
            )
    user_input = input_obj.get("user_input")
    has_user_input = isinstance(user_input, str) and bool(user_input.strip())
    if user_input is not None and not has_user_input:
        raise ValueError(f"{sample_id} input.user_input must be a non-empty string when present")
    if has_user_input and history and history[-1]["role"] != "assistant":
        raise ValueError(f"{sample_id} history must end with assistant before user_input")
    if not has_user_input and (not history or history[-1]["role"] != "tool"):
        raise ValueError(f"{sample_id} input without user_input requires a trailing tool message")
    if pending_tool_call_ids:
        raise ValueError(f"{sample_id} history contains an unresolved tool call")


def _validate_context(sample_id: str, input_obj: dict[str, Any]) -> None:
    current_state = input_obj.get("current_state")
    if current_state is not None and not isinstance(current_state, dict):
        raise ValueError(f"{sample_id} input.current_state must be an object or null")


def _tuple_of_strings(sample_id: str, obj: dict[str, Any], key: str) -> tuple[str, ...]:
    value = obj.get(key)
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        raise ValueError(f"{sample_id} reply_expectations.{key} must be a string list")
    return tuple(value)


def _parse_reply_expectations(
    sample_id: str,
    record: dict[str, Any],
    expected: dict[str, Any],
    layer: Layer,
) -> ReplyExpectations | None:
    value = record.get("reply_expectations")
    if layer == "tool_call" or expected.get("action") == "tool_call":
        if value is not None:
            raise ValueError(f"{sample_id} tool_call samples must not define reply_expectations")
        return None
    if not isinstance(value, dict):
        raise ValueError(f"{sample_id} final samples must define reply_expectations")
    return ReplyExpectations(
        required_acts=_tuple_of_strings(sample_id, value, "required_acts"),
        forbidden_acts=_tuple_of_strings(sample_id, value, "forbidden_acts"),
        required_fields=_tuple_of_strings(sample_id, value, "required_fields"),
        references=_tuple_of_strings(sample_id, value, "references"),
    )


def sample_from_record(record: dict[str, Any]) -> Sample:
    sample_id = record.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample missing non-empty field: id")

    layer = record.get("layer")
    if layer not in {"final", "tool_call", "multi_turn"}:
        raise ValueError(f"{sample_id} has unsupported layer: {layer}")

    input_obj = _require_dict(record, "input")
    expected = _require_dict(record, "expected")
    _validate_context(sample_id, input_obj)
    _validate_history(sample_id, input_obj)
    reply_expectations = _parse_reply_expectations(sample_id, record, expected, layer)

    return Sample(
        id=sample_id,
        layer=layer,
        input=input_obj,
        expected=expected,
        reply_expectations=reply_expectations,
        assertions=_require_list_of_str(record, "assertions"),
        tags=_require_list_of_str(record, "tags"),
        chain_id=record.get("chain_id") if isinstance(record.get("chain_id"), str) else None,
        step=record.get("step") if isinstance(record.get("step"), int) else None,
    )


def load_samples(path: str | Path) -> list[Sample]:
    return [sample_from_record(record) for record in read_jsonl(path)]
