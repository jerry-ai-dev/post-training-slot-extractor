# src/slot_extractor/schemas/sample.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from slot_extractor.utils.jsonl import read_jsonl

Layer = Literal["final", "tool_call", "multi_turn"]


@dataclass(frozen=True)
class Sample:
    id: str
    layer: Layer
    input: dict[str, Any]
    expected: dict[str, Any]
    assertions: list[str]
    gold_facts: dict[str, Any]
    tags: list[str]


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


def sample_from_record(record: dict[str, Any]) -> Sample:
    sample_id = record.get("id")
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample missing non-empty field: id")

    layer = record.get("layer")
    if layer not in {"final", "tool_call", "multi_turn"}:
        raise ValueError(f"{sample_id} has unsupported layer: {layer}")

    return Sample(
        id=sample_id,
        layer=layer,
        input=_require_dict(record, "input"),
        expected=_require_dict(record, "expected"),
        assertions=_require_list_of_str(record, "assertions"),
        gold_facts=record.get("gold_facts") if isinstance(record.get("gold_facts"), dict) else {},
        tags=_require_list_of_str(record, "tags"),
    )


def load_samples(path: str | Path) -> list[Sample]:
    return [sample_from_record(record) for record in read_jsonl(path)]
