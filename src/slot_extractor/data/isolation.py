from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterable
from typing import Any


class IsolationError(ValueError):
    pass


def _normalize(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value.strip())
    if isinstance(value, dict):
        return {key: _normalize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize(item) for item in value]
    return value


def _record_input(record: Any) -> dict[str, Any]:
    value = record.input if hasattr(record, "input") else record.get("input", {})
    return value if isinstance(value, dict) else {}


def input_fingerprint(record: Any) -> str:
    input_obj = _record_input(record)
    payload = {
        "history": input_obj.get("history", []),
        "user_input": input_obj.get("user_input"),
        "current_time": input_obj.get("current_time"),
    }
    encoded = json.dumps(
        _normalize(payload), ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _record_id(record: Any) -> str:
    return str(record.id if hasattr(record, "id") else record.get("id", "<unknown>"))


def assert_no_eval_overlap(train_records: Iterable[Any], eval_records: Iterable[Any]) -> None:
    evaluation = {input_fingerprint(record): _record_id(record) for record in eval_records}
    for record in train_records:
        fingerprint = input_fingerprint(record)
        if fingerprint in evaluation:
            raise IsolationError(
                f"train {_record_id(record)} overlaps eval {evaluation[fingerprint]}: {fingerprint}"
            )
