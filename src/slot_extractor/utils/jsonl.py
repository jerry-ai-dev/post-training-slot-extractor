# src/slot_extractor/utils/jsonl.py
from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any


def read_jsonl(path: str | Path) -> Iterator[dict[str, Any]]:
    jsonl_path = Path(path)
    with jsonl_path.open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                value = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{jsonl_path}:{line_no} is not valid JSON: {exc.msg}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{jsonl_path}:{line_no} must contain a JSON object")
            yield value
