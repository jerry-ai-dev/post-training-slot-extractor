from __future__ import annotations

import json

FIND_TECHNICIANS = {
    "name": "find_technicians",
    "description": "按预约条件查询技师",
    "parameters": {
        "type": "object",
        "properties": {
            "technician_name": {"type": ["string", "null"]},
            "start_time": {"type": "string"},
            "duration_minutes": {"type": "integer"},
            "gender_preference": {"type": ["string", "null"], "enum": ["female", "male", None]},
            "preferences": {"type": "array", "items": {"type": "string"}},
        },
        "required": [
            "technician_name",
            "start_time",
            "duration_minutes",
            "gender_preference",
            "preferences",
        ],
    },
}
TOOL_SCHEMAS = {"find_technicians": FIND_TECHNICIANS}


def render_tools(available_tools: list[str]) -> str:
    unknown = set(available_tools) - set(TOOL_SCHEMAS)
    if unknown:
        raise ValueError(f"unknown tools: {sorted(unknown)}")
    return json.dumps(
        [TOOL_SCHEMAS[name] for name in available_tools], ensure_ascii=False, separators=(",", ":")
    )
