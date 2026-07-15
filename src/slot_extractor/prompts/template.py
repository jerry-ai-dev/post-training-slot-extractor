# src/slot_extractor/prompts/template.py
from __future__ import annotations

import json

from slot_extractor.prompts.rules import FINAL_SCHEMA_HINT, SYSTEM_RULES, TOOL_SCHEMA_HINT
from slot_extractor.schemas.sample import Sample


class PromptBuilder:
    def build(self, sample: Sample) -> str:
        input_json = json.dumps(sample.input, ensure_ascii=False, sort_keys=True, indent=2)
        return (
            f"{SYSTEM_RULES}\n"
            f"{FINAL_SCHEMA_HINT}\n"
            f"{TOOL_SCHEMA_HINT}\n"
            f"Sample ID: {sample.id}\n"
            f"Layer: {sample.layer}\n"
            f"Input JSON:\n{input_json}\n"
            "Output JSON:"
        )
