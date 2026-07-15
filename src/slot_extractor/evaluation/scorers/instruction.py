# src/slot_extractor/evaluation/scorers/instruction.py
from __future__ import annotations

from slot_extractor.schemas.output import (
    OutputValidationError,
    parse_model_json,
    validate_final_output,
    validate_tool_call_output,
)
from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


class InstructionScorer:
    dimension = "protocol"

    def applies_to(self, sample: Sample) -> bool:
        return True

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        try:
            data = parse_model_json(result.text)
            if data.get("action") == "tool_call":
                validate_tool_call_output(data)
            elif data.get("action") == "final":
                validate_final_output(data)
            else:
                raise OutputValidationError("action must be 'final' or 'tool_call'")
        except OutputValidationError as exc:
            return DimensionScore(self.dimension, 0.0, False, str(exc))
        return DimensionScore(self.dimension, 1.0, True, "raw JSON and schema are valid")
