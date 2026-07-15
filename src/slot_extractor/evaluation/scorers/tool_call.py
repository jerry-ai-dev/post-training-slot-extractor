# src/slot_extractor/evaluation/scorers/tool_call.py
from __future__ import annotations

from slot_extractor.schemas.output import OutputValidationError, parse_model_json
from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


class ToolCallScorer:
    dimension = "tool_call"

    def applies_to(self, sample: Sample) -> bool:
        return sample.layer == "tool_call"

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        try:
            actual = parse_model_json(result.text)
        except OutputValidationError as exc:
            return DimensionScore(self.dimension, 0.0, False, str(exc))
        expected_args = sample.expected.get("arguments", {})
        actual_args = actual.get("arguments", {})
        checks = {
            "action": actual.get("action") == sample.expected.get("action"),
            "tool_name": actual.get("tool_name") == sample.expected.get("tool_name"),
            "arguments": all(actual_args.get(key) == value for key, value in expected_args.items()),
            "no_extra_arguments": set(actual_args).issubset(set(expected_args)),
        }
        passed_count = sum(1 for value in checks.values() if value)
        score = passed_count / len(checks)
        failed = [key for key, value in checks.items() if not value]
        return DimensionScore(self.dimension, score, score == 1.0, f"failed={failed}")
