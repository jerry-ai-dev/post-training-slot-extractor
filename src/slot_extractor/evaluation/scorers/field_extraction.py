# src/slot_extractor/evaluation/scorers/field_extraction.py
from __future__ import annotations

from slot_extractor.schemas.output import OutputValidationError, parse_model_json
from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


class FieldExtractionScorer:
    dimension = "field_extraction"

    def applies_to(self, sample: Sample) -> bool:
        return sample.layer in {"final", "multi_turn"}

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        expected_items = {
            key: value
            for key, value in sample.expected.items()
            if key not in {"action", "tool_name", "arguments"}
        }
        if not expected_items:
            return DimensionScore(self.dimension, None, None, "no final fields for this sample")
        try:
            actual = parse_model_json(result.text)
        except OutputValidationError as exc:
            return DimensionScore(self.dimension, 0.0, False, str(exc))
        hits = [key for key, value in expected_items.items() if actual.get(key) == value]
        score = len(hits) / len(expected_items)
        missing = sorted(set(expected_items) - set(hits))
        return DimensionScore(
            self.dimension,
            score,
            score == 1.0,
            f"matched={sorted(hits)} mismatched={missing}",
        )
