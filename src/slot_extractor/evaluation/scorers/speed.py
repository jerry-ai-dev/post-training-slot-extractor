# src/slot_extractor/evaluation/scorers/speed.py
from __future__ import annotations

from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


class SpeedScorer:
    dimension = "speed"

    def __init__(self, first_token_good_ms: float = 800, total_good_ms: float = 1500) -> None:
        self._first_token_good_ms = first_token_good_ms
        self._total_good_ms = total_good_ms

    def applies_to(self, sample: Sample) -> bool:
        return True

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        first_ok = (
            result.first_token_ms is not None and result.first_token_ms <= self._first_token_good_ms
        )
        total_ok = result.total_ms <= self._total_good_ms
        score = (int(first_ok) + int(total_ok)) / 2
        return DimensionScore(
            self.dimension,
            score,
            score == 1.0,
            f"first_token_ms={result.first_token_ms} total_ms={result.total_ms} "
            f"tokens_per_s={result.tokens_per_s}",
        )
