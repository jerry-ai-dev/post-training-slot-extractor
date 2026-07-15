# src/slot_extractor/evaluation/scorers/not_available.py
from __future__ import annotations

from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


class NotAvailableScorer:
    def __init__(self, dimension: str) -> None:
        self.dimension = dimension

    def applies_to(self, sample: Sample) -> bool:
        return True

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        return DimensionScore(self.dimension, None, None, "n/a in phase one")
