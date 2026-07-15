# src/slot_extractor/evaluation/scorer.py
from __future__ import annotations

from typing import Protocol

from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample


class Scorer(Protocol):
    dimension: str

    def applies_to(self, sample: Sample) -> bool:
        raise NotImplementedError

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        raise NotImplementedError
