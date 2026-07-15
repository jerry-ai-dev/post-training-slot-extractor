# src/slot_extractor/schemas/results.py
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class GenerationResult:
    text: str
    model: str
    prefill_ms: float | None
    first_token_ms: float | None
    total_ms: float
    output_tokens: int | None = None
    tokens_per_s: float | None = None
    raw: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DimensionScore:
    dimension: str
    score: float | None
    passed: bool | None
    detail: str


@dataclass(frozen=True)
class CaseResult:
    sample_id: str
    layer: str
    model_output: str
    dimensions: dict[str, DimensionScore]


@dataclass(frozen=True)
class Scorecard:
    model: str
    n: int
    dimensions: dict[str, DimensionScore]
    cases: list[CaseResult]
