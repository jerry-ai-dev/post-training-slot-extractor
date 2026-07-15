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
    total_ms: float | None = None
    first_token_ms: float | None = None
    tokens_per_s: float | None = None


@dataclass(frozen=True)
class TimingSummary:
    """全样本原始时延统计（不卡阈值，供后续分析）。单位：毫秒 / tokens/s。"""

    count: int
    total_ms_mean: float | None
    total_ms_p50: float | None
    total_ms_p95: float | None
    total_ms_max: float | None
    total_ms_min: float | None
    first_token_ms_mean: float | None
    tokens_per_s_mean: float | None


@dataclass(frozen=True)
class Scorecard:
    model: str
    n: int
    dimensions: dict[str, DimensionScore]
    cases: list[CaseResult]
    timing: TimingSummary | None = None
