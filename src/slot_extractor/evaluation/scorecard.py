# src/slot_extractor/evaluation/scorecard.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from slot_extractor.schemas.results import (
    CaseResult,
    DimensionScore,
    Scorecard,
    TimingSummary,
)

DIMENSION_ORDER = [
    "protocol",
    "task_correctness",
    "resource",
]


def _percentile(sorted_values: list[float], pct: float) -> float:
    """线性插值分位数（pct∈[0,1]）。sorted_values 非空且已升序。"""
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = pct * (len(sorted_values) - 1)
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] + (sorted_values[high] - sorted_values[low]) * frac


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def summarize_timing(cases: list[CaseResult]) -> TimingSummary:
    totals = sorted(c.total_ms for c in cases if c.total_ms is not None)
    first_tokens = [c.first_token_ms for c in cases if c.first_token_ms is not None]
    throughputs = [c.tokens_per_s for c in cases if c.tokens_per_s is not None]
    return TimingSummary(
        count=len(totals),
        total_ms_mean=_mean(totals),
        total_ms_p50=_percentile(totals, 0.50) if totals else None,
        total_ms_p95=_percentile(totals, 0.95) if totals else None,
        total_ms_max=totals[-1] if totals else None,
        total_ms_min=totals[0] if totals else None,
        first_token_ms_mean=_mean(first_tokens),
        tokens_per_s_mean=_mean(throughputs),
    )


def aggregate_scorecard(model: str, cases: list[CaseResult]) -> Scorecard:
    aggregated: dict[str, DimensionScore] = {}
    for dimension in DIMENSION_ORDER:
        scores = [
            case.dimensions[dimension].score
            for case in cases
            if dimension in case.dimensions and case.dimensions[dimension].score is not None
        ]
        if not scores:
            aggregated[dimension] = DimensionScore(dimension, None, None, "n/a")
            continue
        score = sum(scores) / len(scores)
        aggregated[dimension] = DimensionScore(
            dimension,
            score,
            score == 1.0,
            f"mean over {len(scores)} applicable case(s)",
        )
    return Scorecard(
        model=model,
        n=len(cases),
        dimensions=aggregated,
        cases=cases,
        timing=summarize_timing(cases),
    )


def _fmt_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0f}ms"


def render_scorecard(scorecard: Scorecard) -> str:
    lines = [
        f"===== Appointment-Agent 评估分数卡 (model={scorecard.model} / n={scorecard.n}) =====",
        "质量角度",
    ]
    labels = {
        "protocol": "输出协议遵循",
        "task_correctness": "任务正确性",
        "resource": "资源",
    }
    for dimension in DIMENSION_ORDER:
        score = scorecard.dimensions[dimension]
        value = "n/a" if score.score is None else f"{score.score * 100:.1f}%"
        lines.append(f"  {labels[dimension]:<14} {value:<8} ({score.detail})")

    timing = scorecard.timing
    lines.append("速度 / 时延（原始统计，未卡阈值）")
    if timing is None or timing.count == 0:
        lines.append("  无计时数据")
    else:
        tps = "n/a" if timing.tokens_per_s_mean is None else f"{timing.tokens_per_s_mean:.1f}"
        lines.append(f"  样本数            {timing.count}")
        lines.append(f"  总时延 均值        {_fmt_ms(timing.total_ms_mean)}")
        lines.append(f"  总时延 p50        {_fmt_ms(timing.total_ms_p50)}")
        lines.append(f"  总时延 p95        {_fmt_ms(timing.total_ms_p95)}")
        lines.append(
            f"  总时延 最快/最慢   {_fmt_ms(timing.total_ms_min)} / {_fmt_ms(timing.total_ms_max)}"
        )
        lines.append(f"  首字延迟 均值      {_fmt_ms(timing.first_token_ms_mean)}")
        lines.append(f"  吞吐 均值          {tps} tok/s")
    return "\n".join(lines)


def write_scorecard_json(scorecard: Scorecard, report_dir: str | Path) -> Path:
    output_dir = Path(report_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"scorecard-{scorecard.model}.json"
    output_path.write_text(
        json.dumps(asdict(scorecard), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path
