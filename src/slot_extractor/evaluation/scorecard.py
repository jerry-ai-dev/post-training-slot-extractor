# src/slot_extractor/evaluation/scorecard.py
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from slot_extractor.schemas.results import CaseResult, DimensionScore, Scorecard

DIMENSION_ORDER = [
    "instruction",
    "tool_call",
    "field_extraction",
    "hallucination",
    "intent",
    "restraint",
    "multiturn",
    "speed",
    "resource",
]


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
    return Scorecard(model=model, n=len(cases), dimensions=aggregated, cases=cases)


def render_scorecard(scorecard: Scorecard) -> str:
    lines = [
        f"===== Slot-Extractor 评估分数卡 (model={scorecard.model} / n={scorecard.n}) =====",
        "质量角度",
    ]
    labels = {
        "instruction": "指令 / 规则遵循",
        "tool_call": "工具调用准确性",
        "field_extraction": "字段抽取准确性",
        "hallucination": "幻觉率",
        "intent": "意图判定准确性",
        "restraint": "克制与约束遵守",
        "multiturn": "多轮鲁棒性",
        "speed": "速度",
        "resource": "资源",
    }
    for dimension in DIMENSION_ORDER:
        score = scorecard.dimensions[dimension]
        value = "n/a" if score.score is None else f"{score.score * 100:.1f}%"
        lines.append(f"  {labels[dimension]:<14} {value:<8} ({score.detail})")
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
