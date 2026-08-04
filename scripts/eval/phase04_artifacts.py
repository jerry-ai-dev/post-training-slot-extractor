from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EFFECTIVE_TASK_THRESHOLD = 0.95


def _effective(record: dict[str, Any]) -> tuple[bool, list[str]]:
    dimensions = record["dimensions"]
    protocol = dimensions.get("protocol", {}).get("score")
    task = dimensions.get("task_correctness", {}).get("score")
    reasons: list[str] = []
    if protocol != 1.0:
        reasons.append("protocol")
    if task is None or task < EFFECTIVE_TASK_THRESHOLD:
        reasons.append("task_correctness")
    return not reasons, reasons


def _kind_summary(rows: list[dict[str, Any]], output_kind: str) -> dict[str, Any]:
    selected = [row for row in rows if row.get("output_kind") == output_kind]
    dimensions: dict[str, float | None] = {}
    for name in ("protocol", "task_correctness"):
        values = [
            row["dimensions"].get(name, {}).get("score")
            for row in selected
            if row["dimensions"].get(name, {}).get("score") is not None
        ]
        dimensions[name] = sum(values) / len(values) if values else None
    return {"count": len(selected), "dimensions": dimensions}


def _scenario_labels(record: dict[str, Any]) -> list[str]:
    labels: set[str] = set()
    expected = record.get("expected", {})
    if record.get("output_kind") == "tool_call":
        labels.add("tool_call")
    if record.get("conversation_kind") == "multi_turn":
        labels.add("multi_turn")
    if expected.get("unrelated") is True:
        labels.add("unrelated")
    if expected.get("confirmation") is True:
        labels.add("confirmation")
    if expected.get("info_complete") is False and expected.get("missing_info"):
        labels.add("missing_information")
    history = record.get("input", {}).get("history") or []
    if any(turn.get("role") == "tool" for turn in history if isinstance(turn, dict)):
        labels.add("tool_result")
    return sorted(labels)


def write_phase04_artifacts(
    analysis: dict[str, Any],
    run_dir: Path,
    *,
    run_id: str,
    evaluation_environment: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    run_dir.mkdir(parents=True, exist_ok=True)
    predictions: list[dict[str, Any]] = []
    for record in analysis["records"]:
        effective, reasons = _effective(record)
        predictions.append(
            {
                "id": record["id"],
                "run_id": run_id,
                "output_kind": record["output_kind"],
                "conversation_kind": record["conversation_kind"],
                "tags": record.get("tags", []),
                "scenario_labels": _scenario_labels(record),
                "input": record["input"],
                "expected": record["expected"],
                "model_output": record["model_output"],
                "dimensions": record["dimensions"],
                "effective_pass": effective,
                "failure_reasons": reasons,
                "timing": record["timing"],
            }
        )
    predictions_path = run_dir / "predictions.jsonl"
    predictions_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in predictions),
        encoding="utf-8",
        newline="\n",
    )
    numerator = sum(row["effective_pass"] for row in predictions)
    denominator = len(predictions)
    scorecard = {
        "run_id": run_id,
        "model": analysis["model"],
        "status": "evaluated",
        "dataset": analysis["cases_path"],
        "backend_config": analysis["backend_config"],
        "n": denominator,
        "aggregate_dimensions": analysis["aggregate_dimensions"],
        "effective_pass": {
            "numerator": numerator,
            "denominator": denominator,
            "rate": numerator / denominator if denominator else None,
        },
        "output_kinds": {
            "final": _kind_summary(predictions, "final"),
            "tool_call": _kind_summary(predictions, "tool_call"),
        },
        "scenario_slices": analysis["scenario_slices"],
        "timing": analysis["aggregate_timing"],
        "evaluation_environment": evaluation_environment or {},
    }
    scorecard_path = run_dir / "scorecard.json"
    scorecard_path.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return predictions_path, scorecard_path
