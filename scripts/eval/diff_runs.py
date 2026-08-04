from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


def _mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def _dimension_mean(rows: list[dict[str, Any]], dimension: str) -> float | None:
    return _mean(
        [
            row["dimensions"][dimension]["score"]
            for row in rows
            if row["dimensions"].get(dimension, {}).get("score") is not None
        ]
    )


def _scenario_means(rows: list[dict[str, Any]]) -> dict[str, float]:
    labels = sorted(
        {label for row in rows for label in row.get("scenario_labels", row.get("tags", []))}
    )
    result: dict[str, float] = {}
    for label in labels:
        values = [
            row["dimensions"]["task_correctness"]["score"]
            for row in rows
            if label in row.get("scenario_labels", row.get("tags", []))
            and row["dimensions"].get("task_correctness", {}).get("score") is not None
        ]
        if values:
            result[label] = sum(values) / len(values)
    return result


def diff_predictions(
    left_rows: list[dict[str, Any]],
    right_rows: list[dict[str, Any]],
    *,
    left_run_id: str,
    right_run_id: str,
) -> dict[str, Any]:
    left = {row["id"]: row for row in left_rows}
    right = {row["id"]: row for row in right_rows}
    if set(left) != set(right):
        raise ValueError("sample id sets differ")
    ids = sorted(left)
    flipped_positive = [
        case_id
        for case_id in ids
        if not left[case_id]["effective_pass"] and right[case_id]["effective_pass"]
    ]
    flipped_negative = [
        case_id
        for case_id in ids
        if left[case_id]["effective_pass"] and not right[case_id]["effective_pass"]
    ]
    left_scenarios = _scenario_means(left_rows)
    right_scenarios = _scenario_means(right_rows)
    scenario_delta = {
        label: right_scenarios[label] - left_scenarios[label]
        for label in sorted(set(left_scenarios) & set(right_scenarios))
    }
    changes = []
    for case_id in [*flipped_positive, *flipped_negative]:
        changes.append(
            {
                "id": case_id,
                "direction": "positive" if case_id in flipped_positive else "negative",
                "left_output": left[case_id]["model_output"],
                "right_output": right[case_id]["model_output"],
                "left_failure_reasons": left[case_id]["failure_reasons"],
                "right_failure_reasons": right[case_id]["failure_reasons"],
            }
        )
    dimension_delta = {}
    for dimension in ("protocol", "task_correctness"):
        left_mean = _dimension_mean(left_rows, dimension)
        right_mean = _dimension_mean(right_rows, dimension)
        dimension_delta[dimension] = (
            None if left_mean is None or right_mean is None else right_mean - left_mean
        )
    return {
        "left_run_id": left_run_id,
        "right_run_id": right_run_id,
        "case_ids_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "flipped_positive": flipped_positive,
        "flipped_negative": flipped_negative,
        "net_effective_pass": len(flipped_positive) - len(flipped_negative),
        "dimension_delta": dimension_delta,
        "scenario_delta": scenario_delta,
        "changes": changes,
    }


def _load(path: Path) -> tuple[str, list[dict[str, Any]]]:
    predictions = path / "predictions.jsonl" if path.is_dir() else path
    rows = [json.loads(line) for line in predictions.read_text(encoding="utf-8").splitlines()]
    run_id = rows[0].get("run_id", predictions.parent.name) if rows else predictions.parent.name
    return run_id, rows


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Diff two phase04 prediction runs.")
    parser.add_argument("--left", type=Path, required=True)
    parser.add_argument("--right", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        left_id, left = _load(args.left)
        right_id, right = _load(args.right)
        result = diff_predictions(left, right, left_run_id=left_id, right_run_id=right_id)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(f"diff error: {error}", file=sys.stderr)
        return 2
    print(args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
