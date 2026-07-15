from __future__ import annotations

import json
from collections import defaultdict

from slot_extractor.schemas.sample import Sample


def scenario_labels(sample: Sample) -> set[str]:
    labels: set[str] = set()
    expected = sample.expected
    if expected.get("action") == "tool_call":
        labels.add("tool_call")
    if any(
        turn.get("role") == "tool" and _is_json_object(turn.get("content"))
        for turn in sample.input.get("history", [])
        if isinstance(turn, dict)
    ):
        labels.add("tool_result")
    if sample.layer == "multi_turn":
        labels.add("multi_turn")
    if expected.get("unrelated") is True:
        labels.add("unrelated")
    if expected.get("confirmation") is True:
        labels.add("confirmation")
    if expected.get("info_complete") is False and expected.get("missing_info"):
        labels.add("missing_information")
    return labels


def _is_json_object(value: object) -> bool:
    if not isinstance(value, str):
        return False
    try:
        return isinstance(json.loads(value), dict)
    except json.JSONDecodeError:
        return False


def aggregate_scenario_slices(
    samples: list[Sample],
    task_scores: dict[str, float],
) -> dict[str, dict[str, float | int]]:
    grouped: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        score = task_scores.get(sample.id)
        if score is None:
            continue
        for label in scenario_labels(sample):
            grouped[label].append(score)
    return {
        label: {
            "count": len(scores),
            "task_correctness": sum(scores) / len(scores),
        }
        for label, scores in sorted(grouped.items())
    }
