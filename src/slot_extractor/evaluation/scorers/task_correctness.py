from __future__ import annotations

import json
from typing import Any

from slot_extractor.evaluation.scorers.preferences import (
    PreferenceMatcher,
    PreferenceSemanticScorer,
)
from slot_extractor.evaluation.scorers.reply_faithfulness import ReplyFaithfulnessScorer
from slot_extractor.evaluation.scorers.reply_semantic import ReplySemanticScorer
from slot_extractor.schemas.output import OutputValidationError, parse_model_json
from slot_extractor.schemas.results import DimensionScore, GenerationResult
from slot_extractor.schemas.sample import Sample

_FINAL_EXACT_FIELDS = (
    "action",
    "gender",
    "start_time",
    "duration_minutes",
    "technician_name",
    "technician_status",
    "confirmation",
    "info_complete",
    "unrelated",
    "missing_info",
)
_TOOL_EXACT_ARGUMENTS = (
    "technician_name",
    "start_time",
    "duration_minutes",
    "gender",
)


def _exact_match(actual: dict[str, Any], expected: dict[str, Any], key: str) -> bool:
    return key in actual and actual[key] == expected.get(key)


def _preference_list_score(expected: Any, actual: Any) -> float:
    if not isinstance(expected, list) or not isinstance(actual, list):
        return 0.0
    if not expected and not actual:
        return 1.0
    if not expected or not actual:
        return 0.0
    matcher = PreferenceMatcher()
    candidates = sorted(
        (
            (matcher.similarity(expected_value, actual_value), expected_index, actual_index)
            for expected_index, expected_value in enumerate(expected)
            for actual_index, actual_value in enumerate(actual)
        ),
        reverse=True,
    )
    matched_expected: set[int] = set()
    matched_actual: set[int] = set()
    for similarity, expected_index, actual_index in candidates:
        if similarity < matcher.similarity_threshold:
            break
        if expected_index in matched_expected or actual_index in matched_actual:
            continue
        matched_expected.add(expected_index)
        matched_actual.add(actual_index)
    matches = len(matched_expected)
    precision = matches / len(actual)
    recall = matches / len(expected)
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


class TaskCorrectnessScorer:
    dimension = "task_correctness"

    def applies_to(self, sample: Sample) -> bool:
        return True

    def score(self, sample: Sample, result: GenerationResult) -> DimensionScore:
        try:
            actual = parse_model_json(result.text)
        except OutputValidationError as exc:
            return DimensionScore(
                self.dimension,
                0.0,
                False,
                json.dumps(
                    {
                        "mode": sample.expected.get("action"),
                        "structured_score": 0.0,
                        "reply_score": 0.0,
                        "errors": [f"unparseable_output:{exc}"],
                    },
                    ensure_ascii=False,
                    separators=(",", ":"),
                ),
            )
        if sample.expected.get("action") == "tool_call":
            return self._score_tool_call(sample, result, actual)
        return self._score_final(sample, result, actual)

    def _score_tool_call(
        self,
        sample: Sample,
        result: GenerationResult,
        actual: dict[str, Any],
    ) -> DimensionScore:
        del result
        expected = sample.expected
        expected_args = expected.get("arguments", {})
        actual_args = actual.get("arguments")
        if not isinstance(actual_args, dict):
            actual_args = {}
        checks = [
            ("wrong_action", _exact_match(actual, expected, "action")),
            ("wrong_tool", _exact_match(actual, expected, "tool_name")),
        ]
        checks.extend(
            (
                f"wrong_argument:{key}",
                key in actual_args and actual_args[key] == expected_args.get(key),
            )
            for key in _TOOL_EXACT_ARGUMENTS
        )
        preference_score = _preference_list_score(
            expected_args.get("preferences"), actual_args.get("preferences")
        )
        score = (sum(passed for _, passed in checks) + preference_score) / 7
        errors = [name for name, passed in checks if not passed]
        if preference_score != 1.0:
            errors.append("wrong_argument:preferences")
        detail = {
            "mode": "tool_call",
            "structured_score": score,
            "reply_score": None,
            "errors": errors,
        }
        return DimensionScore(
            self.dimension,
            score,
            score == 1.0,
            json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
        )

    def _score_final(
        self,
        sample: Sample,
        result: GenerationResult,
        actual: dict[str, Any],
    ) -> DimensionScore:
        expected = sample.expected
        exact_checks = [
            (f"wrong_field:{key}", _exact_match(actual, expected, key))
            for key in _FINAL_EXACT_FIELDS
        ]
        preference = PreferenceSemanticScorer().score(sample, result)
        preference_score = 0.0 if preference.score is None else preference.score
        structured_score = (
            sum(passed for _, passed in exact_checks) + preference_score
        ) / (len(exact_checks) + 1)

        reply_type_score = 1.0 if _exact_match(actual, expected, "reply_type") else 0.0
        semantic = ReplySemanticScorer().score(sample, result)
        faithfulness = ReplyFaithfulnessScorer().score(sample, result)
        semantic_score = 0.0 if semantic.score is None else semantic.score
        faithfulness_score = 0.0 if faithfulness.score is None else faithfulness.score
        reply_score = (reply_type_score + semantic_score + faithfulness_score) / 3
        score = 0.70 * structured_score + 0.30 * reply_score

        errors = [name for name, passed in exact_checks if not passed]
        if preference_score != 1.0:
            errors.append("wrong_field:preferences")
        if reply_type_score != 1.0:
            errors.append("wrong_reply_type")
        if semantic.passed is False:
            errors.append(f"reply_semantic:{semantic.detail}")
        if faithfulness.passed is False:
            errors.append(f"reply_faithfulness:{faithfulness.detail}")
        detail = {
            "mode": "final",
            "structured_score": structured_score,
            "reply_score": reply_score,
            "errors": errors,
        }
        return DimensionScore(
            self.dimension,
            score,
            score == 1.0,
            json.dumps(detail, ensure_ascii=False, separators=(",", ":")),
        )
