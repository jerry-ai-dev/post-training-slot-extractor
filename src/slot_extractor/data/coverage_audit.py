from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

from slot_extractor.data.raw_sample import RawSample

REQUIRED_NEGATIVE_STATUSES = {"unavailable", "not_found", "no_match"}
REQUIRED_REPLY_TYPES = {
    "inform_unavailable",
    "inform_not_found",
    "inform_no_match",
    "acknowledge_result",
}


@dataclass(frozen=True)
class SemanticCoverageReport:
    missing_statuses: set[str]
    missing_reply_types: set[str]
    missing_confirmation_false: bool
    missing_minimal_technician_replacement: bool

    @property
    def ok(self) -> bool:
        return not (
            self.missing_statuses
            or self.missing_reply_types
            or self.missing_confirmation_false
            or self.missing_minimal_technician_replacement
        )


def _is_minimal_replacement(sample: RawSample) -> bool:
    state = sample.input.get("current_state")
    if sample.expected.get("action") != "tool_call" or not isinstance(state, dict):
        return False
    arguments = sample.expected.get("arguments")
    if not isinstance(arguments, dict):
        return False
    old_name = state.get("technician_name")
    new_name = arguments.get("technician_name")
    if not old_name or not new_name or old_name == new_name:
        return False
    inherited = ("start_time", "duration_minutes", "preferences", "gender_preference")
    return all(arguments.get(field) == state.get(field) for field in inherited)


def audit_semantic_coverage(samples: Iterable[RawSample]) -> SemanticCoverageReport:
    items = list(samples)
    statuses = {
        str(sample.expected.get("technician_status"))
        for sample in items
        if sample.expected.get("action") == "final"
    }
    reply_types = {
        str(sample.expected.get("reply_type"))
        for sample in items
        if sample.expected.get("action") == "final"
    }
    has_negative_confirmation = any(
        sample.expected.get("confirmation") is False
        and sample.expected.get("reply_type") in {"appointment_paused", "acknowledge_result"}
        for sample in items
    )
    return SemanticCoverageReport(
        missing_statuses=REQUIRED_NEGATIVE_STATUSES - statuses,
        missing_reply_types=REQUIRED_REPLY_TYPES - reply_types,
        missing_confirmation_false=not has_negative_confirmation,
        missing_minimal_technician_replacement=not any(
            _is_minimal_replacement(sample) for sample in items
        ),
    )
