from __future__ import annotations

from copy import deepcopy

import pytest

from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record


def _record() -> dict[str, object]:
    return {
        "id": "phase03-ask-001",
        "output_kind": "final",
        "conversation_kind": "single_turn",
        "tags": ["追问", "hard"],
        "input": {
            "history": [],
            "user_input": "明天下午",
            "current_time": "2026-07-26 10:00",
            "current_state": None,
            "available_tools": ["find_technicians"],
        },
        "expected": {"action": "final"},
        "dpo_targets": ["P7"],
    }


def test_parse_raw_sample() -> None:
    sample = raw_sample_from_record(_record())
    assert isinstance(sample, RawSample)
    assert sample.category == "追问"
    assert sample.dpo_targets == ("P7",)


def test_reject_out_of_category_dpo_target() -> None:
    record = deepcopy(_record())
    record["dpo_targets"] = ["P4"]
    with pytest.raises(ValueError, match="dpo_targets"):
        raw_sample_from_record(record)


def test_requires_exactly_seven_top_level_fields() -> None:
    record = _record()
    record["extra"] = True
    with pytest.raises(ValueError, match="top-level fields"):
        raw_sample_from_record(record)
