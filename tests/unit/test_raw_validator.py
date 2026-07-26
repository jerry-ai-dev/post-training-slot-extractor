from copy import deepcopy

import pytest

from slot_extractor.data.raw_sample import raw_sample_from_record
from slot_extractor.data.raw_validator import RawValidationError, validate_raw_sample


def _final() -> dict:
    return {"id":"x","output_kind":"final","conversation_kind":"single_turn","tags":["追问"],"input":{"history":[],"user_input":"明天","current_time":"2026-07-26 10:00","current_state":None,"available_tools":["find_technicians"]},"expected":{"action":"final","gender_preference":None,"technician_gender":None,"start_time":"2026-07-27 14:00","duration_minutes":None,"preferences":[],"technician_name":None,"technician_status":"not_checked","confirmation":False,"info_complete":False,"unrelated":False,"missing_info":["duration_minutes"],"reply_type":"ask_duration","reply":"请问做多久？"},"dpo_targets":["P7"]}


def test_validate_valid_raw() -> None:
    validate_raw_sample(raw_sample_from_record(_final()))


def test_reject_inconsistent_missing_info() -> None:
    record = deepcopy(_final())
    record["expected"]["missing_info"] = []
    with pytest.raises(RawValidationError, match="missing_info"):
        validate_raw_sample(raw_sample_from_record(record))


def test_reject_wrong_conversation_kind() -> None:
    record = deepcopy(_final())
    record["conversation_kind"] = "multi_turn"
    with pytest.raises(RawValidationError, match="conversation_kind"):
        validate_raw_sample(raw_sample_from_record(record))
