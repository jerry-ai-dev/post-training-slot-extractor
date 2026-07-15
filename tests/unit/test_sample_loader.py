import json
from pathlib import Path

import pytest

from slot_extractor.schemas.sample import (
    ReplyExpectations,
    Sample,
    load_samples,
    sample_from_record,
)


def _base_record() -> dict:
    return {
        "id": "case-1",
        "layer": "final",
        "input": {
            "history": [],
            "current_state": None,
            "user_input": "约明天两点",
            "current_time": "2026-06-08 10:00",
            "available_tools": ["find_technicians"],
        },
        "expected": {"action": "final"},
        "reply_expectations": {
            "required_acts": ["ask_for_duration"],
            "forbidden_acts": ["claim_booking_success"],
            "required_fields": [],
            "references": ["请问您想按摩多长时间呢？"],
        },
        "assertions": ["no_field_outside_schema"],
        "tags": ["smoke"],
    }


def test_load_samples_parses_reply_aware_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    record = _base_record()
    path.write_text(json.dumps(record, ensure_ascii=False) + "\n", encoding="utf-8")

    samples = load_samples(path)

    assert samples == [
        Sample(
            id="case-1",
            layer="final",
            input=record["input"],
            expected={"action": "final"},
            reply_expectations=ReplyExpectations(
                required_acts=("ask_for_duration",),
                forbidden_acts=("claim_booking_success",),
                required_fields=(),
                references=("请问您想按摩多长时间呢？",),
            ),
            assertions=["no_field_outside_schema"],
            tags=["smoke"],
        )
    ]


def test_load_samples_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"case-1","layer":"final","input":{},"assertions":[],"tags":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case-1.*expected"):
        load_samples(path)


def test_sample_accepts_natural_language_assistant_history() -> None:
    record = _base_record()
    record["input"]["history"] = [
        {"role": "user", "content": "想预约按摩"},
        {"role": "assistant", "content": "请问您想什么时候过来呢？"},
    ]
    record["input"]["user_input"] = "明天下午两点"

    sample = sample_from_record(record)

    assert sample.input["history"][-1]["role"] == "assistant"


def test_sample_rejects_json_assistant_history() -> None:
    record = _base_record()
    record["input"]["history"] = [
        {"role": "user", "content": "想预约按摩"},
        {
            "role": "assistant",
            "content": '{"action":"final","reply":"请问您想什么时候过来呢？"}',
        },
    ]
    record["input"]["user_input"] = "明天下午两点"

    with pytest.raises(ValueError, match="assistant content must be natural language"):
        sample_from_record(record)


def test_sample_accepts_tool_result_history_without_user_input() -> None:
    record = _base_record()
    record["id"] = "chain-1-result"
    record["chain_id"] = "chain-1"
    record["step"] = 2
    record["input"]["user_input"] = None
    record["input"]["current_state"] = {
        "gender": None,
        "start_time": "2026-06-09 14:00",
        "duration_minutes": 60,
        "preferences": [],
        "technician_name": "王芳",
        "technician_status": "not_checked",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "last_reply_type": None,
    }
    record["input"]["history"] = [
        {"role": "user", "content": "明天下午两点，60分钟，找王芳"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "find_technicians",
                        "arguments": '{"technician_name":"王芳","start_time":"2026-06-09 14:00","duration_minutes":60,"gender":null,"preferences":[]}',
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": '{"mode":"specific","status":"available","requested_technician":"王芳","technician":{"name":"王芳","gender":"female"}}',
        },
    ]

    sample = sample_from_record(record)

    assert sample.chain_id == "chain-1"
    assert sample.step == 2
    assert json.loads(sample.input["history"][-1]["content"])["status"] == "available"


def test_sample_rejects_missing_user_input_without_tool_result() -> None:
    record = _base_record()
    record["input"]["user_input"] = None

    with pytest.raises(ValueError, match="requires a trailing tool message"):
        sample_from_record(record)


def test_sample_rejects_orphan_tool_message() -> None:
    record = _base_record()
    record["input"]["history"] = [
        {"role": "user", "content": "明天下午两点找王芳做60分钟"},
        {"role": "tool", "tool_call_id": "missing", "content": '{"status":"available"}'},
    ]

    with pytest.raises(ValueError, match="has no pending call"):
        sample_from_record(record)


def test_sample_rejects_history_ending_with_user_when_user_input_exists() -> None:
    record = _base_record()
    record["input"]["history"] = [{"role": "user", "content": "第一条用户消息"}]
    record["input"]["user_input"] = "第二条用户消息"

    with pytest.raises(ValueError, match="history must end with assistant"):
        sample_from_record(record)


def test_tool_call_sample_rejects_reply_expectations() -> None:
    record = _base_record()
    record["layer"] = "tool_call"

    with pytest.raises(ValueError, match="tool_call samples must not define reply_expectations"):
        sample_from_record(record)
