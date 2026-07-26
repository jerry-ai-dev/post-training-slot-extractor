import json

from test_raw_validator import _final

from slot_extractor.data.raw_sample import raw_sample_from_record
from slot_extractor.data.sft_render import render_sft


def test_render_sft_keeps_system_and_target_role() -> None:
    row = render_sft(raw_sample_from_record(_final()))
    assert set(row) == {"system", "tools", "conversations"}
    assert "当前状态：null" in row["system"]
    assert json.loads(row["tools"])[0]["name"] == "find_technicians"
    assert row["conversations"][-1]["from"] == "gpt"
    assert json.loads(row["conversations"][-1]["value"])["action"] == "final"


def test_unknown_tool_is_rejected() -> None:
    record = _final()
    record["input"]["available_tools"] = ["unknown"]
    try:
        render_sft(raw_sample_from_record(record))
    except ValueError as exc:
        assert "unknown" in str(exc)
    else:
        raise AssertionError("unknown tool accepted")
