import json
from copy import deepcopy

import pytest
from test_raw_validator import _final

from slot_extractor.data.dpo_perturb import perturb
from slot_extractor.data.raw_sample import raw_sample_from_record
from slot_extractor.schemas.output import validate_final_output, validate_tool_call_output


def _tool(targets: list[str]) -> dict:
    record = _final()
    record.update(id="tool", output_kind="tool_call", tags=["工具调用"])
    record["expected"] = {
        "action": "tool_call",
        "tool_name": "find_technicians",
        "arguments": {
            "technician_name": None,
            "start_time": "2026-07-27 14:00",
            "duration_minutes": 60,
            "gender_preference": None,
            "preferences": [],
        },
    }
    record["dpo_targets"] = targets
    return record


def _confirm() -> dict:
    record = _final()
    record.update(id="confirm", tags=["确认"])
    record["expected"].update(
        duration_minutes=60,
        missing_info=[],
        info_complete=True,
        confirmation=True,
        reply_type="booking_authorized",
        reply="已确认预约。",
    )
    record["dpo_targets"] = ["P5"]
    return record


def _p4() -> dict:
    record = _final()
    record.update(id="p4", conversation_kind="multi_turn", tags=["最终 JSON"])
    record["input"].pop("user_input")
    record["input"]["history"] = [
        {"role": "user", "content": "明天下午做一小时"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "find_technicians",
                        "arguments": json.dumps(
                            {
                                "technician_name": None,
                                "start_time": "2026-07-27 14:00",
                                "duration_minutes": 60,
                                "gender_preference": None,
                                "preferences": [],
                            }
                        ),
                    },
                }
            ],
        },
        {
            "role": "tool",
            "tool_call_id": "call-1",
            "content": json.dumps({"technicians": [{"name": "王芳"}]}),
        },
    ]
    record["expected"].update(
        duration_minutes=60,
        missing_info=[],
        info_complete=True,
        technician_name="王芳",
        technician_status="available",
        reply_type="confirm_available",
        reply="王芳可以预约。",
    )
    record["dpo_targets"] = ["P4"]
    return record


@pytest.mark.parametrize(
    ("target", "record"),
    [
        ("P4", _p4()),
        ("P6", _tool(["P6"])),
        ("P5", _confirm()),
        ("P7", _final()),
        ("P2P3", _tool(["P2P3"])),
    ],
)
def test_perturbation_is_valid_and_different(target: str, record: dict) -> None:
    pair = perturb(raw_sample_from_record(deepcopy(record)), target)
    chosen = json.loads(pair["chosen"]["value"])
    rejected = json.loads(pair["rejected"]["value"])
    (validate_final_output if chosen["action"] == "final" else validate_tool_call_output)(chosen)
    (validate_final_output if rejected["action"] == "final" else validate_tool_call_output)(
        rejected
    )
    assert chosen != rejected


def test_p4_uses_name_outside_tool_result() -> None:
    pair = perturb(raw_sample_from_record(_p4()), "P4")
    assert json.loads(pair["rejected"]["value"])["technician_name"] != "王芳"


def test_p7_fills_default_time_and_duration() -> None:
    pair = perturb(raw_sample_from_record(_final()), "P7")
    arguments = json.loads(pair["rejected"]["value"])["arguments"]
    assert arguments["duration_minutes"] == 60
    assert arguments["start_time"].endswith("14:00")
