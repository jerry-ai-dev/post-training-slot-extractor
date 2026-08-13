import json

from slot_extractor.tool_loop.models import CompareEvent, ToolLoopEvent
from slot_extractor.tool_loop.ndjson import encode_event, encode_events


def test_event_is_one_compact_json_line_with_stable_keys():
    event = ToolLoopEvent(2, "reply", {"reply": "已找到"})
    line = encode_event(CompareEvent("left", event, comparable=True))
    assert "\n" not in line
    assert json.loads(line) == {
        "side": "left",
        "comparable": True,
        "seq": 2,
        "type": "reply",
        "payload": {"reply": "已找到"},
    }
    assert list(encode_events([CompareEvent("left", event, True)]))[0].endswith("\n")
