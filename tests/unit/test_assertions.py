# tests/unit/test_assertions.py
from slot_extractor.evaluation.assertions import (
    evaluate_assertion,
    hallucinated_entities,
    is_tool_call,
    within_business_hours,
)
from slot_extractor.schemas.sample import Sample


def _sample(expected=None, input_obj=None) -> Sample:
    return Sample(
        id="c1",
        layer="final",
        input=input_obj or {},
        expected=expected or {},
        assertions=[],
        tags=[],
    )


def test_field_equality_assertion_passes() -> None:
    out = {"start_time": "2026-06-09 14:00"}
    r = evaluate_assertion("start_time == 2026-06-09 14:00", out, _sample())
    assert r.passed is True
    assert r.dimension == "field_extraction"


def test_args_equality_assertion_detects_mismatch() -> None:
    out = {"arguments": {"technician_name": "李明"}}
    r = evaluate_assertion("args.technician_name == 小王", out, _sample())
    assert r.passed is False
    assert r.dimension == "field_extraction"


def test_tool_name_assertion_maps_to_tool_call_dimension() -> None:
    out = {"tool_name": "find_technicians"}
    r = evaluate_assertion("tool_name == find_technicians", out, _sample())
    assert r.passed is True
    assert r.dimension == "tool_call"


def test_named_predicate_no_field_outside_schema() -> None:
    good = {
        "action": "final",
        "gender": None,
        "start_time": None,
        "duration_minutes": None,
        "preferences": [],
        "technician_name": None,
        "technician_status": "not_checked",
        "confirmation": False,
        "info_complete": False,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "ask_start_time_and_duration",
        "reply": "请问您想什么时候过来，按摩多长时间呢？",
    }
    r = evaluate_assertion("no_field_outside_schema", good, _sample())
    assert r.passed is True
    assert r.dimension == "instruction"


def test_no_hallucinated_entity_predicate() -> None:
    out = {"action": "final", "technician_name": "张三", "technician_status": "not_checked"}
    s = _sample(input_obj={"user_input": "我想找李明"})
    r = evaluate_assertion("no_hallucinated_entity", out, s)
    assert r.passed is False
    assert r.dimension == "hallucination"


def test_malformed_assertion_marked_not_passed() -> None:
    r = evaluate_assertion("this is not valid", {}, _sample())
    assert r.passed is False
    assert "malformed" in r.detail


def test_hallucinated_entities_helper() -> None:
    out = {"action": "final", "technician_name": "张三", "technician_status": "not_checked"}
    sample = _sample(input_obj={"user_input": "我想找李明"})
    assert hallucinated_entities(out, sample) == ["张三"]


def test_hallucinated_entities_uses_tool_history() -> None:
    out = {"action": "final", "technician_name": "王芳", "technician_status": "available"}
    sample = _sample(
        input_obj={
            "history": [
                {
                    "role": "tool",
                    "tool_call_id": "call-1",
                    "content": '{"mode":"specific","status":"available","technician":{"name":"王芳"}}',
                }
            ]
        }
    )
    assert hallucinated_entities(out, sample) == []


def test_hallucinated_entities_uses_pending_current_state_on_confirmation() -> None:
    out = {"action": "final", "technician_name": "李明", "technician_status": "unavailable"}
    sample = _sample(
        input_obj={
            "current_state": {
                "technician_name": "李明",
                "technician_status": "unavailable",
            },
            "user_input": "知道了",
        }
    )

    assert hallucinated_entities(out, sample) == []


def test_is_tool_call_helper() -> None:
    assert is_tool_call({"action": "tool_call"}) is True
    assert is_tool_call({"action": "final"}) is False


def test_within_business_hours_helper() -> None:
    assert within_business_hours("2026-06-09 14:00") is True
    assert within_business_hours("2026-06-09 07:00") is False
    assert within_business_hours(None) is False
