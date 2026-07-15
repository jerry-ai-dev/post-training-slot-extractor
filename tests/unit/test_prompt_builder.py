from slot_extractor.prompts.rules import TOOL_SPECS
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.sample import Sample


def _sample(**input_updates: object) -> Sample:
    input_obj = {
        "history": [],
        "current_state": None,
        "user_input": "明天下午两点做60分钟按摩",
        "current_time": "2026-06-08 10:00",
        "available_tools": [],
    }
    input_obj.update(input_updates)
    return Sample(
        id="case-1",
        layer="final",
        input=input_obj,
        expected={"action": "final"},
        assertions=[],
        tags=[],
    )


def test_build_messages_has_system_then_natural_history_then_user() -> None:
    sample = _sample(
        history=[
            {"role": "user", "content": "约个精油按摩"},
            {"role": "assistant", "content": "请问您想什么时候过来，按摩多长时间呢？"},
        ],
        current_state={
            "gender": None,
            "start_time": None,
            "duration_minutes": None,
            "preferences": ["精油"],
            "technician_name": None,
            "technician_status": "not_checked",
            "confirmation": False,
            "info_complete": False,
            "unrelated": False,
            "missing_info": ["start_time", "duration_minutes"],
            "last_reply_type": "ask_start_time_and_duration",
        },
        user_input="明天下午两点",
    )

    messages = PromptBuilder().build_messages(sample)

    assert messages[0]["role"] == "system"
    assert "只输出一个 JSON 对象" in messages[0]["content"]
    assert "当前状态：" in messages[0]["content"]
    assert '"preferences":["精油"]' in messages[0]["content"]
    assert "当前工具上下文" not in messages[0]["content"]
    assert "2026-06-08 10:00" in messages[0]["content"]
    assert messages[0]["_sample_id"] == "case-1"
    assert messages[1] == {"role": "user", "content": "约个精油按摩"}
    assert messages[2] == {
        "role": "assistant",
        "content": "请问您想什么时候过来，按摩多长时间呢？",
    }
    assert messages[-1] == {"role": "user", "content": "明天下午两点"}


def test_build_messages_preserves_tool_exchange_without_empty_user_message() -> None:
    sample = _sample(
        history=[
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
                            "arguments": '{"technician_name":"王芳"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"mode":"specific","status":"available"}',
            },
        ],
        current_state={"start_time": "2026-06-09 14:00", "duration_minutes": 60},
        user_input=None,
    )

    messages = PromptBuilder().build_messages(sample)

    assert [message["role"] for message in messages] == [
        "system",
        "user",
        "assistant",
        "tool",
    ]
    assert messages[-2]["tool_calls"][0]["function"]["name"] == "find_technicians"
    assert '"status":"available"' in messages[-1]["content"]


def test_build_messages_renders_tool_contract_only_for_active_tools() -> None:
    with_tools = _sample(available_tools=["find_technicians"])
    without_tools = _sample(available_tools=[])

    tool_system = PromptBuilder().build_messages(with_tools)[0]["content"]
    final_system = PromptBuilder().build_messages(without_tools)[0]["content"]

    assert (
        "find_technicians(technician_name, start_time, duration_minutes, gender, preferences)"
        in tool_system
    )
    assert "tool_call 顶层仅含 action、tool_name、arguments" in tool_system
    assert "tool_call 不得包含 reply_type 或 reply" in tool_system
    assert "find_technicians" not in final_system
    assert "tool_call 顶层" not in final_system


def test_system_prompt_defines_reply_aware_final_contract() -> None:
    system = PromptBuilder().build_messages(_sample())[0]["content"]

    assert "final 必须且只能使用以下 13 个字段" in system
    assert '"reply_type":"ask_start_time"' in system
    assert '"reply":"请问您想什么时候过来呢？"' in system
    assert "handoff 时 reply=null" in system
    assert "未真正创建预约前不得声称预约成功" in system


def test_system_prompt_defines_complete_unrelated_output() -> None:
    system = PromptBuilder().build_messages(_sample())[0]["content"]

    assert '"action":"final"' in system
    assert '"confirmation":false' in system
    assert '"info_complete":false' in system
    assert '"unrelated":true' in system
    assert '"missing_info":[]' in system
    assert '"reply_type":"handoff"' in system
    assert '"reply":null' in system


def test_system_prompt_defines_state_inheritance_and_reply_types() -> None:
    system = PromptBuilder().build_messages(_sample())[0]["content"]

    assert "current_state" in system
    assert "最新明确修改覆盖旧值" in system
    assert "未修改字段继承 current_state" in system
    assert "appointment_paused" in system
    assert "confirm_available" in system
    assert "booking_authorized" in system
    assert "相对时间结合当前时间换算" in system
    assert "明天=当前日期+1天" not in system
    assert "更换指定技师时" in system
    assert "只表示该字段暂未确定，不等于暂停预约" in system


def test_system_prompt_keeps_tool_result_evidence_boundaries() -> None:
    system = PromptBuilder().build_messages(
        _sample(available_tools=["find_technicians"])
    )[0]["content"]

    assert "available/unavailable/not_found/no_match 只能来自最新 tool 消息" in system
    assert "更改查询条件后旧工具结果失效" in system
    assert "指定失败不选替代" in system


def test_system_distinguishes_gender_filter_from_tool_derived_gender() -> None:
    system = PromptBuilder().build_messages(_sample())[0]["content"]

    assert "tool_call.arguments.gender 只表示用户当前明确要求" in system
    assert "不得自动变成下一次查询的 gender 条件" in system
    assert "Final 必须复制其 name 和 gender" in system


def test_system_prompt_remains_compact() -> None:
    system = PromptBuilder().build_messages(
        _sample(available_tools=["find_technicians"])
    )[0]["content"]

    assert len(system) < 4200


def test_only_technician_lookup_tool_is_registered() -> None:
    assert set(TOOL_SPECS) == {"find_technicians"}
