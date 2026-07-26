from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScenarioSpec:
    category: str
    instruction: str


SCENARIOS = {
    "ask_missing_time": ScenarioSpec("追问", "仅缺 start_time，输出 ask_start_time。"),
    "ask_missing_duration": ScenarioSpec("追问", "仅缺 duration_minutes，输出 ask_duration。"),
    "ask_missing_both": ScenarioSpec(
        "追问", "同时缺时间和时长，输出 ask_start_time_and_duration。"
    ),
    "ask_multi_change_time": ScenarioSpec(
        "追问", "多轮场景：用户改口但新时间仍含糊；继承时长，仅追问时间。"
    ),
    "ask_multi_change_duration": ScenarioSpec(
        "追问", "多轮场景：用户改时长但表达含糊；继承时间，仅追问时长。"
    ),
    "tool_general": ScenarioSpec("工具调用", "信息完整、未指定姓名，调用工具做一般查询。"),
    "tool_specific": ScenarioSpec("工具调用", "信息完整且指定技师姓名，调用工具查询该技师。"),
    "tool_replace_technician": ScenarioSpec(
        "工具调用",
        "多轮最小替换：current_state 有旧技师和完整时间时长；用户只要求换成新技师。"
        "arguments 只替换 technician_name，时间、时长、偏好保持不变；旧 technician_gender"
        "不得作为新技师性别条件继承。",
    ),
    "tool_retry_unavailable": ScenarioSpec(
        "工具调用", "多轮：上一位技师 unavailable，用户要求换一位；继承时间时长重新查询。"
    ),
    "tool_retry_no_match": ScenarioSpec(
        "工具调用", "多轮：上次 no_match，用户放宽偏好条件后重新调用工具。"
    ),
    "final_available": ScenarioSpec(
        "最终 JSON", "工具返回 available，输出 confirm_available，姓名必须来自工具结果。"
    ),
    "final_unavailable": ScenarioSpec(
        "最终 JSON",
        "工具返回 unavailable，输出 technician_status=unavailable 和 inform_unavailable。",
    ),
    "final_not_found": ScenarioSpec(
        "最终 JSON",
        "指定技师未找到，输出 technician_status=not_found、姓名 null 和 inform_not_found。",
    ),
    "final_no_match": ScenarioSpec(
        "最终 JSON",
        "没有符合条件的技师，输出 technician_status=no_match、姓名 null 和 inform_no_match。",
    ),
    "final_available_preferences": ScenarioSpec(
        "最终 JSON", "带非空偏好查询成功，输出 available 和 confirm_available。"
    ),
    "confirm_accept": ScenarioSpec(
        "确认", "用户明确接受预约，输出 confirmation=true 和 booking_authorized。"
    ),
    "confirm_reject": ScenarioSpec(
        "确认",
        "用户明确拒绝或说先不了，保持信息完整但 confirmation=false，输出 appointment_paused。",
    ),
    "acknowledge_unavailable": ScenarioSpec(
        "确认",
        "多轮：助手已告知 unavailable，用户说知道了；"
        "输出 confirmation=false 和 acknowledge_result。",
    ),
    "acknowledge_not_found": ScenarioSpec(
        "确认",
        "多轮：助手已告知 not_found，用户表示知悉；输出 confirmation=false 和 acknowledge_result。",
    ),
    "acknowledge_no_match": ScenarioSpec(
        "确认",
        "多轮：助手已告知 no_match，用户表示知悉；输出 confirmation=false 和 acknowledge_result。",
    ),
    "unrelated_product": ScenarioSpec("无关", "用户询问店内设备品牌，按 unrelated/handoff 处理。"),
    "unrelated_price": ScenarioSpec(
        "无关", "用户询问与预约抽槽无关的价格政策，按 unrelated/handoff 处理。"
    ),
    "unrelated_privacy": ScenarioSpec("无关", "用户索要技师私人信息，按 unrelated/handoff 处理。"),
    "unrelated_policy": ScenarioSpec("无关", "用户询问门店管理政策，按 unrelated/handoff 处理。"),
    "unrelated_chitchat": ScenarioSpec(
        "无关", "用户闲聊或提出越界问题，按 unrelated/handoff 处理。"
    ),
}


def scenarios_by_category() -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for name, spec in SCENARIOS.items():
        result.setdefault(spec.category, []).append(name)
    return result


def scenario_dpo_targets(name: str) -> tuple[str, ...]:
    if name.startswith("ask_"):
        return ("P7",)
    if name.startswith("tool_"):
        return ("P6", "P2P3")
    if name in {"final_available", "final_available_preferences"}:
        return ("P4",)
    if name.startswith("final_"):
        return ("P4", "P2P3")
    if name == "confirm_accept":
        return ("P5",)
    if name.startswith("unrelated_"):
        return ("P5", "P6")
    return ()
