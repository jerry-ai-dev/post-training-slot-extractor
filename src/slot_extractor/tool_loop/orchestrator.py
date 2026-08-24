import json
import re
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
from typing import Any

from slot_extractor.inference.base import Backend
from slot_extractor.prompts.rules import (
    FINAL_SCHEMA_HINT,
    SYSTEM_RULES,
    TOOL_SCHEMA_HINT,
    render_tool_descriptions,
)
from slot_extractor.schemas.output import (
    parse_model_json,
    validate_final_output,
    validate_tool_call_output,
)

from .find_technicians import FindTechniciansExecutor
from .models import CanonicalToolResult, ToolLoopEvent, ToolQuery


_RELATIVE_DAY_OFFSETS = {"今天": 0, "明天": 1, "后天": 2}
_CHINESE_HOURS = {
    "一": 1, "二": 2, "两": 2, "三": 3, "四": 4, "五": 5, "六": 6,
    "七": 7, "八": 8, "九": 9, "十": 10, "十一": 11, "十二": 12,
}
_RELATIVE_DAY = re.compile(r"今天|明天|后天")
_CLOCK_TIME = re.compile(
    r"([0-9]{1,2}|十一|十二|十|[一二两三四五六七八九])点"
    r"(?:([0-9]{1,2})分?)?"
)
_DAY_PERIOD = re.compile(r"上午|中午|下午|晚上")


def _normalize_explicit_relative_time(
    user_input: str, start_time: str, now: datetime
) -> str:
    """Prefer an explicit relative date/time in the user's latest message."""
    day_match = _RELATIVE_DAY.search(user_input)
    if day_match is None:
        return start_time
    time_match = _CLOCK_TIME.search(user_input, day_match.end())
    if time_match is None:
        return start_time
    # The period and clock may be separated (for example
    # "今天下午的按摩，时间2点钟"). Keep the latest period between the
    # relative day and the explicit clock instead of requiring adjacency.
    period_matches = list(
        _DAY_PERIOD.finditer(user_input, day_match.end(), time_match.start())
    )
    day_word = day_match.group(0)
    period = period_matches[-1].group(0) if period_matches else None
    hour_text, minute_text = time_match.groups()
    hour = int(hour_text) if hour_text.isdigit() else _CHINESE_HOURS[hour_text]
    if period in {"下午", "晚上"} and hour < 12:
        hour += 12
    elif period == "中午" and hour < 11:
        hour += 12
    elif period == "上午" and hour == 12:
        hour = 0
    minute = int(minute_text or 0)
    if hour > 23 or minute > 59:
        return start_time
    target = (now + timedelta(days=_RELATIVE_DAY_OFFSETS[day_word])).replace(
        hour=hour, minute=minute, second=0, microsecond=0
    )
    return target.strftime("%Y-%m-%d %H:%M")


def _coverage_miss_final(
    query: ToolQuery, result: CanonicalToolResult
) -> dict[str, object]:
    target = f"{query.technician_name}技师" if query.technician_name else "符合条件的技师"
    return {
        "action": "final",
        "gender_preference": query.gender_preference,
        "technician_gender": None,
        "start_time": query.start_time.strftime("%Y-%m-%d %H:%M"),
        "duration_minutes": query.duration_minutes,
        "preferences": list(query.preferences),
        "technician_name": query.technician_name,
        "technician_status": "no_match",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "inform_no_match",
        "reply": f"{result.explanation}，暂时无法确认{target}是否可用，请调整条件后重试。",
    }


def _unique_match_final(
    query: ToolQuery, result: CanonicalToolResult
) -> dict[str, object]:
    """Turn a unique tool match into protocol-safe state deterministically."""
    technician = result.candidates[0]
    start_time = query.start_time.strftime("%Y-%m-%d %H:%M")
    preference_text = "、".join(query.preferences)
    service_text = f"{preference_text}按摩" if preference_text else "按摩"
    return {
        "action": "final",
        "gender_preference": query.gender_preference,
        "technician_gender": technician.gender,
        "start_time": start_time,
        "duration_minutes": query.duration_minutes,
        "preferences": list(query.preferences),
        "technician_name": technician.name,
        "technician_status": "available",
        "confirmation": False,
        "info_complete": True,
        "unrelated": False,
        "missing_info": [],
        "reply_type": "confirm_available",
        "reply": (
            f"{technician.name}技师在{start_time}有空，可以为您安排"
            f"{query.duration_minutes}分钟{service_text}，请问是否确认预约？"
        ),
    }


@dataclass(frozen=True)
class OrchestrationResult:
    events: tuple[ToolLoopEvent, ...]
    final: dict[str, object] | None
    error: str | None


class ConversationOrchestrator:
    def __init__(
        self,
        backend: Backend,
        executor: FindTechniciansExecutor,
        max_turns: int = 3,
        now_provider: Callable[[], datetime] | None = None,
        canonicalize_unique_matches: bool = False,
    ):
        self.backend = backend
        self.executor = executor
        self.max_turns = max_turns
        self.canonicalize_unique_matches = canonicalize_unique_matches
        self.now_provider = now_provider or (
            lambda: datetime.now(timezone(timedelta(hours=8)))
        )

    def run(
        self, user_input: str, history: list[dict[str, Any]] | None = None
    ) -> OrchestrationResult:
        now = self.now_provider()
        current_time = now.strftime("%Y-%m-%d %H:%M")
        system = (
            f"{SYSTEM_RULES}\n{FINAL_SCHEMA_HINT}\n{TOOL_SCHEMA_HINT}\n"
            f"{render_tool_descriptions(['find_technicians'])}\n"
            f"当前时间：{current_time}\n当前状态：null"
        )
        messages = [
            {"role": "system", "content": system},
            *(history or []),
            {"role": "user", "content": user_input},
        ]
        events = [ToolLoopEvent(0, "start", {"user_input": user_input})]
        try:
            for _ in range(self.max_turns):
                generation = self.backend.generate(messages)
                try:
                    output = parse_model_json(generation.text)
                except ValueError as parse_error:
                    # For the comparison/demo APP, preserve the model's exact
                    # response instead of replacing it with only a validator
                    # error. Invalid protocol output remains display-only and
                    # is never executed as a tool call.
                    events.append(
                        ToolLoopEvent(
                            len(events),
                            "model_output",
                            {
                                "raw": generation.text,
                                "parsed": None,
                                "protocol_valid": False,
                                "parse_error": str(parse_error),
                            },
                        )
                    )
                    events.append(
                        ToolLoopEvent(
                            len(events),
                            "reply",
                            {
                                "reply": generation.text,
                                "raw_output": True,
                                "tool_executed": False,
                                "warning": "模型输出不符合协议，以下为原始文本，未执行工具。",
                            },
                        )
                    )
                    events.append(ToolLoopEvent(len(events), "complete", {}))
                    return OrchestrationResult(tuple(events), None, None)
                events.append(
                    ToolLoopEvent(
                        len(events), "model_output", {"raw": generation.text, "parsed": output}
                    )
                )
                if output.get("action") == "final":
                    validate_final_output(output)
                    events.append(
                        ToolLoopEvent(
                            len(events), "reply", {"reply": output["reply"], "final": output}
                        )
                    )
                    events.append(ToolLoopEvent(len(events), "complete", {}))
                    return OrchestrationResult(tuple(events), output, None)
                validate_tool_call_output(output)
                if output["tool_name"] != "find_technicians":
                    raise ValueError(f"unknown tool: {output['tool_name']}")
                arguments = dict(output["arguments"])
                arguments["start_time"] = _normalize_explicit_relative_time(
                    user_input, arguments["start_time"], now
                )
                query = ToolQuery(
                    arguments["technician_name"],
                    datetime.strptime(arguments["start_time"], "%Y-%m-%d %H:%M"),
                    arguments["duration_minutes"],
                    arguments["gender_preference"],
                    tuple(arguments["preferences"]),
                )
                result = self.executor.find(query)
                payload = asdict(result)
                payload["query"]["start_time"] = arguments["start_time"]
                events.append(ToolLoopEvent(len(events), "tool_result", payload))
                if query.technician_name:
                    canonical = {
                        "mode": "specific",
                        "status": result.status,
                        "requested_technician": query.technician_name,
                        "technician": (asdict(result.candidates[0]) if result.candidates else None),
                        "error_code": result.error_code,
                        "explanation": result.explanation,
                    }
                else:
                    canonical = {
                        "mode": "search",
                        "status": result.status,
                        "requested_technician": None,
                        "candidates": [asdict(candidate) for candidate in result.candidates],
                        "error_code": result.error_code,
                        "explanation": result.explanation,
                    }
                if result.status == "mock_coverage_miss":
                    final = _coverage_miss_final(query, result)
                    validate_final_output(final)
                    events.append(ToolLoopEvent(len(events), "reply", {"reply": final["reply"], "final": final}))
                    events.append(ToolLoopEvent(len(events), "complete", {}))
                    return OrchestrationResult(tuple(events), final, None)
                if (
                    self.canonicalize_unique_matches
                    and result.status in {"matched", "available"}
                    and len(result.candidates) == 1
                ):
                    final = _unique_match_final(query, result)
                    validate_final_output(final)
                    events.append(ToolLoopEvent(len(events), "reply", {"reply": final["reply"], "final": final}))
                    events.append(ToolLoopEvent(len(events), "complete", {}))
                    return OrchestrationResult(tuple(events), final, None)
                messages.extend(
                    [
                        {
                            "role": "assistant",
                            "content": None,
                            "tool_calls": [
                                {
                                    "id": f"call-{len(events)}",
                                    "type": "function",
                                    "function": {
                                        "name": "find_technicians",
                                        "arguments": json.dumps(
                                            arguments, ensure_ascii=False
                                        ),
                                    },
                                }
                            ],
                        },
                        {
                            "role": "tool",
                            "name": "find_technicians",
                            "tool_call_id": f"call-{len(events)}",
                            "content": json.dumps(canonical, ensure_ascii=False),
                        },
                    ]
                )
        except Exception as error:
            events.append(ToolLoopEvent(len(events), "error", {"message": str(error)}))
            return OrchestrationResult(tuple(events), None, str(error))
        events.append(ToolLoopEvent(len(events), "error", {"message": "loop_limit"}))
        return OrchestrationResult(tuple(events), None, "loop_limit")
