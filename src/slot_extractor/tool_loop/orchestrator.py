import json
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
from .models import ToolLoopEvent, ToolQuery


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
    ):
        self.backend = backend
        self.executor = executor
        self.max_turns = max_turns
        self.now_provider = now_provider or (
            lambda: datetime.now(timezone(timedelta(hours=8)))
        )

    def run(
        self, user_input: str, history: list[dict[str, Any]] | None = None
    ) -> OrchestrationResult:
        current_time = self.now_provider().strftime("%Y-%m-%d %H:%M")
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
                output = parse_model_json(generation.text)
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
                arguments = output["arguments"]
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
                    }
                else:
                    canonical = {
                        "mode": "search",
                        "status": result.status,
                        "requested_technician": None,
                        "candidates": [asdict(candidate) for candidate in result.candidates],
                    }
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
