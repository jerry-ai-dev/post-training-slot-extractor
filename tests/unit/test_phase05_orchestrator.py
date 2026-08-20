import json
from datetime import datetime
from pathlib import Path

from slot_extractor.schemas.output import validate_final_output
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.tool_loop.find_technicians import FindTechniciansExecutor
from slot_extractor.tool_loop.fixture_store import FixtureStore
from slot_extractor.tool_loop.orchestrator import ConversationOrchestrator


class Backend:
    model = "fake"

    def __init__(self, outputs):
        self.outputs, self.calls, self.messages = list(outputs), 0, []

    def generate(self, messages, params=None):
        self.messages.append(messages)
        text = self.outputs[min(self.calls, len(self.outputs) - 1)]
        self.calls += 1
        return GenerationResult(text, self.model, 0, 0, 0, 1, 0, {})


EXECUTOR = FindTechniciansExecutor(
    FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml"))
)
TOOL = {
    "action": "tool_call",
    "tool_name": "find_technicians",
    "arguments": {
        "technician_name": "王芳",
        "start_time": "2026-08-13 15:00",
        "duration_minutes": 60,
        "gender_preference": "female",
        "preferences": ["肩颈"],
    },
}
FINAL = {
    "action": "final",
    "gender_preference": "female",
    "technician_gender": "female",
    "start_time": "2026-08-13 15:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": False,
    "info_complete": True,
    "unrelated": False,
    "missing_info": [],
    "reply_type": "confirm_available",
    "reply": "王芳可以服务，请确认。",
}


def test_orchestrator_runs_tool_then_final_and_keeps_trace():
    result = ConversationOrchestrator(Backend([json.dumps(TOOL), json.dumps(FINAL)]), EXECUTOR).run(
        "预约"
    )
    assert result.final == FINAL
    assert any(event.kind == "tool_result" and event.payload["trace"] for event in result.events)


def test_orchestrator_stops_after_three_model_turns():
    backend = Backend([json.dumps(TOOL)])
    result = ConversationOrchestrator(backend, EXECUTOR, max_turns=3).run("预约")
    assert result.error == "loop_limit" and backend.calls == 3


def test_orchestrator_returns_valid_final_for_coverage_miss():
    tool = {
        **TOOL,
        "arguments": {
            **TOOL["arguments"],
            "start_time": "2026-08-20 17:00",
        },
    }
    backend = Backend([json.dumps(tool)])

    result = ConversationOrchestrator(backend, EXECUTOR).run("今天下午5点预约王芳")

    assert result.error is None
    assert backend.calls == 1
    assert result.final is not None
    validate_final_output(result.final)
    assert result.final["technician_name"] == "王芳"
    assert result.final["technician_status"] == "no_match"
    assert result.final["missing_info"] == []
    assert "查询超出 Demo 日历范围" in result.final["reply"]
    assert [event.kind for event in result.events] == [
        "start", "model_output", "tool_result", "reply", "complete"
    ]
    tool_event = next(event for event in result.events if event.kind == "tool_result")
    assert tool_event.payload["error_code"] == "unsupported_time"
    assert tool_event.payload["explanation"] == "查询超出 Demo 日历范围"


def test_orchestrator_uses_runtime_shanghai_time_in_system_prompt():
    backend = Backend([json.dumps(FINAL)])
    orchestrator = ConversationOrchestrator(
        backend,
        EXECUTOR,
        now_provider=lambda: datetime(2026, 8, 13, 16, 30),
    )
    orchestrator.run("今天下午两点")
    system = backend.messages[0][0]["content"]
    assert "当前时间：2026-08-13 16:30" in system
    assert "当前时间：2026-08-12 10:00" not in system


def test_orchestrator_corrects_explicit_today_afternoon_time_before_tool_call():
    wrong_tool = {
        **TOOL,
        "arguments": {
            **TOOL["arguments"],
            "start_time": "2026-08-21 15:00",
            "preferences": ["按摩"],
        },
    }
    corrected_final = {
        **FINAL,
        "start_time": "2026-08-20 17:00",
        "preferences": ["按摩"],
    }
    shifted_executor = FindTechniciansExecutor(
        FixtureStore.from_yaml(
            Path("data/fixtures/technicians/phase05-v1.yaml"),
            target_date=datetime(2026, 8, 20).date(),
        )
    )
    backend = Backend([json.dumps(wrong_tool), json.dumps(corrected_final)])

    result = ConversationOrchestrator(
        backend,
        shifted_executor,
        now_provider=lambda: datetime(2026, 8, 20, 10, 0),
    ).run("今天下午5点，请王芳技师来按摩，时间一个小时")

    tool_event = next(event for event in result.events if event.kind == "tool_result")
    assert tool_event.payload["query"]["start_time"] == "2026-08-20 17:00"
    tool_call = backend.messages[1][-2]["tool_calls"][0]["function"]["arguments"]
    assert json.loads(tool_call)["start_time"] == "2026-08-20 17:00"
