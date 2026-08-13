import json
from datetime import datetime
from pathlib import Path

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
