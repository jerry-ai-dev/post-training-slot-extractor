import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi.testclient import TestClient

from slot_extractor.quantization.registry import ModelRegistry
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.tool_loop.app import create_app
from slot_extractor.tool_loop.fixture_store import FixtureStore


class Backend:
    model = "fake"

    def generate(self, messages, params=None):
        final = {
            "action": "final",
            "gender_preference": None,
            "technician_gender": None,
            "start_time": None,
            "duration_minutes": None,
            "preferences": [],
            "technician_name": None,
            "technician_status": "not_checked",
            "confirmation": False,
            "info_complete": False,
            "unrelated": False,
            "missing_info": ["start_time", "duration_minutes"],
            "reply_type": "ask_start_time_and_duration",
            "reply": "请提供预约时间和时长。",
        }
        return GenerationResult(
            json.dumps(final, ensure_ascii=False), self.model, 0, 0, 1, 1, 1, {}
        )


def client():
    return TestClient(
        create_app(
            FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml")),
            ModelRegistry.from_config(Path("configs/quantization/phase05.yaml")),
            lambda spec: Backend(),
        )
    )


def test_models_technicians_and_compare_endpoints():
    api = client()
    assert len(api.get("/api/models").json()) == 10
    assert {item["name"] for item in api.get("/api/technicians").json()["technicians"]} == {
        "王芳",
        "李明",
    }
    response = api.post(
        "/api/compare",
        json={
            "left_model_id": "qwen3-0.6b-base-q4-k-m",
            "right_model_id": "qwen3-0.6b-sft-q4-k-m",
            "mode": "sequential",
            "user_input": "预约",
            "left_history": [],
            "right_history": [],
        },
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert response.headers["content-type"].startswith("application/x-ndjson")
    assert {event["side"] for event in events} == {"left", "right"}


def test_selected_models_are_loaded_once_and_reused_across_turns():
    creations = []

    def factory(spec):
        creations.append(spec.model_id)
        return Backend()

    api = TestClient(
        create_app(
            FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml")),
            ModelRegistry.from_config(Path("configs/quantization/phase05.yaml")),
            factory,
        )
    )
    left = "qwen3-0.6b-base-q4-k-m"
    right = "qwen3-0.6b-sft-q4-k-m"
    assert api.post("/api/model-slots/left/load", json={"model_id": left}).json() == {
        "side": "left", "model_id": left, "status": "ready"
    }
    assert api.post("/api/model-slots/right/load", json={"model_id": right}).status_code == 200

    payload = {
        "left_model_id": left,
        "right_model_id": right,
        "mode": "sequential",
        "user_input": "你好",
        "left_history": [],
        "right_history": [],
    }
    assert api.post("/api/compare", json=payload).status_code == 200
    payload["user_input"] = "明天下午两点"
    assert api.post("/api/compare", json=payload).status_code == 200
    assert creations == [left, right]


def test_app_writes_structured_local_diagnostic_log(tmp_path):
    log_path = tmp_path / "app.jsonl"
    app = create_app(
        FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml")),
        ModelRegistry.from_config(Path("configs/quantization/phase05.yaml")),
        lambda spec: Backend(),
        log_path=log_path,
    )
    left = "qwen3-0.6b-base-q4-k-m"
    right = "qwen3-0.6b-sft-q4-k-m"
    with TestClient(app) as api:
        api.post("/api/model-slots/left/load", json={"model_id": left})
        api.post("/api/model-slots/right/load", json={"model_id": right})
        response = api.post(
            "/api/compare",
            json={
                "left_model_id": left,
                "right_model_id": right,
                "mode": "sequential",
                "user_input": "明天下午两点",
                "left_history": [{"role": "user", "content": "你好"}],
                "right_history": [],
            },
        )
        assert response.status_code == 200
        assert api.post(
            "/api/client-logs",
            json={"level": "error", "message": "render failed", "context": {"side": "left"}},
        ).status_code == 204

    records = [json.loads(line) for line in log_path.read_text(encoding="utf-8").splitlines()]
    event_names = [record["event"] for record in records]
    assert "app_started" in event_names
    assert event_names.count("model_ready") == 2
    assert "comparison_started" in event_names
    assert event_names.count("side_completed") == 2
    assert "client_error" in event_names
    comparison = next(record for record in records if record["event"] == "comparison_started")
    assert comparison["user_input"] == "明天下午两点"
    assert comparison["left_history"] == [{"role": "user", "content": "你好"}]
    assert all("timestamp" in record for record in records)


def test_compare_streams_each_side_lifecycle_with_generation_only_timing():
    class TimedBackend(Backend):
        def generate(self, messages, params=None):
            time.sleep(0.02)
            return super().generate(messages, params)

    api = TestClient(
        create_app(
            FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml")),
            ModelRegistry.from_config(Path("configs/quantization/phase05.yaml")),
            lambda spec: (time.sleep(0.06), TimedBackend())[1],
        )
    )
    response = api.post(
        "/api/compare",
        json={
            "left_model_id": "qwen3-0.6b-base-q4-k-m",
            "right_model_id": "qwen3-0.6b-sft-q4-k-m",
            "mode": "sequential",
            "user_input": "预约",
            "left_history": [],
            "right_history": [],
        },
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    statuses = [
        (event["side"], event["payload"]["status"], event["payload"])
        for event in events
        if event["type"] == "side_status"
    ]
    assert [(side, status) for side, status, _ in statuses] == [
        ("left", "inferencing"),
        ("left", "complete"),
        ("right", "inferencing"),
        ("right", "complete"),
    ]
    for _, status, payload in statuses:
        if status == "complete":
            assert 15 <= payload["inference_duration_ms"] < 60


def test_index_contains_two_columns_and_export_controls():
    html = client().get("/").text
    for element_id in (
        "left-model",
        "right-model",
        "left-conversation",
        "right-conversation",
        "comparison-mode",
        "technician-library",
        "export-ndjson",
        "export-json",
        "parallel-warning",
        "run-status",
    ):
        assert f'id="{element_id}"' in html


def test_browser_streams_comparison_and_exposes_running_and_error_states():
    javascript = client().get("/static/app.js").text
    assert "response.body.getReader()" in javascript
    assert "run.disabled=true" in javascript
    assert "updateRunAvailability()" in javascript
    assert "run-status" in javascript
    assert "response.ok" in javascript
    assert "fallbackReply" not in javascript
    assert "该请求不属于预约任务，已移交主管机器人处理。" in javascript
    assert "reply_type==='handoff'" in javascript


def test_browser_renders_independent_side_progress():
    html = client().get("/").text
    javascript = client().get("/static/app.js").text
    assert 'id="left-status"' in html
    assert 'id="right-status"' in html
    assert "side_status" in javascript
    assert "inference_duration_ms" in javascript
    for label in ("等待推理", "模型加载中", "推理中", "推理完成"):
        assert label in javascript


def test_browser_loads_each_model_on_selection_before_enabling_compare():
    javascript = client().get("/static/app.js").text
    assert "/api/model-slots/" in javascript
    assert "/api/client-logs" in javascript
    assert "reportClientError" in javascript
    assert "change" in javascript
    assert "status==='ready'" in javascript


def test_browser_preserves_independent_multi_turn_conversations_until_reset():
    html = client().get("/").text
    javascript = client().get("/static/app.js").text
    assert 'id="clear-conversations"' in html
    assert "appendUserTurn" in javascript
    assert "commitReply" in javascript
    assert "histories[e.side].push" in javascript
    assert "submittedInput" in javascript
    assert "histories.left.length=0" in javascript
    assert "histories.right.length=0" in javascript
    assert "您好，请问您想预约什么时间和项目？" not in javascript
    assert "if(reply)histories[e.side].push" not in javascript


def test_expanded_trace_cannot_break_comparison_grid():
    css = client().get("/static/styles.css").text
    for rule in (
        "grid-template-columns:minmax(0,1fr) minmax(0,1fr)",
        ".grid article{min-width:0}",
        ".event{min-width:0;max-width:100%",
        "pre{max-width:100%;white-space:pre-wrap;overflow-wrap:anywhere;overflow-x:auto}",
    ):
        assert rule in css


def test_comparisons_are_serialized_to_protect_shared_llama_server_port():
    state = {"active": 0, "maximum": 0}
    state_lock = threading.Lock()

    class SlowBackend(Backend):
        def generate(self, messages, params=None):
            with state_lock:
                state["active"] += 1
                state["maximum"] = max(state["maximum"], state["active"])
            time.sleep(0.05)
            try:
                return super().generate(messages, params)
            finally:
                with state_lock:
                    state["active"] -= 1

    app = create_app(
        FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml")),
        ModelRegistry.from_config(Path("configs/quantization/phase05.yaml")),
        lambda spec: SlowBackend(),
    )
    payload = {
        "left_model_id": "qwen3-0.6b-base-q4-k-m",
        "right_model_id": "qwen3-0.6b-sft-q4-k-m",
        "mode": "sequential",
        "user_input": "你好",
        "left_history": [],
        "right_history": [],
    }

    def request():
        return TestClient(app).post("/api/compare", json=payload).status_code

    with ThreadPoolExecutor(max_workers=2) as pool:
        assert list(pool.map(lambda _: request(), range(2))) == [200, 200]
    assert state["maximum"] == 1
