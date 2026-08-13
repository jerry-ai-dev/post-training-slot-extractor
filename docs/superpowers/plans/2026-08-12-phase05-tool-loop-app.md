# Phase 05 Tool-Loop App Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个可复现的本地双模型预约工具循环对比应用：左右模型各自最多执行三轮 `find_technicians`，由版本化 YAML Mock 技师库提供确定性事实，FastAPI 通过事件级 NDJSON 驱动原生双栏 UI，并展示回复、工具结果、逐技师 trace、匹配解释与导出内容。

**Architecture:** 每侧 `ConversationOrchestrator` 负责 PromptBuilder、所选 Backend 与工具执行器之间的最多三轮状态机；模型选择消费量化计划的 canonical registry，顺序模式依次运行两侧完整循环，并行模式隔离两个服务且标记性能不可比；UI 与两侧 executor 都从同一份固定日期 YAML fixture 加载不可变技师库。工具层以规范化 canonical result 和 per-technician trace 返回确定性结果，应用层只追加事件，不改既有 evaluator；FastAPI 将带 `side`/`comparable` 的每个事件作为一行 NDJSON 发出，浏览器以 `fetch` 读取并增量渲染，而不是 token streaming。

**Tech Stack:** Python 3.12、现有 Backend/PromptBuilder/Sample/output validators、PyYAML、FastAPI、Uvicorn、原生 HTML/CSS/JavaScript、pytest、ruff。

## Global Constraints

- 技师库必须是版本化 YAML 固定日期 fixture，不得改成 Python 常量；UI 展示和 executor 查询必须加载同一文件。
- fixture 至少包含王芳、李明的姓名、性别、专长和 availability；availability 使用半开区间 `[start, end)`。
- `find_technicians(technician_name, start_time, duration_minutes, gender_preference, preferences)` 五个参数必须全部出现且顺序/含义固定。
- 指定技师结果只允许 `available`、`unavailable`、`not_found`；条件搜索只允许 `matched`、`no_match`。
- 多候选命中必须返回 fixture ambiguity error；未建模偏好或越界输入必须返回 `mock_coverage_miss`，不得猜测事实。
- canonical tool result 必须稳定序列化，并包含每位参与技师的逐技师 trace 与匹配解释。
- `ConversationOrchestrator` 最多执行 3 次模型/工具循环，超限必须产生明确的 `loop_limit` 事件/错误。
- 复用现有 `PromptBuilder`、`Sample`、output validators；不得修改 `src/slot_extractor/evaluation/` evaluator 或冻结 `data/eval/`。
- 事件传输使用事件级 NDJSON；不得实现 token streaming。主显示为 `reply`，JSON/tool 内容放在淡色 details。
- UI 使用 FastAPI + 原生双模型对比 HTML/CSS/JavaScript；左右各可从 canonical registry 选择模型并保持独立多轮历史，且必须有技师库面板、匹配解释、顺序公平/并行体验和导出功能。
- 依赖 `fastapi` 与 `uvicorn` 必须加入 `pyproject.toml`，不得引入前端构建工具。
- 每个任务先写真实失败测试，再实现最小代码；每个任务末尾单独提交。

---

## File Map

- `pyproject.toml`: 添加 FastAPI/Uvicorn runtime dependencies。
- `configs/tool_loop/phase05.yaml`: 运行配置，只保存 fixture 路径、canonical registry 路径、默认左右模型 ID、最大循环数和事件版本；不复制模型元数据。
- `data/fixtures/technicians/phase05-v1.yaml`: 版本化固定日期技师库；唯一事实来源。
- `src/slot_extractor/tool_loop/models.py`: 工具参数、技师记录、canonical result、trace 和事件的数据类型。
- `src/slot_extractor/tool_loop/fixture_store.py`: YAML 加载、schema 校验、固定排序及只读查询。
- `src/slot_extractor/tool_loop/find_technicians.py`: 五参确定性工具执行器、状态判定和 coverage/ambiguity 错误。
- `src/slot_extractor/tool_loop/orchestrator.py`: 单请求最多三轮的模型-工具状态机。
- `src/slot_extractor/tool_loop/ndjson.py`: 事件 envelope 和 NDJSON 编码。
- `src/slot_extractor/tool_loop/app.py`: FastAPI app、模型注册表注入、双侧执行模式、API 路由、统一 fixture 注入和静态 UI 响应。
- `src/slot_extractor/tool_loop/static/index.html`: 原生双模型对比 UI 结构。
- `src/slot_extractor/tool_loop/static/app.js`: NDJSON 读取、reply/details/trace 渲染、导出。
- `src/slot_extractor/tool_loop/static/styles.css`: 双栏布局、淡色 details、响应式布局。
- `tests/unit/test_phase05_fixture_store.py`: fixture 加载、版本、共享实例与 availability 边界测试。
- `tests/unit/test_phase05_find_technicians.py`: 五参工具所有确定性状态和错误测试。
- `tests/unit/test_phase05_orchestrator.py`: 三轮上限、工具结果回填、canonical trace 和复用验证器测试。
- `tests/unit/test_phase05_ndjson.py`: 事件 envelope/编码测试。
- `tests/integration/test_phase05_app.py`: FastAPI API、NDJSON 顺序、fixture 面板和导出测试。

---

### Task 1: Fixture schema, dependency and typed contracts

**Files:**
- Create: `data/fixtures/technicians/phase05-v1.yaml`
- Create: `configs/tool_loop/phase05.yaml`
- Create: `src/slot_extractor/tool_loop/__init__.py`
- Create: `src/slot_extractor/tool_loop/models.py`
- Create: `tests/unit/test_phase05_fixture_store.py`
- Modify: `pyproject.toml:6-11`

**Interfaces:**
- Produces `Technician(name: str, gender: Literal["female", "male"], specialties: tuple[str, ...], availability: tuple[AvailabilityWindow, ...])`.
- Produces `AvailabilityWindow(start: datetime, end: datetime)` with invariant `start < end` and half-open containment.
- Produces `ToolQuery(technician_name: str | None, start_time: datetime, duration_minutes: int, gender_preference: Literal["female", "male"] | None, preferences: tuple[str, ...])`.
- Produces `FixtureStore.from_yaml(path: Path) -> FixtureStore`, `FixtureStore.technicians() -> tuple[Technician, ...]`, and `FixtureStore.version -> str`.

- [ ] **Step 1: Write the failing tests and fixture contract.**

```python
from datetime import datetime
from pathlib import Path

from slot_extractor.tool_loop.fixture_store import FixtureStore

FIXTURE = Path("data/fixtures/technicians/phase05-v1.yaml")

def test_fixture_is_versioned_and_contains_named_technicians():
    store = FixtureStore.from_yaml(FIXTURE)
    assert store.version == "phase05-v1"
    assert {tech.name for tech in store.technicians()} >= {"王芳", "李明"}
    assert all(tech.specialties for tech in store.technicians())

def test_availability_is_half_open():
    store = FixtureStore.from_yaml(FIXTURE)
    wang = next(tech for tech in store.technicians() if tech.name == "王芳")
    window = wang.availability[0]
    assert window.contains(window.start)
    assert not window.contains(window.end)
```

Fixture must contain literal fixed date `2026-08-12` windows and the exact dataset-contract specialties, for example:

```yaml
version: phase05-v1
date: 2026-08-12
technicians:
  - name: 王芳
    gender: female
    specialties: [肩颈, 精油, 足部]
    availability:
      - start: "2026-08-12 09:00"
        end: "2026-08-12 14:30"
  - name: 李明
    gender: male
    specialties: [泰式, 足部, 精油]
    availability:
      - start: "2026-08-12 13:00"
        end: "2026-08-12 17:00"
```

- [ ] **Step 2: Run the focused test to verify it fails.**

Run: `uv run pytest tests/unit/test_phase05_fixture_store.py -v`

Expected: FAIL with `ModuleNotFoundError` for `slot_extractor.tool_loop.fixture_store`.

- [ ] **Step 3: Add dependencies and implement typed contracts plus YAML loading.**

Add to `pyproject.toml` dependencies:

```toml
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
```

Implement `models.py` with frozen dataclasses and `fixture_store.py` with `yaml.safe_load`, exact required keys, ISO minute parsing, duplicate-name rejection, deterministic name sorting, and defensive tuple returns. Reject malformed YAML with `ValueError` naming the field.

- [ ] **Step 4: Run the focused tests and lint.**

Run: `uv run pytest tests/unit/test_phase05_fixture_store.py -v && uv run ruff check src/slot_extractor/tool_loop tests/unit/test_phase05_fixture_store.py`

Expected: `2 passed`; ruff exits `0`.

- [ ] **Step 5: Commit.**

```bash
git add pyproject.toml data/fixtures/technicians/phase05-v1.yaml configs/tool_loop/phase05.yaml src/slot_extractor/tool_loop tests/unit/test_phase05_fixture_store.py
git commit -m "feat: add phase05 fixture and tool-loop contracts"
```

---

### Task 2: Deterministic `find_technicians` executor

**Files:**
- Create: `src/slot_extractor/tool_loop/find_technicians.py`
- Create: `tests/unit/test_phase05_find_technicians.py`
- Modify: `src/slot_extractor/tool_loop/models.py`

**Interfaces:**
- Produces `CanonicalToolResult(status: str, query: ToolQuery, candidates: tuple[TechnicianMatch, ...], trace: tuple[TechnicianTrace, ...], explanation: str, error_code: str | None)`.
- Produces `TechnicianTrace(name: str, considered: bool, matched: bool, reasons: tuple[str, ...])`.
- Produces `FindTechniciansExecutor(store: FixtureStore).find(query: ToolQuery) -> CanonicalToolResult`.

- [ ] **Step 1: Write failing tests for every required result branch.**

```python
from datetime import datetime
from pathlib import Path
import pytest
from slot_extractor.tool_loop.find_technicians import FindTechniciansExecutor
from slot_extractor.tool_loop.models import ToolQuery
from slot_extractor.tool_loop.fixture_store import FixtureStore

EXECUTOR = FindTechniciansExecutor(FixtureStore.from_yaml(Path("data/fixtures/technicians/phase05-v1.yaml")))
def query(**overrides):
    values = dict(technician_name=None, start_time=datetime(2026, 8, 12, 14), duration_minutes=60,
                  gender_preference=None, preferences=("精油",))
    return ToolQuery(**(values | overrides))

def test_specific_available_and_boundary_is_available():
    result = EXECUTOR.find(query(technician_name="李明"))
    assert result.status == "available"
    assert result.error_code is None
    assert result.trace and result.explanation

def test_specific_unavailable_and_not_found():
    assert EXECUTOR.find(query(technician_name="王芳")).status == "unavailable"
    assert EXECUTOR.find(query(technician_name="不存在")).status == "not_found"

def test_search_no_match_and_ambiguity():
    no_match = EXECUTOR.find(query(start_time=datetime(2026, 8, 12, 18)))
    assert no_match.status == "no_match"
    ambiguous = EXECUTOR.find(query(start_time=datetime(2026, 8, 12, 13), preferences=("精油",)))
    assert ambiguous.error_code == "fixture_ambiguity"

def test_unmodeled_preference_and_out_of_fixture_range_are_coverage_misses():
    assert EXECUTOR.find(query(preferences=("芳疗",))).error_code == "mock_coverage_miss"
    assert EXECUTOR.find(query(start_time=datetime(2027, 1, 1, 9))).error_code == "mock_coverage_miss"
```

- [ ] **Step 2: Run tests to verify failure.**

Run: `uv run pytest tests/unit/test_phase05_find_technicians.py -v`

Expected: FAIL because executor and canonical result are not defined.

- [ ] **Step 3: Implement the minimal deterministic executor.**

Validate all five fields before querying. A requested interval is `[start_time, start_time + duration]`; it is contained only when `window.start <= requested_start` and `requested_end <= window.end`. Specific mode checks only the named technician. Search mode filters gender and exact specialty membership, preserves fixture ordering, returns one match only, returns `fixture_ambiguity` for multiple matches, and returns `no_match` for zero matches. Unsupported preferences/date range return `mock_coverage_miss`. Every technician gets a stable trace reason such as `name_mismatch`, `outside_availability`, `gender_mismatch`, `specialty_mismatch`, or `matched`.

- [ ] **Step 4: Run focused tests and lint.**

Run: `uv run pytest tests/unit/test_phase05_find_technicians.py -v && uv run ruff check src/slot_extractor/tool_loop/find_technicians.py tests/unit/test_phase05_find_technicians.py`

Expected: all branch tests PASS; ruff exits `0`.

- [ ] **Step 5: Commit.**

```bash
git add src/slot_extractor/tool_loop/models.py src/slot_extractor/tool_loop/find_technicians.py tests/unit/test_phase05_find_technicians.py
git commit -m "feat: add deterministic phase05 technician tool"
```

---

### Task 3: Conversation orchestrator and validator reuse

**Files:**
- Create: `src/slot_extractor/tool_loop/orchestrator.py`
- Create: `tests/unit/test_phase05_orchestrator.py`
- Modify: `src/slot_extractor/tool_loop/models.py`

**Interfaces:**
- Consumes existing `Backend.generate(messages, params)`, `PromptBuilder.build_messages(sample)`, `parse_model_json`, `validate_tool_call_output`, and `validate_final_output` without changing their files.
- Produces `ConversationOrchestrator(backend: Backend, executor: FindTechniciansExecutor, prompt_builder: PromptBuilder | None = None, max_turns: int = 3)`.
- Produces `ConversationOrchestrator.run(sample: Sample) -> OrchestrationResult`.
- `OrchestrationResult` contains `events: tuple[ToolLoopEvent, ...]`, `final: dict[str, object] | None`, and `error: str | None`.

- [ ] **Step 1: Write failing tests for one loop, tool-result continuation, and the three-turn cap.**

```python
from slot_extractor.schemas.output import validate_final_output

def test_orchestrator_reuses_prompt_builder_and_returns_canonical_tool_trace(fake_backend, sample):
    result = ConversationOrchestrator(fake_backend, executor).run(sample)
    assert result.final["action"] == "final"
    assert any(event.kind == "tool_result" for event in result.events)
    assert result.events[-1].payload["trace"]

def test_orchestrator_stops_after_three_model_turns(looping_backend, sample):
    result = ConversationOrchestrator(looping_backend, executor, max_turns=3).run(sample)
    assert result.error == "loop_limit"
    assert looping_backend.calls == 3

def test_final_output_is_checked_by_existing_validator(final_backend, sample):
    result = ConversationOrchestrator(final_backend, executor).run(sample)
    validate_final_output(result.final)
```

- [ ] **Step 2: Run focused tests to verify failure.**

Run: `uv run pytest tests/unit/test_phase05_orchestrator.py -v`

Expected: FAIL because `ConversationOrchestrator` and event types are not defined.

- [ ] **Step 3: Implement the bounded state machine.**

For each turn, construct the current `Sample` through `PromptBuilder`, call the backend once, parse raw JSON, validate either tool-call or final output, and emit `model_output`. For a tool call, reject any tool other than `find_technicians`, convert the exact five arguments into `ToolQuery`, call the executor, then append the existing history contract exactly: an assistant message with `content: null` and one OpenAI-style `tool_calls` entry whose `function.arguments` is a JSON object string, followed by a tool message with the matching `tool_call_id` and canonical tool-result JSON string. Rebuild the next `Sample` from this cloned history and continue. For final output, emit `reply` and stop. Preserve `tool_result` payload fields `status`, `candidates`, `trace`, `explanation`, and `error_code`; never expose non-canonical fixture internals. On parse/validation/argument errors emit `error` with stable codes. When `turn == max_turns`, return `loop_limit` without a fourth backend call. Use an explicit cloned sample/history adapter rather than modifying evaluator code.

- [ ] **Step 4: Run tests and lint.**

Run: `uv run pytest tests/unit/test_phase05_orchestrator.py -v && uv run ruff check src/slot_extractor/tool_loop/orchestrator.py tests/unit/test_phase05_orchestrator.py`

Expected: all tests PASS; ruff exits `0`.

- [ ] **Step 5: Commit.**

```bash
git add src/slot_extractor/tool_loop/orchestrator.py src/slot_extractor/tool_loop/models.py tests/unit/test_phase05_orchestrator.py
git commit -m "feat: add bounded conversation orchestrator"
```

---

### Task 4: Event-level NDJSON contract

**Files:**
- Create: `src/slot_extractor/tool_loop/ndjson.py`
- Create: `tests/unit/test_phase05_ndjson.py`
- Modify: `src/slot_extractor/tool_loop/models.py`

**Interfaces:**
- Produces `ToolLoopEvent(seq: int, kind: Literal["start", "model_output", "tool_result", "reply", "error", "complete"], payload: dict[str, object])`.
- Produces `CompareEvent(side: Literal["left", "right"], event: ToolLoopEvent, comparable: bool)`; serialization flattens `event.seq/kind/payload` beside `side` and `comparable`.
- Produces `encode_event(event: CompareEvent) -> str` and `encode_events(events: Iterable[CompareEvent]) -> Iterator[str>`.

- [ ] **Step 1: Write failing tests.**

```python
import json
from slot_extractor.tool_loop.ndjson import encode_event

def test_event_is_one_compact_json_line_with_stable_keys():
    event = ToolLoopEvent(2, "reply", {"reply": "已找到"})
    line = encode_event(CompareEvent("left", event, comparable=True))
    assert "\n" not in line
    assert json.loads(line) == {
        "side": "left", "comparable": True,
        "seq": 2, "type": "reply", "payload": {"reply": "已找到"},
    }
```

- [ ] **Step 2: Run test to verify failure.**

Run: `uv run pytest tests/unit/test_phase05_ndjson.py -v`

Expected: FAIL because the event encoder is missing.

- [ ] **Step 3: Implement compact UTF-8-safe encoding.**

Use `json.dumps(..., ensure_ascii=False, separators=(",", ":"))`; preserve side, comparable flag and per-side event sequence numbers. `encode_events` must yield one line per event and a trailing newline for HTTP streaming; payload strings may contain escaped newlines, but each encoded envelope remains exactly one physical line.

- [ ] **Step 4: Run tests and lint.**

Run: `uv run pytest tests/unit/test_phase05_ndjson.py -v && uv run ruff check src/slot_extractor/tool_loop/ndjson.py tests/unit/test_phase05_ndjson.py`

Expected: PASS; ruff exits `0`.

- [ ] **Step 5: Commit.**

```bash
git add src/slot_extractor/tool_loop/ndjson.py src/slot_extractor/tool_loop/models.py tests/unit/test_phase05_ndjson.py
git commit -m "feat: define phase05 event NDJSON protocol"
```

---

### Task 5: Registry-driven two-model FastAPI API and shared fixture wiring

**Files:**
- Create: `src/slot_extractor/tool_loop/app.py`
- Create: `tests/integration/test_phase05_app.py`
- Modify: `configs/tool_loop/phase05.yaml`

**Interfaces:**
- Consumes `ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))`, `read_and_verify_manifest(spec.manifest_path)`, `LlamaServerManager`, and registry model IDs; it must not maintain a second model catalog.
- Produces `create_app(store: FixtureStore | None = None, registry: ModelRegistry | None = None, backend_factory: Callable[[ModelSpec], Backend] | None = None) -> FastAPI`; production wiring uses `LlamaServerManager`, while tests inject fake backends.
- `GET /api/models` returns every registry target/anchor with `available` and optional `unavailable_reason`; unavailable entries cannot be selected.
- `POST /api/compare` accepts available left/right registry model IDs, execution mode, shared user input, and independent histories, then returns side-tagged `application/x-ndjson` event lines.
- `GET /api/technicians` returns fixture version/date and read-only technician summaries.
- `GET /api/technicians/{name}` returns the request-independent profile and availability details; request-specific matching explanations only appear in `/api/compare` trace events.
- `GET /` serves `static/index.html`; `GET /static/{path}` serves only declared UI assets.

- [ ] **Step 1: Write failing integration tests.**

```python
from fastapi.testclient import TestClient
from slot_extractor.tool_loop.app import create_app

def test_compare_runs_two_registry_models_and_tags_each_event(client: TestClient):
    response = client.post("/api/compare", json={
        "left_model_id": "qwen3-0.6b-base-q4-k-m",
        "right_model_id": "qwen3-0.6b-sft-q4-k-m",
        "mode": "sequential",
        "user_input": "2026年8月12日14点找李明做60分钟精油",
        "left_history": [],
        "right_history": [],
    })
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    assert {event["side"] for event in events} == {"left", "right"}
    assert any(event["type"] == "reply" for event in events if event["side"] == "left")
    assert any(event["type"] == "reply" for event in events if event["side"] == "right")
    assert all(event["comparable"] is True for event in events)

def test_parallel_mode_marks_every_event_non_comparable(client: TestClient):
    payload = {
        "left_model_id": "qwen3-0.6b-base-q4-k-m",
        "right_model_id": "qwen3-0.6b-sft-q4-k-m",
        "mode": "parallel", "user_input": "测试", "left_history": [], "right_history": [],
    }
    events = [json.loads(line) for line in client.post("/api/compare", json=payload).text.splitlines()]
    assert events and all(event["comparable"] is False for event in events)

def test_technician_endpoint_uses_same_fixture(client):
    payload = client.get("/api/technicians").json()
    assert payload["version"] == "phase05-v1"
    assert {item["name"] for item in payload["technicians"]} >= {"王芳", "李明"}
    models = client.get("/api/models").json()
    assert {item["model_id"] for item in models} >= {
        "qwen3-0.6b-base-q4-k-m", "qwen3-0.6b-sft-q4-k-m",
    }
    assert all("available" in item for item in models)
```

- [ ] **Step 2: Run integration tests to verify failure.**

Run: `uv run pytest tests/integration/test_phase05_app.py -v`

Expected: FAIL because `create_app` and routes do not exist.

- [ ] **Step 3: Implement registry-driven two-model app factory and request conversion.**

Load `configs/tool_loop/phase05.yaml` and the canonical `configs/quantization/phase05.yaml` registry once in `create_app`; inject the exact same `FixtureStore` into both sides and `/api/technicians`. Resolve each model ID through `ModelRegistry.get(model_id)`, verify `spec.manifest_path` with `read_and_verify_manifest()`, pass its `ModelSpec` to `backend_factory`, build one backend and `ConversationOrchestrator` per side, and preserve independent histories/current state. In `sequential` mode use the shared `LlamaServerManager` to run the full left tool loop, stop/release it, then run the right loop, and mark metrics comparable. In `parallel` mode start isolated server instances/ports with split thread budgets, run both with bounded two-worker execution, and attach `comparable: false` to every event. Build minimal `Sample` adapters using each side's history and the shared user input. Return side-tagged `StreamingResponse(..., media_type="application/x-ndjson")`; this streams completed events only, never tokens. Unknown model IDs and malformed bodies return HTTP 422; internal loop/tool errors become side-specific events without leaking stack traces.

- [ ] **Step 4: Run tests, lint, and import check.**

Run: `uv run pytest tests/integration/test_phase05_app.py -v && uv run ruff check src/slot_extractor/tool_loop/app.py tests/integration/test_phase05_app.py`

Expected: all integration tests PASS; ruff exits `0`.

- [ ] **Step 5: Commit.**

```bash
git add src/slot_extractor/tool_loop/app.py configs/tool_loop/phase05.yaml tests/integration/test_phase05_app.py
git commit -m "feat: expose phase05 tool loop over FastAPI"
```

---

### Task 6: Native two-model comparison UI, details, explanation and export

**Files:**
- Create: `src/slot_extractor/tool_loop/static/index.html`
- Create: `src/slot_extractor/tool_loop/static/app.js`
- Create: `src/slot_extractor/tool_loop/static/styles.css`
- Modify: `src/slot_extractor/tool_loop/app.py`
- Modify: `tests/integration/test_phase05_app.py`

**Interfaces:**
- UI loads `/api/models`, provides independent left/right model selectors, posts shared input plus separate histories to `/api/compare`, and consumes each side-tagged NDJSON line.
- Left and right model columns each render their own latest `reply` as the primary answer and keep independent multi-turn history.
- Each model column renders collapsible low-contrast JSON/tool result, per-technician trace, status and explanation beneath its reply.
- A shared fixture panel renders `/api/technicians` and availability windows; mode control selects sequential fairness or parallel experience; export downloads both complete event streams as `.ndjson` and a readable `.json` summary.

- [ ] **Step 1: Write failing static/API tests for required two-model UI affordances.**

```python
def test_index_contains_two_columns_and_export_controls(client):
    html = client.get("/").text
    assert 'id="left-model"' in html
    assert 'id="right-model"' in html
    assert 'id="left-conversation"' in html
    assert 'id="right-conversation"' in html
    assert 'id="comparison-mode"' in html
    assert 'id="technician-library"' in html
    assert 'id="export-ndjson"' in html
    assert 'id="export-json"' in html
    assert 'id="parallel-warning"' in html

def test_ui_script_uses_event_ndjson_and_routes_both_sides(client):
    js = client.get("/static/app.js").text
    assert "application/x-ndjson" in js or "split('\\n')" in js
    assert 'event.side === "left"' in js
    assert 'event.side === "right"' in js
    assert "token_delta" not in js
```

- [ ] **Step 2: Run tests to verify failure.**

Run: `uv run pytest tests/integration/test_phase05_app.py -v`

Expected: FAIL because static assets and UI IDs are absent.

- [ ] **Step 3: Implement native UI.**

`index.html` must provide left/right model selectors populated exclusively from `/api/models`, two complete conversation columns, a sequential/parallel mode selector, and a shared technician-library drawer. `app.js` loads `/api/models`, posts shared input and independent side histories to `/api/compare`, uses `response.body.getReader()`, newline buffering and `JSON.parse`, then routes each event by `side`. Each side updates its reply only on `reply` and places its own tool JSON/trace in `<details>`. Sequential mode explains that the full left loop finishes before the right loop and displays comparable metrics; parallel mode displays a persistent “资源竞争，性能不可比较” warning. `styles.css` supplies a responsive two-model grid, muted details backgrounds, readable Chinese text, and mobile single-column fallback. Export buttons serialize both retained event arrays and selected model/fixture metadata via Blob without server-side state.

- [ ] **Step 4: Run static tests and lint.**

Run: `uv run pytest tests/integration/test_phase05_app.py -v && uv run ruff check src/slot_extractor/tool_loop/app.py tests/integration/test_phase05_app.py`

Expected: all tests PASS; ruff exits `0`.

- [ ] **Step 5: Commit.**

```bash
git add src/slot_extractor/tool_loop/static src/slot_extractor/tool_loop/app.py tests/integration/test_phase05_app.py
git commit -m "feat: add native phase05 tool-loop interface"
```

---

### Task 7: End-to-end verification and operator entrypoint

**Files:**
- Modify: `configs/tool_loop/phase05.yaml`
- Modify: `src/slot_extractor/tool_loop/app.py`
- Modify: `tests/integration/test_phase05_app.py`
- Modify: `README.md`

**Interfaces:**
- `python -m uvicorn slot_extractor.tool_loop.app:create_app --factory --host 127.0.0.1 --port 8000` starts the app using the configured fixture and canonical model registry.
- `phase05.yaml` defines `fixture`, `registry: configs/quantization/phase05.yaml`, `max_turns: 3`, `event_protocol: ndjson-v1`, default left/right model IDs, and default `mode: sequential` without embedding technician or model data.

- [ ] **Step 1: Add failing end-to-end assertions for all required branches.**

```python
@pytest.mark.parametrize("text,expected", [
    ("2026年8月12日14点找王芳做60分钟精油", "unavailable"),
    ("2026年8月12日14点找不存在做60分钟精油", "not_found"),
    ("2026年8月12日18点找李明做60分钟精油", "unavailable"),
])
def test_api_exposes_canonical_status_and_trace_on_both_sides(client, text, expected):
    payload = {
        "left_model_id": "qwen3-0.6b-base-q4-k-m",
        "right_model_id": "qwen3-0.6b-sft-q4-k-m",
        "mode": "sequential",
        "user_input": text,
        "left_history": [],
        "right_history": [],
    }
    events = [json.loads(line) for line in client.post("/api/compare", json=payload).text.splitlines()]
    tools = [event for event in events if event["type"] == "tool_result"]
    assert {event["side"] for event in tools} == {"left", "right"}
    assert all(event["payload"]["status"] == expected for event in tools)
    assert all(event["payload"]["trace"] for event in tools)
```

- [ ] **Step 2: Run the complete Phase 05 tests to verify any missing wiring.**

Run: `uv run pytest tests/unit/test_phase05_fixture_store.py tests/unit/test_phase05_find_technicians.py tests/unit/test_phase05_orchestrator.py tests/unit/test_phase05_ndjson.py tests/integration/test_phase05_app.py -v`

Expected: failures identify missing branch wiring before final implementation.

- [ ] **Step 3: Complete configuration and operator documentation.**

Keep all fixture data in `data/fixtures/technicians/phase05-v1.yaml`; configure the app factory to resolve fixture and canonical registry paths from repository root. Document `uv run uvicorn ...`, model switching, independent left/right histories, fixed fixture date/version, sequential fairness, parallel non-comparability, event-level NDJSON behavior, technician panel and export controls in `README.md`. Do not alter evaluator imports or scoring behavior.

- [ ] **Step 4: Run full verification.**

Run: `uv run pytest -q && uv run ruff check . && uv run python -c "from slot_extractor.tool_loop.app import create_app; print(create_app().title)"`

Expected: all tests PASS, ruff exits `0`, and the command prints the FastAPI title.

- [ ] **Step 5: Commit.**

```bash
git add configs/tool_loop/phase05.yaml src/slot_extractor/tool_loop/app.py tests/integration/test_phase05_app.py README.md
git commit -m "feat: verify phase05 tool-loop app"
```

---

## Self-review

- **规格覆盖：** 版本化固定日期 YAML、王芳/李明资料、五参确定性查询、半开区间、四种指定/搜索状态、ambiguity 与 `mock_coverage_miss`、canonical result、逐技师 trace、最多三轮、PromptBuilder/Sample/validators 复用、evaluator 不变、registry 驱动左右模型选择、双侧独立多轮历史、事件级 NDJSON、reply 主显示、淡色 details、技师库/匹配解释、顺序公平、并行体验和双方导出均有任务覆盖。
- **文件边界：** fixture、查询、编排、传输和 UI 分文件；冻结 evaluator 与 `data/eval` 不在修改清单中。
- **类型一致性：** `ToolQuery` → `FindTechniciansExecutor.find` → `CanonicalToolResult` → `ToolLoopEvent` → side-tagged `encode_events` → `/api/compare` 贯通；`ConversationOrchestrator.run(Sample)` 与既有 `PromptBuilder`/validators 对齐；模型选择只消费子计划 1 的 `ModelRegistry`。
- **占位扫描：** 计划不使用 TBD、TODO、"implement later" 或未定义的接口；每个任务含失败测试、命令、Expected、真实实现内容和 commit。
- **依赖审查：** 仅增加 `fastapi` 与 `uvicorn`；不添加前端构建链，不替换 evaluator，并消费量化计划的 canonical registry/manifest/server interfaces。
- **交互审查：** NDJSON 是完整事件增量，不是 token streaming；左右各自显示 reply 和 details，并保持独立多轮历史；顺序公平与并行体验有不同可比性标记；技师库和导出覆盖双方完整事件。
