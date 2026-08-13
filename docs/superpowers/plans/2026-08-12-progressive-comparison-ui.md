# Progressive Comparison UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stream each model's lifecycle and result into its own column, report generation-only inference time, and keep expanded traces within the desktop two-column grid.

**Architecture:** Convert the comparison response into a lock-owning generator that emits lifecycle dictionaries around each side's existing trace. Wrap each backend in a generation timer so server startup is excluded. Extend the static UI to render per-side statuses and constrain trace content with CSS.

**Tech Stack:** Python 3.12, FastAPI `StreamingResponse`, NDJSON, browser Fetch streams, CSS Grid, pytest.

---

### Task 1: Stream lifecycle events and inference timing

**Files:**
- Modify: `src/slot_extractor/tool_loop/app.py`
- Modify: `src/slot_extractor/tool_loop/ndjson.py`
- Test: `tests/integration/test_phase05_app.py`

- [ ] **Step 1: Write failing lifecycle and timing tests**

Add a recording backend and parse the streaming response to assert event order is `left loading`, `left inferencing`, left trace, `left complete`, then the equivalent right events. Assert `inference_duration_ms` is positive and excludes a simulated setup delay.

- [ ] **Step 2: Verify the tests fail**

Run: `.venv\Scripts\pytest.exe -q tests/integration/test_phase05_app.py`

Expected: lifecycle assertions fail because only trace events currently exist and both sides are accumulated before response construction.

- [ ] **Step 3: Implement timing decorator and streaming generator**

Add a backend decorator whose `generate()` measures `perf_counter()` around the delegated call and accumulates milliseconds. Replace `combined` accumulation with a generator that holds `comparison_lock`, emits lifecycle dictionaries, creates and starts each backend, runs the orchestrator through the timer, yields trace events, emits completion timing, and always stops the owned server.

- [ ] **Step 4: Verify backend tests pass**

Run: `.venv\Scripts\pytest.exe -q tests/integration/test_phase05_app.py`

Expected: all integration tests pass, including comparison serialization.

### Task 2: Render independent side progress

**Files:**
- Modify: `src/slot_extractor/tool_loop/static/index.html`
- Modify: `src/slot_extractor/tool_loop/static/app.js`
- Test: `tests/integration/test_phase05_app.py`

- [ ] **Step 1: Write failing static contract tests**

Assert the HTML contains `left-status` and `right-status`. Assert JavaScript handles `side_status`, renders waiting/loading/inferencing/complete states, and formats `inference_duration_ms` without using the total fetch duration.

- [ ] **Step 2: Verify the tests fail**

Run: `.venv\Scripts\pytest.exe -q tests/integration/test_phase05_app.py::test_browser_renders_independent_side_progress`

Expected: failure because per-side status elements and handlers do not exist.

- [ ] **Step 3: Implement side state rendering**

Add one accessible status element to each model card. Reset both to waiting on click. Route lifecycle events to status rendering rather than trace rendering; immediately append ordinary events. Show completion as `推理完成 · X.XX 秒` and show failure in the affected column.

- [ ] **Step 4: Verify UI contract tests pass**

Run: `.venv\Scripts\pytest.exe -q tests/integration/test_phase05_app.py`

Expected: all integration tests pass.

### Task 3: Preserve two-column layout with expanded traces

**Files:**
- Modify: `src/slot_extractor/tool_loop/static/styles.css`
- Test: `tests/integration/test_phase05_app.py`

- [ ] **Step 1: Write failing CSS contract test**

Assert CSS constrains `.grid article`, `.event`, `details`, and `pre` with `min-width:0`, `max-width:100%`, `overflow-x:auto`, and wrapping rules.

- [ ] **Step 2: Verify the test fails**

Run: `.venv\Scripts\pytest.exe -q tests/integration/test_phase05_app.py::test_expanded_trace_cannot_break_comparison_grid`

Expected: failure because the current CSS has no overflow constraints.

- [ ] **Step 3: Add resilient grid and trace styles**

Constrain grid tracks with `minmax(0,1fr)`, set grid children and events to `min-width:0`, and set trace `pre` blocks to `max-width:100%`, `white-space:pre-wrap`, `overflow-wrap:anywhere`, and `overflow-x:auto`.

- [ ] **Step 4: Verify the complete suite and real app**

Run: `.venv\Scripts\ruff.exe check .` and `.venv\Scripts\pytest.exe -q`.

Restart only the PID recorded in `reports/phase05/app/uvicorn-8001.pid`, then POST the existing `hello-request.json`. Assert HTTP 200, left completion precedes right inference, both sides complete, both durations are positive, and the app root returns HTTP 200.

- [ ] **Step 5: Commit implementation**

Run: `git add src/slot_extractor/tool_loop tests/integration/test_phase05_app.py docs/superpowers/plans/2026-08-12-progressive-comparison-ui.md && git commit -m "fix: stream per-model inference progress"`.
