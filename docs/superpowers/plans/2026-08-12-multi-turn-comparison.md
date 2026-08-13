# Multi-turn Comparison Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Preserve visible turns and independent left/right model context until the user explicitly clears the conversation.

**Architecture:** Keep session state in the existing browser `histories` object. Commit natural user/assistant pairs as streamed reply events arrive, append each new turn to the DOM, and expose one reset control for all browser session state.

**Tech Stack:** Browser JavaScript, HTML, CSS, pytest static integration contracts.

---

### Task 1: Specify multi-turn browser behavior

**Files:**
- Test: `tests/integration/test_phase05_app.py`

- [ ] Add failing assertions for persistent DOM, per-side history updates, visible user turns, and explicit reset.
- [ ] Run `.venv\Scripts\pytest.exe -q tests/integration/test_phase05_app.py` and confirm the new test fails for missing behavior.

### Task 2: Implement persistent independent conversations

**Files:**
- Modify: `src/slot_extractor/tool_loop/static/index.html`
- Modify: `src/slot_extractor/tool_loop/static/app.js`
- Modify: `src/slot_extractor/tool_loop/static/styles.css`

- [ ] Add the reset control and user-turn styling.
- [ ] Snapshot the submitted input, append it to both columns, and stop clearing the columns at run start.
- [ ] On each non-empty reply, append the submitted user message and that side's assistant reply to only that side's history.
- [ ] Reset both histories, events, columns, and statuses only from the reset control.
- [ ] Run the focused integration tests and confirm they pass.

### Task 3: Verify and deploy

**Files:**
- Test: `tests/integration/test_phase05_app.py`

- [ ] Run Ruff and the complete pytest suite.
- [ ] Restart the owned app on port 8001 and verify root health.
- [ ] Commit the implementation on `phase05-quantization-inference-acceleration` without merging.
