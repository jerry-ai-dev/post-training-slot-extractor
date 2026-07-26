# Phase 03 Strict Structured Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make GPT-5.6-sol generate 25 schema-valid phase03 raw samples and complete the existing dataset pipeline with exit code 0.

**Architecture:** Define one strict raw JSON Schema and pass it through optional generation parameters to the Responses API `text.format`. Keep semantic validation and bounded corrective retries in `RawGenerator`; do not weaken any existing contract.

**Tech Stack:** Python 3.12, OpenAI Responses-compatible HTTP API, pytest, ruff.

---

### Task 1: Define the strict raw response schema

**Files:**
- Create: `src/slot_extractor/data/raw_schema.py`
- Create: `tests/unit/test_raw_schema.py`

- [ ] Write tests asserting seven required top-level keys, `additionalProperties=false`, final/tool branches, exact final status enums, exact history message shapes, and all five DPO tokens.
- [ ] Run `uv run pytest tests/unit/test_raw_schema.py -v`; expect import failure.
- [ ] Implement `raw_response_schema() -> dict[str, object]` with reusable final, tool-call, history, input and top-level definitions.
- [ ] Run the test and `uv run ruff check src/slot_extractor/data/raw_schema.py tests/unit/test_raw_schema.py`; expect pass.

### Task 2: Transport JSON Schema through generation parameters

**Files:**
- Modify: `src/slot_extractor/inference/base.py`
- Modify: `src/slot_extractor/inference/openai_responses.py`
- Modify: `tests/unit/test_inference_backend.py`

- [ ] Add a failing test that calls the Responses backend with `GenerationParams(response_schema=...)` and asserts request payload `text.format == {type: json_schema, name: phase03_raw, strict: true, schema: ...}`.
- [ ] Run the focused test; expect missing `response_schema` argument.
- [ ] Add optional `response_schema` and `response_schema_name` fields to `GenerationParams`; emit `text.format` only when provided.
- [ ] Add a test proving ordinary Responses calls still omit `text`.
- [ ] Improve empty-output errors to include Responses `status` and `incomplete_details`.
- [ ] Run all inference backend tests and Ruff; expect pass.

### Task 3: Require structured output in RawGenerator

**Files:**
- Modify: `src/slot_extractor/data/generator.py`
- Modify: `tests/unit/test_generator.py`

- [ ] Add a failing backend-spy test asserting every generation attempt receives `GenerationParams` with the phase03 schema and name.
- [ ] Run the test; expect `params` to be `None`.
- [ ] Pass `GenerationParams(max_tokens=4096, response_schema=raw_response_schema(), response_schema_name="phase03_raw")` on every attempt.
- [ ] Include sample id and attempt count in exhausted-retry errors.
- [ ] Run generator tests and Ruff; expect pass.

### Task 4: Verify mock and real pipelines

**Files:**
- Modify: `project-log/phase-03-dataset/log.md`

- [ ] Run `uv run slot-build-dataset --mock --strict-audit --config configs/data/phase03.yaml`; expect raw=25/train=20/val=5/dpo=40.
- [ ] Run phase03 focused tests, full non-local tests, and `uv run ruff check .`; expect all pass.
- [ ] Run `uv run slot-build-dataset --generate --config configs/data/phase03.yaml --output-root experiments/runs/phase03-gpt-smoke`; expect exit 0 and 25/20/5 counts.
- [ ] Repeat the real command once to demonstrate stable generation, then update the phase log with both results and artifact paths.
- [ ] Re-run the complete verification suite before reporting completion.

## Self-review

- Spec coverage: schema, backend transport, retries, atomic build, error reporting, tests and real acceptance are mapped above.
- Placeholder scan: no deferred implementation steps.
- Type consistency: `GenerationParams.response_schema` is the single transport mechanism used by `RawGenerator` and `OpenAIResponsesBackend`.
