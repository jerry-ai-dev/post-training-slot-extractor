# Phase 05 Evaluation Reporting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Windows CPU 上对冻结的 Q4 量化模型集合及两个 SFT F16 GGUF anchors 执行可复现的质量与固定 workload 性能评测，并在 `reports/phase05` 生成真实、可审计的汇总报告。

**Architecture:** 消费子计划 1 已定义的统一 registry/manifest/server-manager 接口；本计划只实现 Phase 05 的评测编排、指标汇总和报告落盘，不重新定义 `ModelSpec`。评测器顺序加载单模型，分别执行预热后的 cold/hot 质量与短/中/2K/4K 固定 benchmark，隔离单模型失败并支持 `--skip-complete` 断点续跑。质量复用现有 `collect_analysis.py`、`default_scorers()` 和 artifacts writer；性能只记录观察值，不调用 `select_winner`，不设 threshold/pass-fail。

**Tech Stack:** Python 3.12、现有 slot-extractor evaluation harness、llama-server OpenAI-compatible API、Windows CPU、PyYAML、httpx、pytest、ruff、JSON/JSONL 报告 artifacts。

## Global Constraints

- 只消费子计划 1 的统一 registry/manifest/server manager 接口；不得重新定义 `ModelSpec`。
- 质量评测使用 8 个 Q4 全冻结集；复用 `scripts/eval/collect_analysis.py`、`default_scorers()` 和现有 artifacts writer。
- 必须包含两个 SFT F16 GGUF anchors；anchor 仅用于对照，不扩展为完整模型矩阵。
- workload 固定为短/中/2K/4K；8K 仅可选，首轮默认不运行。
- Windows CPU 单模型顺序执行；每个模型必须预热，cold 与 hot 必须分开记录。
- 每个 workload 记录 `load_ms`、`prefill_ms`、`ttft_ms`、`decode_ms`、`total_ms`、`tokens`、`peak_rss_mb`、`file_size_bytes`。
- 聚合统计必须包含 `count`、`mean`、`median`、`p90`、`min`、`max`；空/失败 workload 不伪造数值。
- 单模型失败隔离；`--skip-complete` 只跳过完整 artifacts，不跳过部分或失败记录。
- 输出根目录固定为 `reports/phase05`；不调用 `select_winner`，不生成 threshold/pass-fail 结论。
- diff 仅作为观察性输出，不作门禁、不作排名、不作 winner 选择。
- 真实运行必须写入 local marker 与 `project-log/phase-05-quantization-deploy/log.md`；测试不得编造真实数据或冒充 local marker。
- 计划只描述实施，不在本任务中运行真实模型、生成模型文件或填写实验数值。

---

## File map

- `scripts/eval/run_phase05_local.py`: Phase 05 顺序编排入口；消费 `ModelRegistry.from_config`、`read_and_verify_manifest`、`LlamaServerManager`，执行质量和 workload，隔离失败并写完成 marker。
- `scripts/eval/phase05_metrics.py`: workload 原始样本、cold/hot 记录及 `count/mean/median/p90/min/max` 聚合接口；不包含质量 scorer。
- `scripts/eval/phase05_artifacts.py`: 统一写单模型 result、失败记录、matrix summary、observational diff 和真实-run marker。
- `scripts/eval/phase05_reports.py`: 将已落盘 artifacts 渲染为机器可读 JSON 与人读 Markdown；不执行选择逻辑。
- `configs/evaluation/phase05.yaml`: 仅保存 8 个 Q4 ID、两个 SFT F16 anchor ID、dataset、workload、预热和 CPU 顺序策略；模型元数据仍来自 `configs/quantization/phase05.yaml`。
- `tests/unit/test_phase05_metrics.py`: 统计和 cold/hot schema 的 TDD 测试。
- `tests/unit/test_phase05_artifacts.py`: 原子写入、失败隔离、skip-complete、marker 和无 pass/fail 字段测试。
- `tests/unit/test_run_phase05_local.py`: registry/server-manager 调用顺序、质量复用、单模型隔离的编排测试。
- `tests/integration/test_phase05_reports.py`: fixture-only 报告 schema、diff 观察性和 no-winner 约束测试。
- `project-log/phase-05-quantization-deploy/log.md`: 真实 Windows local marker、命令、环境和结果索引；不填写未运行的数据。

### Task 1: Lock Phase 05 registry consumption and evaluation config

**Files:**
- Create: `configs/evaluation/phase05.yaml`
- Create: `tests/unit/test_run_phase05_local.py`
- Modify: `scripts/eval/run_phase05_local.py`

**Interfaces:**
- Consumes: 子计划 1 提供的 `ModelRegistry.from_config(path: Path) -> ModelRegistry`、`ModelRegistry.get(model_id) -> ModelSpec`、`read_and_verify_manifest(path: Path) -> StageManifest`、`LlamaServerManager.start(model_id: str, log_path: Path) -> subprocess.Popen[str]`、`.wait_ready(process, timeout_s) -> None` 和 `.stop(process) -> None`。
- Produces: `load_phase05_config(path: Path) -> Phase05Config`、`run_matrix(config_path: Path, *, skip_complete: bool = False) -> MatrixSummary`。

- [ ] **Step 1: Write the failing test**

```python
def test_phase05_config_has_eight_q4_and_two_f16_anchors(tmp_path):
    config = load_phase05_config(Path("configs/evaluation/phase05.yaml"))
    registry = ModelRegistry.from_config(config.registry)
    assert len(config.q4_model_ids) == 8
    assert tuple(m.model_id for m in registry.quantization_targets()) == config.q4_model_ids
    assert len(config.f16_anchor_ids) == 2
    assert tuple(m.model_id for m in registry.anchors()) == config.f16_anchor_ids
    assert config.workloads == ("short", "medium", "2k", "4k")
    assert config.include_8k is False
    assert config.execution == "windows_cpu_sequential"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_run_phase05_local.py::test_phase05_config_has_eight_q4_and_two_f16_anchors -v`

Expected: FAIL because the Phase 05 config loader and config file do not exist.

- [ ] **Step 3: Write minimal implementation**

Add the exact configuration contract:

```yaml
registry: configs/quantization/phase05.yaml
manifests_root: models/quantization
reports_root: reports/phase05
quality_cases: data/eval/test.jsonl
q4_model_ids:
  - qwen3-0.6b-base-q4-k-m
  - qwen3-0.6b-sft-q4-k-m
  - qwen3-0.6b-dpo-b01-q4-k-m
  - qwen3-0.6b-dpo-b03-q4-k-m
  - qwen3-1.7b-base-q4-k-m
  - qwen3-1.7b-sft-q4-k-m
  - qwen3-1.7b-dpo-b01-q4-k-m
  - qwen3-1.7b-dpo-b03-q4-k-m
f16_anchor_ids: [qwen3-0.6b-sft-f16, qwen3-1.7b-sft-f16]
workloads: [short, medium, 2k, 4k]
include_8k: false
warmup_requests: 1
repetitions: 5
execution: windows_cpu_sequential
```

Implement `Phase05Config` as a frozen dataclass and parse/validate only these fields; obtain model metadata from `ModelRegistry.from_config(config.registry)` rather than creating a second model-spec type. Validation must assert that Q4 IDs exactly equal `registry.quantization_targets()` and anchor IDs exactly equal `registry.anchors()` in registry order.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_run_phase05_local.py::test_phase05_config_has_eight_q4_and_two_f16_anchors -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add configs/evaluation/phase05.yaml scripts/eval/run_phase05_local.py tests/unit/test_run_phase05_local.py
git commit -m "feat: define phase05 evaluation matrix"
```

### Task 2: Implement timing and resource aggregation

**Files:**
- Create: `scripts/eval/phase05_metrics.py`
- Create: `tests/unit/test_phase05_metrics.py`

**Interfaces:**
- Consumes: `WorkloadSample(workload: str, phase: Literal["cold", "hot"], load_ms: float | None, prefill_ms: float | None, ttft_ms: float | None, decode_ms: float | None, total_ms: float | None, tokens: int | None, peak_rss_mb: float | None, file_size_bytes: int | None)`.
- Produces: `summarize(values: Sequence[float | int]) -> dict[str, float | int]`, `aggregate_workload(samples: Sequence[WorkloadSample]) -> WorkloadAggregate`.

- [ ] **Step 1: Write the failing test**

```python
def test_aggregate_reports_required_statistics_and_separates_cold_hot():
    rows = [WorkloadSample("short", "cold", 100, 20, 25, 40, 65, 10, 512, 1000),
            WorkloadSample("short", "hot", 10, 20, 22, 35, 57, 10, 510, 1000),
            WorkloadSample("short", "hot", 12, 21, 23, 36, 59, 11, 511, 1000)]
    result = aggregate_workload(rows)
    assert set(result.phases) == {"cold", "hot"}
    assert result.phases["hot"].total_ms == {
        "count": 2, "mean": 58.0, "median": 58.0, "p90": 58.8,
        "min": 57.0, "max": 59.0,
    }
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_phase05_metrics.py -v`

Expected: FAIL with missing `phase05_metrics` symbols.

- [ ] **Step 3: Write minimal implementation**

Implement deterministic percentile interpolation for P90 and omit a metric when all values are `None`; retain `count` as the number of usable values for that metric. Do not turn missing measurements into zero and do not add pass/fail fields.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_phase05_metrics.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/phase05_metrics.py tests/unit/test_phase05_metrics.py
git commit -m "feat: aggregate phase05 workload metrics"
```

### Task 3: Write model artifacts, failures, and completion markers

**Files:**
- Create: `scripts/eval/phase05_artifacts.py`
- Create: `tests/unit/test_phase05_artifacts.py`

**Interfaces:**
- Consumes: `ModelResult`, `WorkloadAggregate`, quality payload from `collect_analysis.py`, and `model_id`/manifest provenance from registry.
- Produces: `write_model_result(root: Path, model_id: str, payload: Mapping[str, Any]) -> Path`, `write_failure(root: Path, model_id: str, error: BaseException) -> Path`, `is_complete(root: Path, model_id: str) -> bool`, `write_matrix_summary(root: Path, payload: Mapping[str, Any]) -> Path`, `write_local_marker(root: Path, marker: Mapping[str, Any]) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
def test_failure_isolated_and_complete_requires_result_and_marker(tmp_path):
    failed_id = "qwen3-0.6b-base-q4-k-m"
    complete_id = "qwen3-0.6b-sft-q4-k-m"
    write_failure(tmp_path, failed_id, RuntimeError("server exited"))
    assert json.loads((tmp_path / failed_id / "failure.json").read_text())["status"] == "failed"
    assert not is_complete(tmp_path, failed_id)
    write_model_result(tmp_path, complete_id, {"status": "complete", "quality": {}, "workloads": {}})
    write_local_marker(tmp_path, {"marker": "phase05-local", "models": [complete_id]})
    assert is_complete(tmp_path, complete_id)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_phase05_artifacts.py -v`

Expected: FAIL because Phase 05 artifact functions are undefined.

- [ ] **Step 3: Write minimal implementation**

Write JSON atomically through a sibling temporary file and `Path.replace()`. A complete model directory must contain `result.json`, `workloads.json`, `quality.json`, `manifest.json`, and `complete.marker`; a failure directory contains only auditable error metadata and is never considered complete. `write_matrix_summary` must preserve failed model IDs and must not emit `winner`, `threshold`, `pass`, or `fail` keys.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_phase05_artifacts.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/phase05_artifacts.py tests/unit/test_phase05_artifacts.py
git commit -m "feat: persist phase05 evaluation artifacts"
```

### Task 4: Add quality evaluation reuse and sequential server lifecycle

**Files:**
- Modify: `scripts/eval/run_phase05_local.py`
- Modify: `scripts/eval/collect_analysis.py`
- Create: `tests/unit/test_run_phase05_local.py`

**Interfaces:**
- Consumes: `ModelRegistry.from_config()`、`read_and_verify_manifest(spec.manifest_path)`、`LlamaServerManager.base_url`、`build_backend_from_config()`、`default_scorers()`、`collect_analysis.main([...])`，以及现有 `write_phase04_artifacts` writer contract。
- Produces: `evaluate_model(spec: ModelSpec, config: Phase05Config) -> ModelResult`; `run_matrix(...)` processes models in registry order and stops each server in `finally`.

- [ ] **Step 1: Write the failing test**

```python
def test_run_matrix_is_sequential_reuses_quality_and_isolates_failure(fake_registry, fake_manager, monkeypatch):
    calls = []
    monkeypatch.setattr("scripts.eval.run_phase05_local.collect_analysis_main",
                        lambda argv: calls.append(("quality", argv)))
    monkeypatch.setattr("scripts.eval.run_phase05_local.measure_workloads",
                        lambda *args, **kwargs: calls.append(("workload", args)) or {})
    summary = run_matrix("configs/evaluation/phase05.yaml", server_manager=fake_manager)
    assert calls[0][0] == "quality"
    assert summary.failed_model_ids == ["qwen3-0.6b-dpo-b01-q4-k-m"]
    assert fake_manager.start_order == fake_manager.stop_order
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_run_phase05_local.py -v`

Expected: FAIL because the Phase 05 lifecycle and measurement functions are not implemented.

- [ ] **Step 3: Write minimal implementation**

For each registry `ModelSpec`, call `read_and_verify_manifest(spec.manifest_path)`, start exactly one server with `LlamaServerManager.start(model_id, log_path)`, wait for readiness, run one warmup request, then run quality collection against `data/eval/test.jsonl` using the existing `collect_analysis.py` path and `default_scorers()`. Run cold measurements after load and hot measurements after warmup; stop the server in `finally`. Catch model-scoped exceptions, write `failure.json`, and continue to the next model. The orchestrator must never import or call `select_phase04.select_winner`.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_run_phase05_local.py -v`

Expected: PASS, with no server overlap and one failed model not preventing later models.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/run_phase05_local.py scripts/eval/collect_analysis.py tests/unit/test_run_phase05_local.py
git commit -m "feat: run phase05 models sequentially"
```

### Task 5: Implement fixed workload measurement and optional 8K guard

**Files:**
- Modify: `scripts/eval/run_phase05_local.py`
- Modify: `scripts/eval/phase05_metrics.py`
- Modify: `tests/unit/test_run_phase05_local.py`
- Modify: `tests/unit/test_phase05_metrics.py`

**Interfaces:**
- Consumes: `LlamaServerManager.base_url`、已启动的 `subprocess.Popen[str]`、`Phase05Config.workloads` 和 backend generation timing/token fields。
- Produces: `measure_workloads(base_url: str, process: subprocess.Popen[str], workloads: Sequence[str], *, warmup_requests: int, repetitions: int) -> list[WorkloadSample]`。

- [ ] **Step 1: Write the failing test**

```python
def test_default_workloads_exclude_8k_and_capture_cold_hot_fields(fake_server):
    rows = measure_workloads(fake_server, ("short", "medium", "2k", "4k"), warmup_requests=1, repetitions=2)
    assert {row.workload for row in rows} == {"short", "medium", "2k", "4k"}
    assert {row.phase for row in rows} == {"cold", "hot"}
    assert set(row.to_dict()) >= {"load_ms", "prefill_ms", "ttft_ms", "decode_ms", "total_ms", "tokens", "peak_rss_mb", "file_size_bytes"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_run_phase05_local.py::test_default_workloads_exclude_8k_and_capture_cold_hot_fields -v`

Expected: FAIL because workload measurement is absent.

- [ ] **Step 3: Write minimal implementation**

Define fixed prompt fixtures for short, medium, 2K and 4K; assert their token-budget labels in the measurement payload. Capture process RSS around each request using the Windows-compatible process API already selected by the project, file size from the manifest path, and server-reported prefill/decode/TTFT where available. If `include_8k` is true, append 8K only after the four required workloads. Never silently substitute an 8K result for a required workload.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_run_phase05_local.py::test_default_workloads_exclude_8k_and_capture_cold_hot_fields -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/run_phase05_local.py scripts/eval/phase05_metrics.py tests/unit/test_run_phase05_local.py tests/unit/test_phase05_metrics.py
git commit -m "feat: measure fixed phase05 workloads"
```

### Task 6: Render reports and observational diffs without selection

**Files:**
- Create: `scripts/eval/phase05_reports.py`
- Create: `tests/integration/test_phase05_reports.py`
- Modify: `scripts/eval/run_phase05_local.py`

**Interfaces:**
- Consumes: `reports/phase05/models/*/result.json`, existing `diff_predictions()` for optional quality observations, and matrix summary artifacts.
- Produces: `render_phase05_reports(reports_root: Path) -> tuple[Path, Path]` writing `summary.json` and `summary.md`; `write_observational_diff(...) -> Path`.

- [ ] **Step 1: Write the failing test**

```python
def test_report_is_observational_and_has_no_selection_or_gate_fields(fixture_reports):
    summary_json, summary_md = render_phase05_reports(fixture_reports)
    payload = json.loads(summary_json.read_text())
    text = summary_md.read_text()
    assert "winner" not in payload
    assert "threshold" not in payload
    assert "pass/fail" not in text.lower()
    assert payload["comparison_mode"] == "observational"
    assert "cold" in payload["models"][0]["workloads"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/integration/test_phase05_reports.py -v`

Expected: FAIL because the Phase 05 report renderer does not exist.

- [ ] **Step 3: Write minimal implementation**

Render one row per model with provenance, quality aggregate dimensions, cold/hot workload aggregates, and explicit failed/skipped status. Render Q4-versus-F16 differences only as `observations`; do not calculate a winner, threshold, pass rate gate, or recommendation. Preserve missing values as `null` and include the exact source artifact paths.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/integration/test_phase05_reports.py -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/phase05_reports.py scripts/eval/run_phase05_local.py tests/integration/test_phase05_reports.py
git commit -m "feat: render observational phase05 reports"
```

### Task 7: Add operator command, skip-complete, and real-run logging contract

**Files:**
- Modify: `scripts/eval/run_phase05_local.py`
- Modify: `scripts/eval/phase05_artifacts.py`
- Modify: `tests/unit/test_phase05_artifacts.py`
- Modify: `project-log/phase-05-quantization-deploy/log.md`

**Interfaces:**
- Consumes: all Phase 05 interfaces above.
- Produces: CLI flags `--config`, `--skip-complete`, `--include-8k`, `--reports-root`; marker schema `{ "marker": "phase05-local", "started_at": str, "finished_at": str | null, "host": str, "platform": str, "python": str, "models_attempted": list[str], "models_completed": list[str], "models_failed": list[str], "real_run": true }`.

- [ ] **Step 1: Write the failing test**

```python
def test_skip_complete_does_not_skip_partial_or_failed_models(tmp_path):
    ids = [
        "qwen3-0.6b-base-q4-k-m",
        "qwen3-0.6b-sft-q4-k-m",
        "qwen3-0.6b-dpo-b01-q4-k-m",
    ]
    write_failure(tmp_path, ids[0], RuntimeError("x"))
    (tmp_path / ids[1]).mkdir()
    (tmp_path / ids[1] / "result.json").write_text("{}")
    assert models_to_run(tmp_path, ids, skip_complete=True) == ids
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_phase05_artifacts.py::test_skip_complete_does_not_skip_partial_or_failed_models -v`

Expected: FAIL because skip-complete selection is not implemented.

- [ ] **Step 3: Write minimal implementation**

Add `models_to_run()` based only on `is_complete()`. Add CLI handling that creates `reports/phase05/local-run.json` with `real_run: true` only when the actual command is run; fixture/unit paths must never write this marker. Update the phase-05 log with the exact command, CPU/OS/server versions, artifact locations, and a blank result table until real execution occurs. The command must print `skip complete: <model_id>` only for complete model directories.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_phase05_artifacts.py::test_skip_complete_does_not_skip_partial_or_failed_models -v`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add scripts/eval/run_phase05_local.py scripts/eval/phase05_artifacts.py tests/unit/test_phase05_artifacts.py project-log/phase-05-quantization-deploy/log.md
git commit -m "feat: add phase05 resumable local command"
```

### Task 8: Full verification and self-review

**Files:**
- Modify: `docs/superpowers/plans/2026-08-12-phase05-evaluation-reporting.md` only if review finds a plan defect.

**Interfaces:**
- Consumes: all files and interfaces listed above.
- Produces: verified implementation handoff; no real model data is generated by this task.

- [ ] **Step 1: Run focused tests**

Run: `uv run pytest tests/unit/test_phase05_metrics.py tests/unit/test_phase05_artifacts.py tests/unit/test_run_phase05_local.py tests/integration/test_phase05_reports.py -v`

Expected: all Phase 05 tests PASS.

- [ ] **Step 2: Run repository checks**

Run: `uv run pytest -q && uv run ruff check .`

Expected: all tests PASS and Ruff exits 0.

- [ ] **Step 3: Perform self-review against the specification**

Check that every requirement is covered: exact `ModelRegistry.from_config`/`read_and_verify_manifest`/`LlamaServerManager` consumption without a second `ModelSpec`; eight Q4 full-set quality runs; two SFT F16 anchors; reuse of `collect_analysis`, `default_scorers`, and artifact writing; short/medium/2K/4K with 8K opt-in; Windows CPU sequential lifecycle; warmup and cold/hot separation; all eight raw metrics; six aggregate statistics; model failure isolation; skip-complete; `reports/phase05`; no `select_winner`; no thresholds/pass-fail; observational diff only; and real local marker/log rules. Search the plan and implementation for `TBD`, `TODO`, `select_winner`, and threshold language, then remove any accidental implementation requirement that contradicts these constraints.

- [ ] **Step 4: Commit any review-only plan correction**

```bash
git add docs/superpowers/plans/2026-08-12-phase05-evaluation-reporting.md
git commit -m "docs: refine phase05 evaluation plan"
```

## Self-review

- **Spec coverage:** Tasks 1–2 consume the canonical quantization registry and cover the frozen matrix, workload definitions, warmup and complete metric schema; Tasks 3–4 cover artifact provenance, quality reuse, sequential server management, failure isolation and resumability; Tasks 5–6 cover timing/resource measurements and observational reports; Task 7 covers local marker/logging; Task 8 covers verification and explicit no-selection review.
- **Placeholder scan:** No `TBD`, `TODO`, or unspecified implementation step is used; every task names exact files, interfaces, commands, and expected outcomes.
- **Type consistency:** `Phase05Config`, `WorkloadSample`, `WorkloadAggregate`, artifact writer functions, and orchestration signatures are defined before downstream use. Quantization plan interfaces are consumed exactly as `ModelRegistry.from_config`, `read_and_verify_manifest`, and `LlamaServerManager`; its `ModelSpec` remains the only model-spec contract.
- **Data integrity:** No task fabricates model outputs, latency, RSS, file sizes, or quality scores. Fixture tests use synthetic values only and are explicitly separate from the real local marker.
- **Scope:** This plan creates only the implementation plan requested; it does not implement or run Phase 05.
