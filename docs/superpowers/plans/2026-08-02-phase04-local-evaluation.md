# Phase 04 Local Evaluation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 将阶段四改为 AutoDL 仅训练六组 LoRA，下载后在 Windows CPU 顺序评估六组，本地生成选型与报告。

**Architecture:** AutoDL 输出六个自包含 run 目录并打包。本地 Python 编排器按 manifest 生成 LLaMA-Factory API 配置，顺序启动 base + adapter、健康检查、调用冻结评估链路、停止服务，并支持断点续跑。六组选型后只对冠军 merge/GGUF/llama.cpp 复评。

**Tech Stack:** Python 3.12、PowerShell/Windows subprocess、LLaMA-Factory 0.9.5 API、httpx、PyYAML、pytest、现有 OpenAI-compatible 评估 harness。

---

## File structure

- `scripts/train/package_cloud_artifacts.py`: 校验六个训练 run 并生成可下载 zip 包。
- `scripts/eval/run_phase04_local.py`: 本地六 run 服务生命周期、健康检查、断点续跑、评估及后处理入口。
- `scripts/eval/phase04_artifacts.py`: 在 scorecard 写入评估环境/后端元数据。
- `scripts/eval/render_phase04_reports.py`: 在 M1/M2 报告中显式声明时延口径。
- `tests/unit/test_package_cloud_artifacts.py`: 云端包完整性测试。
- `tests/unit/test_run_phase04_local.py`: 配置生成、顺序、跳过与清理行为测试。
- `tests/unit/test_phase04_eval_artifacts.py`: scorecard 后端元数据测试。
- `tests/integration/test_phase04_pipeline.py`: 报告口径声明测试。
- `project-log/phase-04-training/log.md`: AutoDL 训练/打包/下载和本地评估操作手册。

### Task 1: Cloud artifact package

- [ ] 在 `tests/unit/test_package_cloud_artifacts.py` 写失败测试：六个 `phase04-*` 目录必须通过 `verify_artifacts(..., training_only=True)`，zip 内保留 run 目录层级；缺任一 run 必须失败。
- [ ] 运行 `uv run pytest tests/unit/test_package_cloud_artifacts.py -v`，预期因模块不存在而 FAIL。
- [ ] 实现 `package_runs(runs_root: Path, output: Path) -> Path`，固定六个 run id，先全部校验再原子生成 zip。
- [ ] 重跑测试，预期 PASS。

### Task 2: Local sequential evaluator

- [ ] 在 `tests/unit/test_run_phase04_local.py` 写失败测试：`build_api_config()` 必须从 manifest 产生 `model_name_or_path`、绝对 `adapter_name_or_path`、`template: qwen3`、`enable_thinking: false`、`infer_backend: huggingface`、`device_map: cpu`。
- [ ] 运行单测，预期 FAIL；实现配置生成后重跑，预期 PASS。
- [ ] 写失败测试固定顺序和断点：两个 SFT 先于四个 DPO；只有 predictions/scorecard/server.log 都存在才可 `--skip-complete`。
- [ ] 实现 `run_matrix()`：验证训练产物，用 `subprocess.Popen(["llamafactory-cli", "api", config])` 启动，轮询 `/v1/models`，调用 `python -m scripts.eval.collect_analysis --run-id ... --run-dir ...`，`finally` 中终止并等待子进程。
- [ ] 写失败测试固定健康超时或评估失败时仍停止服务；实现最小清理逻辑并重跑。

### Task 3: Evaluation provenance and reports

- [ ] 先修改 `tests/unit/test_phase04_eval_artifacts.py`，要求 scorecard 含 `evaluation_environment.backend=llamafactory_huggingface`、`device=cpu`、`latency_comparable_to_m0=false`，运行确认 FAIL。
- [ ] 给 `write_phase04_artifacts()` 增加可选 `evaluation_environment`，由 `collect_analysis.py` CLI 传入本地 phase04 元数据，重跑确认 PASS。
- [ ] 先修改集成测试，要求 M1/M2 都出现“LLaMA-Factory CPU 时延不与 M0 llama.cpp 直接比较”，运行确认 FAIL。
- [ ] 修改 `render_phase04_reports.py`，重跑确认 PASS。

### Task 4: Local post-processing

- [ ] 在 `test_run_phase04_local.py` 写失败测试，要求六组都完整后按四个 SFT→DPO 父子对生成 diff，再运行 selection 和 reports；任一组不完整则不选冠军。
- [ ] 实现后处理命令调用和稳定的 diff 输出路径，重跑确认 PASS。

### Task 5: Operator documentation and verification

- [ ] 更新 `project-log/phase-04-training/log.md`：AutoDL 只跑 `run_matrix.sh` 和打包命令；下载/解压后在 Windows 运行一条本地评估命令；说明断点续跑、产物、时延口径和冠军复评。
- [ ] 更新旧计划 `docs/superpowers/plans/2026-07-30-phase04-training.md` 的 Goal/Task 10/Task 11，标注本计划取代云端评估步骤。
- [ ] 运行 `uv run pytest tests/unit/test_package_cloud_artifacts.py tests/unit/test_run_phase04_local.py tests/unit/test_phase04_eval_artifacts.py tests/integration/test_phase04_pipeline.py -v`，预期全部 PASS。
- [ ] 运行 `uv run pytest -q`与 `uv run ruff check .`，预期全部 PASS。

## Self-review

- 规格覆盖：云端只训练和打包、六组下载、Windows CPU base+LoRA 顺序评估、断点续跑、冻结评分、diff/选型/报告、后端时延声明和冠军 llama.cpp 复评均有对应任务。
- 范围控制：不修改冻结数据或 scorer，不为六个候选全量 merge/GGUF。
- 接口一致：全链路继续以 run_id 和 `experiments/runs/phase04-<run_id>` 为唯一关联键。
- 占位扫描：无 TBD/TODO；外部训练权重未存在时只运行自动化测试，不伪造真实评估结果。
