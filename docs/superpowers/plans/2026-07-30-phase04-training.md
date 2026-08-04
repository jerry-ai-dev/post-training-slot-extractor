# Phase 04 Training and Experiment Matrix Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建成可审计的 Qwen3-0.6B/1.7B SFT→DPO 六组实验流水线，完成本地空跑、云端训练、Windows CPU 本地评估与 M1/M2 选型。

> 2026-08-02 变更：Task 10/11 中的云端评估路径已被 `2026-08-02-phase04-local-evaluation.md` 取代。AutoDL 仅训练和打包；六组评估、diff、选型和报告均在本地执行。

**Architecture:** 配置以 base + run override 为唯一真源，由纯 Python 渲染器生成 LLaMA-Factory 完整 YAML；训练编排严格按 SFT 先于 DPO 的依赖顺序执行并逐 run 回收产物。评估继续调用冻结的现有 scorer，只在外围增加统一的逐条产物、聚合摘要、run diff、选型与报告生成，不修改 `src/slot_extractor/evaluation/` 的评分语义。

**Tech Stack:** Python 3.12、PyYAML、pytest、LLaMA-Factory 0.9.5、Transformers/PEFT/TRL/PyTorch、Bash、OpenAI-compatible chat completions API。

---

## 文件结构与职责

- `requirements-train.txt`：本地和 AutoDL 共用的精确训练依赖锁。
- `configs/training/llamafactory/_base_{sft,dpo}.yaml`：共享训练参数。
- `configs/training/llamafactory/{sft,dpo}/*.yaml`：六个 run 的最小差异配置。
- `configs/training/llamafactory/_rendered/`：生成物，不入 git；每个 run 会另存一份可审计快照。
- `configs/inference/phase04-*.yaml`：六个 OpenAI-compatible 评估端点配置，统一 no-think。
- `scripts/train/render_config.py`：合并、校验和渲染配置，不启动训练。
- `scripts/train/dryrun.py`：用 0.6B、CPU/fp32、两步训练验证 SFT/DPO 链路。
- `scripts/train/collect_artifacts.py`：把配置、版本、日志和 adapter 原子回收到固定 run 目录。
- `scripts/train/run_matrix.sh`：按依赖顺序训练六个 run，每个成功后立即回收。
- `scripts/eval/collect_analysis.py`：保留现有评分逻辑，扩展为标准 phase04 产物写入器。
- `scripts/eval/diff_runs.py`：纯离线比较两份 `predictions.jsonl`。
- `scripts/eval/select_phase04.py`：执行预注册红线和 tie-break，输出机器可读结论。
- `scripts/eval/render_phase04_reports.py`：从已有 JSON 生成 M1/M2 README，不重新推理。
- `tests/unit/test_*phase04*.py`：配置、回收、diff、选型和报告单测。
- `tests/integration/test_phase04_pipeline.py`：完全离线的六 run 假产物集成测试。

## Task 1：锁定环境与配置合同

**Files:**
- Create: `requirements-train.txt`
- Modify: `configs/training/llamafactory/VERSION`
- Create: `tests/unit/test_phase04_environment.py`

- [ ] **Step 1: 写失败测试固定版本合同**

```python
from pathlib import Path


def test_training_versions_are_exactly_pinned():
    lines = Path("requirements-train.txt").read_text(encoding="utf-8").splitlines()
    expected = {
        "llamafactory==0.9.5",
        "transformers==5.6.0",
        "peft==0.18.1",
        "trl==0.24.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
    }
    assert set(lines) == expected
    assert Path("configs/training/llamafactory/VERSION").read_text() == "v0.9.5\n"
```

- [ ] **Step 2: 运行并确认失败**

Run: `uv run pytest tests/unit/test_phase04_environment.py -v`

Expected: FAIL，`requirements-train.txt` 不存在。

- [ ] **Step 3: 写入精确版本清单并验证依赖解析**

```text
llamafactory==0.9.5
transformers==5.6.0
peft==0.18.1
trl==0.24.0
torch==2.6.0
torchvision==0.21.0
torchaudio==2.6.0
```

Run: `python -m pip install --dry-run -r requirements-train.txt`

Expected: exit 0；若包元数据报告冲突，只调整被冲突信息明确指出的四个下游版本，并同步测试中的 `expected`，不得放宽为范围版本。

- [ ] **Step 4: 运行合同测试并提交**

Run: `uv run pytest tests/unit/test_phase04_environment.py -v`

Expected: PASS。

```powershell
git add requirements-train.txt configs/training/llamafactory/VERSION tests/unit/test_phase04_environment.py
git commit -m "build(train): pin phase04 training environment"
```

## Task 2：建立六 run 的分层训练配置

**Files:**
- Create: `configs/training/llamafactory/_base_sft.yaml`
- Create: `configs/training/llamafactory/_base_dpo.yaml`
- Create: `configs/training/llamafactory/sft/qwen3-0.6b-sft.yaml`
- Create: `configs/training/llamafactory/sft/qwen3-1.7b-sft.yaml`
- Create: `configs/training/llamafactory/dpo/qwen3-0.6b-dpo-b01.yaml`
- Create: `configs/training/llamafactory/dpo/qwen3-0.6b-dpo-b03.yaml`
- Create: `configs/training/llamafactory/dpo/qwen3-1.7b-dpo-b01.yaml`
- Create: `configs/training/llamafactory/dpo/qwen3-1.7b-dpo-b03.yaml`
- Create: `tests/unit/test_phase04_training_configs.py`

- [ ] **Step 1: 写参数化失败测试固定矩阵和单变量原则**

```python
import yaml
from pathlib import Path

RUNS = {
    "qwen3-0.6b-sft": ("sft", None),
    "qwen3-1.7b-sft": ("sft", None),
    "qwen3-0.6b-dpo-b01": ("dpo", 0.1),
    "qwen3-0.6b-dpo-b03": ("dpo", 0.3),
    "qwen3-1.7b-dpo-b01": ("dpo", 0.1),
    "qwen3-1.7b-dpo-b03": ("dpo", 0.3),
}

def load(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))

def test_phase04_matrix_has_exactly_six_overrides():
    actual = {p.stem for stage in ("sft", "dpo")
              for p in Path(f"configs/training/llamafactory/{stage}").glob("qwen3-*.yaml")}
    assert actual == set(RUNS)

def test_base_contracts():
    sft = load("configs/training/llamafactory/_base_sft.yaml")
    assert (sft["stage"], sft["finetuning_type"], sft["lora_rank"]) == ("sft", "lora", 16)
    assert (sft["learning_rate"], sft["num_train_epochs"]) == (1e-4, 3.0)
    assert sft["train_on_prompt"] is False and sft["mask_history"] is True
    assert sft["template"] == "qwen3" and sft["enable_thinking"] is False
    dpo = load("configs/training/llamafactory/_base_dpo.yaml")
    assert dpo["stage"] == "dpo" and dpo["pref_loss"] == "sigmoid"
```

- [ ] **Step 2: 运行并确认缺失配置**

Run: `uv run pytest tests/unit/test_phase04_training_configs.py -v`

Expected: FAIL，六份 phase04 override 不存在。

- [ ] **Step 3: 写 base 配置**

SFT base 固定 `dataset: phase03_sft_v0_1`、`dataset_dir: data/processed/v0.1`、`eval_dataset: phase03_sft_v0_1` 不可使用（SFT 验证文件尚未注册）；因此先在 `dataset_info.json` 新增 `phase03_sft_val_v0_1` 指向 `sft/v0.1/val.jsonl`，再将 `eval_dataset` 指向该 id。共同参数固定 bf16、batch size 2、gradient accumulation 8、logging steps 5、save/eval strategy `epoch`、load best model at end。

DPO base 固定 `dataset: phase03_dpo_v0_1`、`eval_dataset: phase03_dpo_val_v0_1`、`pref_loss: sigmoid`、bf16、batch size 1、gradient accumulation 8、learning rate `5e-6`、epochs `1.0`，并继承相同 qwen3/no-think/LoRA 合同。

- [ ] **Step 4: 写六份仅含差异项的 override**

```yaml
# sft/qwen3-1.7b-sft.yaml（0.6B 只替换模型名和 run_id）
run_id: qwen3-1.7b-sft
model_name_or_path: Qwen/Qwen3-1.7B
output_dir: models/adapters/qwen3-1.7b-sft
```

```yaml
# dpo/qwen3-1.7b-dpo-b01.yaml（其余三份只替换档位、beta、run_id）
run_id: qwen3-1.7b-dpo-b01
model_name_or_path: Qwen/Qwen3-1.7B
adapter_name_or_path: models/adapters/qwen3-1.7b-sft
pref_beta: 0.1
output_dir: models/adapters/qwen3-1.7b-dpo-b01
```

- [ ] **Step 5: 补全测试，断言 override 不含共享超参并运行**

Run: `uv run pytest tests/unit/test_phase04_training_configs.py tests/unit/test_dataset_contract.py -v`

Expected: PASS；六个 id 唯一，DPO adapter 均指向同档位 SFT，beta 恰为 0.1/0.3。

- [ ] **Step 6: 提交配置合同**

```powershell
git add configs/training/llamafactory data/processed/v0.1/dataset_info.json tests/unit/test_phase04_training_configs.py
git commit -m "feat(train): define phase04 sft dpo matrix"
```

## Task 3：实现确定性配置渲染器

**Files:**
- Create: `scripts/train/__init__.py`
- Create: `scripts/train/render_config.py`
- Create: `tests/unit/test_render_training_config.py`

- [ ] **Step 1: 写失败测试固定深合并、路径和校验行为**

```python
def test_render_sft_is_complete_and_deterministic(tmp_path):
    first = render_run("qwen3-0.6b-sft", output_root=tmp_path)
    second = render_run("qwen3-0.6b-sft", output_root=tmp_path)
    assert first.read_bytes() == second.read_bytes()
    cfg = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert cfg["stage"] == "sft"
    assert cfg["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert "run_id" not in cfg

def test_unknown_run_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="unknown run_id"):
        render_run("missing", output_root=tmp_path)
```

- [ ] **Step 2: 运行并确认模块缺失**

Run: `uv run pytest tests/unit/test_render_training_config.py -v`

Expected: FAIL，无法导入 `scripts.train.render_config`。

- [ ] **Step 3: 实现渲染 API 和 CLI**

```python
def render_run(run_id: str, output_root: Path = RENDERED_ROOT,
               overrides: dict[str, object] | None = None) -> Path:
    override_path = find_override(run_id)
    stage = override_path.parent.name
    base = load_yaml(CONFIG_ROOT / f"_base_{stage}.yaml")
    run = load_yaml(override_path)
    assert_required(run_id, stage, base | run)
    merged = deep_merge(base, run)
    merged.pop("run_id", None)
    if overrides:
        merged = deep_merge(merged, overrides)
    output = output_root / f"{run_id}.yaml"
    atomic_dump_yaml(output, merged)
    return output
```

CLI：`python -m scripts.train.render_config --run-id <id> [--output-root PATH] [--set key=value ...]`。`--set` 只允许空跑白名单：`max_steps/use_cpu/bf16/fp16/output_dir/overwrite_output_dir`，防止真训练时悄悄改矩阵。

- [ ] **Step 4: 运行单测和全矩阵渲染**

Run: `uv run pytest tests/unit/test_render_training_config.py tests/unit/test_phase04_training_configs.py -v`

Expected: PASS。

Run: `uv run python -m scripts.train.render_config --all`

Expected: 输出六个 `_rendered/<run_id>.yaml` 路径；重复执行文件 hash 不变。

- [ ] **Step 5: 提交渲染器**

```powershell
git add scripts/train tests/unit/test_render_training_config.py
git commit -m "feat(train): render validated llamafactory configs"
```

## Task 4：实现本地 CPU 两步空跑闸门

**Files:**
- Create: `scripts/train/dryrun.py`
- Create: `tests/unit/test_phase04_dryrun.py`
- Modify: `.gitignore`

- [ ] **Step 1: 写失败测试固定空跑覆盖项和依赖顺序**

```python
def test_build_dryrun_jobs_uses_06b_and_sft_before_dpo(tmp_path):
    jobs = build_dryrun_jobs(tmp_path)
    assert [j.run_id for j in jobs] == ["qwen3-0.6b-sft", "qwen3-0.6b-dpo-b01"]
    for job in jobs:
        cfg = yaml.safe_load(job.config.read_text(encoding="utf-8"))
        assert cfg["max_steps"] == 2 and cfg["use_cpu"] is True
        assert cfg["bf16"] is False and cfg["fp16"] is False
```

- [ ] **Step 2: 运行并确认失败**

Run: `uv run pytest tests/unit/test_phase04_dryrun.py -v`

Expected: FAIL，`dryrun.py` 不存在。

- [ ] **Step 3: 实现 `--prepare-only` 与真实执行模式**

真实执行对每个 job 调用：

```python
command = [sys.executable, "-m", "llamafactory.cli", "train", str(job.config)]
completed = subprocess.run(command, text=True, stdout=log, stderr=subprocess.STDOUT)
if completed.returncode != 0:
    raise DryRunError(f"{job.run_id} failed; see {job.log_path}")
assert (job.output_dir / "adapter_config.json").exists()
```

DPO dryrun 的 `adapter_name_or_path` 必须被覆盖到刚完成的临时 SFT adapter；运行前加载一条 SFT 和 DPO 数据并断言 assistant 输出不含 `<think>`/`</think>`，DPO chosen/rejected 均存在且不相等。

- [ ] **Step 4: 忽略纯生成目录并运行离线测试**

在 `.gitignore` 增加 `configs/training/llamafactory/_rendered/` 和 `.phase04-dryrun/`。

Run: `uv run pytest tests/unit/test_phase04_dryrun.py -v`

Expected: PASS。

- [ ] **Step 5: 安装锁定环境后执行真实空跑**

Run: `uv run python -m scripts.train.dryrun`

Expected: exit 0；SFT、DPO 各完成 2 step，各自存在 adapter 与日志。若当前机器内存不足，不把闸门标为通过；日志记录实际错误并在有足够资源的本地环境重跑。

- [ ] **Step 6: 提交空跑工具和实际记录**

把命令、版本、开始/结束时间、两个 adapter 检查结果和 no-think 检查写入 `project-log/phase-04-training/log.md`。

```powershell
git add .gitignore scripts/train/dryrun.py tests/unit/test_phase04_dryrun.py project-log/phase-04-training/log.md
git commit -m "feat(train): add phase04 cpu dry-run gate"
```

## Task 5：实现逐 run 原子回收与矩阵编排

**Files:**
- Create: `scripts/train/collect_artifacts.py`
- Create: `scripts/train/run_matrix.sh`
- Create: `tests/unit/test_collect_training_artifacts.py`
- Create: `tests/unit/test_run_matrix_script.py`

- [ ] **Step 1: 写失败测试固定回收清单**

```python
def test_collect_artifacts_creates_auditable_run(tmp_path, fake_training_output):
    result = collect_artifacts("qwen3-0.6b-sft", fake_training_output, tmp_path)
    assert (result / "adapter/adapter_config.json").exists()
    assert (result / "config.rendered.yaml").exists()
    assert (result / "requirements-train.txt").exists()
    assert (result / "trainer_log.jsonl").exists()
    manifest = json.loads((result / "manifest.json").read_text())
    assert manifest["run_id"] == "qwen3-0.6b-sft"
    assert manifest["status"] == "trained"
```

- [ ] **Step 2: 实现先写 staging、校验后 rename 的回收器**

回收器读取 `trainer_state.json` 的 `log_history` 写为一行一个事件的 `trainer_log.jsonl`；manifest 固定包含 run id、UTC 时间、git commit、base model、base revision、stage、adapter 相对路径、训练配置 SHA-256、requirements SHA-256。缺少 adapter/config/trainer state 任一项均返回非零，不覆盖已有完整 run。

- [ ] **Step 3: 写矩阵脚本和失败即停语义**

```bash
RUNS=(qwen3-0.6b-sft qwen3-1.7b-sft qwen3-0.6b-dpo-b01 qwen3-0.6b-dpo-b03 qwen3-1.7b-dpo-b01 qwen3-1.7b-dpo-b03)
for run_id in "${RUNS[@]}"; do
  uv run python -m scripts.train.render_config --run-id "$run_id"
  llamafactory-cli train "configs/training/llamafactory/_rendered/${run_id}.yaml"
  uv run python -m scripts.train.collect_artifacts --run-id "$run_id"
done
```

脚本头部使用 `set -euo pipefail`，支持 `--from-run` 断点续跑；已有 `manifest.status=trained` 的 run 必须显式 `--skip-complete` 才能跳过。

- [ ] **Step 4: 运行单元测试**

Run: `uv run pytest tests/unit/test_collect_training_artifacts.py tests/unit/test_run_matrix_script.py -v`

Expected: PASS；静态测试确认顺序准确、含 `set -euo pipefail`、训练和回收相邻。

- [ ] **Step 5: 提交云端编排工具**

```powershell
git add scripts/train/collect_artifacts.py scripts/train/run_matrix.sh tests/unit/test_collect_training_artifacts.py tests/unit/test_run_matrix_script.py
git commit -m "feat(train): orchestrate and collect phase04 matrix"
```

## Task 6：统一 phase04 评估产物，不改评分逻辑

**Files:**
- Modify: `scripts/eval/collect_analysis.py`
- Create: `scripts/eval/phase04_artifacts.py`
- Create: `configs/inference/phase04-qwen3-0.6b-sft.yaml`
- Create: `configs/inference/phase04-qwen3-1.7b-sft.yaml`
- Create: `configs/inference/phase04-qwen3-0.6b-dpo-b01.yaml`
- Create: `configs/inference/phase04-qwen3-0.6b-dpo-b03.yaml`
- Create: `configs/inference/phase04-qwen3-1.7b-dpo-b01.yaml`
- Create: `configs/inference/phase04-qwen3-1.7b-dpo-b03.yaml`
- Create: `tests/unit/test_phase04_eval_artifacts.py`

- [ ] **Step 1: 写失败测试固定 `predictions.jsonl` 与 `scorecard.json` 合同**

```python
def test_write_phase04_artifacts_has_effective_pass_and_slices(tmp_path, analysis_payload):
    write_phase04_artifacts(analysis_payload, tmp_path)
    rows = [json.loads(x) for x in (tmp_path / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 51
    assert set(rows[0]) >= {"id", "input", "expected", "model_output", "dimensions",
                            "effective_pass", "failure_reasons", "timing"}
    card = json.loads((tmp_path / "scorecard.json").read_text())
    assert card["effective_pass"]["denominator"] == 51
    assert set(card["scenario_slices"]) == {
        "confirmation", "missing_information", "multi_turn",
        "tool_call", "tool_result", "unrelated"
    }
```

- [ ] **Step 2: 实现纯转换层**

`effective_pass = protocol.score == 1.0 and task_correctness.score >= 0.95`。`failure_reasons` 只从已有评分结果派生：protocol 未过加入 `protocol`，task 小于 0.95 加入 `task_correctness`；不得重新解释模型输出。scorecard 同时写 aggregate dimensions、Final/ToolCall、六个场景切片、时延吞吐和评估环境元数据。

- [ ] **Step 3: 扩展 CLI 输出目录模式**

保留现有 `--out` 兼容；新增 `--run-id` 与 `--run-dir`，指定时先生成内存 payload，再调用 `write_phase04_artifacts`。运行目录还复制 `config.rendered.yaml`、requirements 快照、manifest，并把服务 stdout/stderr 路径记录为 `server.log`。

- [ ] **Step 4: 写六份推理配置**

每份使用现有 `llama_server` backend 合同、`base_url: http://127.0.0.1:8000/v1`、temperature 0、max_tokens 512、timeout 180；模型字段等于 run id。no-think 继续由 `LlamaServerBackend` 请求中的 `chat_template_kwargs.enable_thinking=false` 保证，测试显式断言该代码路径未被移除。

- [ ] **Step 5: 运行测试并提交**

Run: `uv run pytest tests/unit/test_phase04_eval_artifacts.py tests/unit/test_inference_backend.py tests/unit/test_scorecard.py -v`

Expected: PASS，且现有评分测试无变化。

```powershell
git add scripts/eval/collect_analysis.py scripts/eval/phase04_artifacts.py configs/inference/phase04-*.yaml tests/unit/test_phase04_eval_artifacts.py
git commit -m "feat(eval): write auditable phase04 run artifacts"
```

## Task 7：实现跨 run diff

**Files:**
- Create: `scripts/eval/diff_runs.py`
- Create: `tests/unit/test_diff_runs.py`

- [ ] **Step 1: 写失败测试固定翻正、翻负与切片净变化**

```python
def test_diff_classifies_flips_and_net_change(tmp_path, left_rows, right_rows):
    result = diff_predictions(left_rows, right_rows)
    assert result["flipped_positive"] == ["case-002"]
    assert result["flipped_negative"] == ["case-003"]
    assert result["net_effective_pass"] == 0
    assert result["scenario_delta"]["tool_call"] == pytest.approx(0.25)

def test_diff_rejects_mismatched_case_sets(left_rows, right_rows):
    with pytest.raises(ValueError, match="sample id sets differ"):
        diff_predictions(left_rows, right_rows[:-1])
```

- [ ] **Step 2: 实现纯离线比较与 CLI**

输入支持 run 目录或 `predictions.jsonl`。输出固定含左右 run id、51 条 id 集合 hash、effective pass 翻正/翻负/不变列表、protocol/task 均值变化、六场景 delta，以及每个变化样本的旧/新 output 和 failure reasons。

Run: `uv run python -m scripts.eval.diff_runs --left experiments/runs/phase04-qwen3-1.7b-sft --right experiments/runs/phase04-qwen3-1.7b-dpo-b01 --out reports/m2-dpo/diff-qwen3-1.7b-b01.json`

Expected: exit 0 并写 JSON；缺 run 时明确列路径并返回 2。

- [ ] **Step 3: 运行测试并提交**

Run: `uv run pytest tests/unit/test_diff_runs.py -v`

Expected: PASS。

```powershell
git add scripts/eval/diff_runs.py tests/unit/test_diff_runs.py
git commit -m "feat(eval): compare phase04 prediction runs"
```

## Task 8：实现预注册选型规则和回归红线

**Files:**
- Create: `scripts/eval/select_phase04.py`
- Create: `tests/unit/test_select_phase04.py`

- [ ] **Step 1: 写失败测试覆盖红线和三层 tie-break**

```python
def test_dpo_protocol_regression_over_two_points_is_ineligible():
    sft = card("sft", effective=25, task=0.80, protocol=0.90, params=1.7)
    dpo = card("dpo", effective=30, task=0.90, protocol=0.879, params=1.7, parent="sft")
    result = select_winner([sft, dpo])
    assert result["winner"] == "sft"
    assert result["runs"]["dpo"]["eligible"] is False

def test_one_case_tie_uses_task_then_smaller_model():
    result = select_winner([
        card("large", effective=26, task=0.81, protocol=0.95, params=1.7),
        card("small", effective=25, task=0.82, protocol=0.95, params=0.6),
    ])
    assert result["winner"] == "small"
```

- [ ] **Step 2: 实现规则**

先按每个档位检查 DPO 相对母版 protocol 的百分点差，下降 `> 0.02` 标为不合格；某档位两个 DPO 均不合格时 SFT 仍参与全局选型。候选按 effective pass 最大；最大值相差 ≤1 时比较 task correctness；仍相同选参数量更小；完全同档同分时用 run id 字典序只保证确定性并在理由中标注。

- [ ] **Step 3: 运行测试并提交**

Run: `uv run pytest tests/unit/test_select_phase04.py -v`

Expected: PASS。

```powershell
git add scripts/eval/select_phase04.py tests/unit/test_select_phase04.py
git commit -m "feat(eval): apply preregistered phase04 selection rules"
```

## Task 9：实现报告生成与全离线集成测试

**Files:**
- Create: `scripts/eval/render_phase04_reports.py`
- Create: `reports/m1-sft/README.md`
- Create: `reports/m2-dpo/README.md`
- Create: `tests/integration/test_phase04_pipeline.py`

- [ ] **Step 1: 写离线集成测试**

fixture 生成六个各含 51 条预测的 run 目录，其中一版 DPO 触发红线、一版胜出。测试依次调用 diff、select、render，断言：两份 README 均列出全部适用 run；failed run 显式显示；M2 报告含母版 diff；selection JSON 的 winner 与报告一致；不访问网络、不加载模型。

- [ ] **Step 2: 实现确定性 Markdown 生成器**

M1 表列 M0 三基线和两个 SFT；M2 表列两个 SFT 与四个 DPO。每行固定输出 status、protocol、task correctness、effective pass、六场景、mean/p95 latency、环境。正文按 selection JSON 原样解释红线和 tie-break，并从 diff 按净变化绝对值选前三个场景及最多五个典型翻正/翻负样本。

- [ ] **Step 3: 运行集成与回归测试**

Run: `uv run pytest tests/integration/test_phase04_pipeline.py tests/unit/test_diff_runs.py tests/unit/test_select_phase04.py -v`

Expected: PASS。

Run: `uv run pytest -m "not local_backend"`

Expected: PASS。

Run: `uv run ruff check .`

Expected: `All checks passed!`。

- [ ] **Step 4: 提交报告流水线**

```powershell
git add scripts/eval/render_phase04_reports.py reports/m1-sft/README.md reports/m2-dpo/README.md tests/integration/test_phase04_pipeline.py
git commit -m "feat(report): generate phase04 comparison reports"
```

## Task 10：执行云端六 run 训练与打包（评估已迁移本地）

**Files:**
- Modify: `project-log/phase-04-training/log.md`
- Create: `experiments/runs/phase04-<run_id>/...`（六份真实产物）
- Modify: `reports/m1-sft/README.md`
- Modify: `reports/m2-dpo/README.md`

- [ ] **Step 1: AutoDL 开机后核验环境和数据**

Run: `python -m pip install -r requirements-train.txt`

Run: `python -c "import torch,transformers,peft,trl; print(torch.cuda.get_device_name(0)); print(torch.__version__, transformers.__version__, peft.__version__, trl.__version__)"`

Expected: 4090 可见，版本与锁文件一致。

Run: `sha256sum data/eval/test.jsonl && cat data/eval/test.sha256`

Expected: hash 相同；不相同立即停止。

- [ ] **Step 2: 顺序执行六 run**

Run: `bash scripts/train/run_matrix.sh`

Expected: 六个 run 顺序完成；每个 run 训练结束后对应 manifest 和 adapter 已回收才开始下一项。任一失败立即停止，在 log 的 Open Issues 写 run id、命令、exit code 和日志路径；修复后用 `--from-run` 恢复，不删除失败记录。

- [ ] **Step 3: 在 AutoDL 校验并打包六个训练 run**

AutoDL 不启动评估服务。六组训练完成后执行：

Run: `python -m scripts.train.package_cloud_artifacts --runs-root experiments/runs --out /root/autodl-tmp/phase04-training-artifacts.zip`

Expected: 只有六个 run 的 manifest/config/requirements/trainer log/adapter 全部完整时才生成 zip。

- [ ] **Step 4: 下载并在 Windows 本地评估六个 run**

Run: `python -m scripts.eval.run_phase04_local --skip-complete`

Expected: 脚本顺序启停六个 LLaMA-Factory CPU API；每个目录有 51 行 `predictions.jsonl`、`scorecard.json` 和 `server.log`。

- [ ] **Step 5: 生成 diff、选型和报告**

上一步命令在六组评估完成后自动生成四份 diff、`reports/phase04-selection.json` 和 M1/M2 报告。

Expected: winner 唯一；failed/不合格/落选 run 均未从表中消失。

- [ ] **Step 6: 提交可审计文本与小型 JSON 产物**

执行前先按仓库策略确认 adapter 总体积；若单文件超过托管上限，使用 Git LFS 并在 README 记录对象 hash，不强行普通 git add。提交不得包含 base 或 merged fp16。

```powershell
git add project-log/phase-04-training/log.md reports/m1-sft reports/m2-dpo configs
git add -f experiments/runs/phase04-*/manifest.json experiments/runs/phase04-*/config.rendered.yaml experiments/runs/phase04-*/requirements-train.txt experiments/runs/phase04-*/trainer_log.jsonl experiments/runs/phase04-*/predictions.jsonl experiments/runs/phase04-*/scorecard.json
git commit -m "exp(train): record phase04 sft dpo matrix"
```

## Task 11：本地六 run 选型与获胜版本部署复评

**Files:**
- Create: `configs/training/llamafactory/export/phase04-winner-fp16.yaml`
- Create: `experiments/runs/phase04-winner-local-fp16/...`
- Modify: `reports/m2-dpo/README.md`
- Modify: `project-log/phase-04-training/log.md`

- [ ] **Step 1: 从 selection JSON 生成导出配置**

配置固定引用获胜 run 的 base model 和 adapter，`export_dir: models/merged/phase04-winner-fp16`、`export_size: 2`、`export_device: cpu`、`export_legacy_format: false`。不得手填另一个 run id。

- [ ] **Step 2: merge 并核验非量化权重**

Run: `llamafactory-cli export configs/training/llamafactory/export/phase04-winner-fp16.yaml`

Expected: merged 目录存在 config/tokenizer/safetensors；配置未含 quantization bit，权重 dtype 为 float16/bfloat16。merged 目录继续被 `.gitignore` 排除。

- [ ] **Step 3: 启动本地 fp16 服务并同口径复评**

Run: `uv run python scripts/eval/collect_analysis.py --backend-config configs/inference/phase04-winner-local-fp16.yaml --cases data/eval/test.jsonl --run-id phase04-winner-local-fp16 --run-dir experiments/runs/phase04-winner-local-fp16`

Expected: 51 条完整；评估集 hash 与 M0 和六组本地 phase04 相同。

- [ ] **Step 4: 转换 GGUF 并记录部署口径差异**

只对 winner 转换目标 GGUF 并用 llama.cpp 复评。用 `diff_runs.py` 比较本地 winner LoRA/fp16/GGUF，报告质量变化；只把 llama.cpp 时延与 M0 llama.cpp 时延比较。

- [ ] **Step 5: 最终验证与阶段收尾**

Run: `uv run pytest -m "not local_backend"`

Expected: PASS。

Run: `uv run ruff check .`

Expected: `All checks passed!`。

Run: `git status --short`

Expected: 仅允许用户原有的 `project-log/phase-03-dataset/log.html` 未跟踪项，以及明确记录但不入库的模型目录；无遗漏 phase04 文本/JSON 产物。

```powershell
git add configs/training/llamafactory/export/phase04-winner-fp16.yaml reports/m2-dpo/README.md project-log/phase-04-training/log.md
git add -f experiments/runs/phase04-winner-local-fp16/manifest.json experiments/runs/phase04-winner-local-fp16/predictions.jsonl experiments/runs/phase04-winner-local-fp16/scorecard.json
git commit -m "exp(eval): anchor phase04 winner with local fp16 evaluation"
```

## 自检结论

- 规格覆盖：六 run、版本锁、本地两步 SFT/DPO 空跑、逐 run 回收、云端打包、Windows CPU 六组评估、全量证据、diff、DPO protocol 红线、三层选型、冠军 llama.cpp 复评和 M1/M2 报告均有对应任务。
- 范围控制：没有修改冻结评估集或 scorer，没有加入量化/GGUF/部署调优，没有扩充 v0.1 训练数据。
- 接口一致：统一使用 `run_id` 关联配置、adapter、run 目录与报告；评估数据统一从现有 `collect_analysis.py` payload 派生，`effective_pass` 固定为 protocol 通过且 task correctness ≥0.95。
- 占位扫描：`<run_id>` 仅用于明确的六 run 循环命令模板，不代表未决实现；云端凭据、模型下载和真实训练结果属于执行期外部状态，失败分支和验收方式均已写明。
