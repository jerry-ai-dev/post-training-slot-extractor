# 阶段四设计：模型训练与实验矩阵（SFT → DPO，M1/M2）

> Status: Approved
> Date: 2026-07-30
> 对应 spec 章节：`finetune-spec.md` 3.4
> 前置：阶段一（推理/评估 harness）、阶段二（冻结评估集 + M0 baseline）、阶段三（SFT/DPO 数据集 v0.1）

---

## 1. 目标与范围

把 Qwen3 两档基座跑出 SFT 与 DPO 权重，用与 M0 完全同一把尺子回归对比，选出速度与效果最平衡的组合，并留下足以支撑后审与阶段六定向补数据的完整证据。

### 交付线（DoD）

- 全矩阵本地空跑记录，写入 `project-log/phase-04-training/log.md`；
- SFT LoRA 权重（M1）与 DPO LoRA 权重（M2），含训练日志与渲染配置；
- 每个 run 的逐条评估明细与聚合分数卡；
- `reports/m1-sft/`、`reports/m2-dpo/` 两份对比报告，含全部 run（含落选与失败）；
- 获胜版本的本地 fp16 复评，以及转换为 GGUF 后的 llama.cpp 部署口径复评，作为阶段五锚点。

### 不在范围内

- **不做** 六个候选版本的全量 merge/量化/部署调优。唯一例外是最终获胜版本：merge fp16 并转换为 GGUF，用 llama.cpp 做一次真实部署口径复评，作为阶段五的输入交接点。
- **不改** `data/eval/test.jsonl`、`test.sha256` 与任何评分逻辑。评分代码一旦改动，M0/M1/M2 即不可比。
- **不改** `src/slot_extractor/` 下的推理与评分核心。阶段四只新增 `scripts/train/` 编排脚本、`scripts/eval/diff_runs.py` 与 `configs/` 下的配置。
- **不扩充**训练数据集。v0.1（SFT 450/50、DPO 135/15）先跑通全链路；若效果碰顶，扩数据归入阶段六数据闭环。

---

## 2. 关键决策

| 决策 | 结论 | 理由 |
|---|---|---|
| 实验矩阵 | Qwen3-1.7B 为主力，Qwen3-0.6B 为速度档位对照 | M0 显示 4B 在 CPU 上 10s 延迟，远超 <1.5s 交付线，不具备落地价值；0.6B 提供速度上限参照 |
| 训练环境 | 本地 CPU 空跑闸门 → AutoDL 4090 真训练 | 低级错误在免费环境扫平，带卡时间只用于真训练 |
| SFT 方法 | LoRA（bf16） | 1.7B/0.6B 在 24G 上 bf16 LoRA 充裕，无需 NF4 量化带来的精度与速度损失；QLoRA 仅作显存不足退路 |
| 偏好对齐 | DPO，在 SFT adapter 上续训 | 压制幻觉与易混边界；续训而非叠加第二个 adapter，避免推理时双层加载 |
| 思考模式 | 统一 no-think | 固定 Schema 抽取任务不需长推理；thinking 使输出 token 翻倍，CPU 延迟严重超标 |
| 超参策略 | SFT 固定，DPO 扫 beta 两档（0.1 / 0.3） | beta 直接决定「压幻觉 vs 过度优化」的平衡，是本任务最值得花预算的旋钮 |
| 主验收指标 | `effective_pass` | 与 M0 口径一致；protocol 与 task_correctness 作诊断维度 |
| 评估路径 | Windows CPU 用 LLaMA-Factory 直接加载 base + LoRA，六组顺序评估 | 云端只训练；避免为六组候选都 merge/GGUF，保持同一本地评估口径 |

### M0 参照基线（`reports/baseline-m0/comparison.json`）

| 模型 | protocol | task_correctness | effective_pass | 平均时延 |
|---|---:|---:|---:|---:|
| Qwen3-0.6B | 39.2% | 37.7% | 2/51 | 3.68s |
| Qwen3-1.7B | 72.5% | 64.7% | 6/51 | 6.57s |
| Qwen3-4B-Instruct | 82.4% | 67.6% | 25/51 | 10.01s |
| GPT-5.6-sol（远端） | 100% | 98.8% | 51/51 | 3.45s |

**M1 目标**：1.7B SFT 版 `effective_pass` 追平或超过 4B baseline（25/51）。
**M2 目标**：DPO 版在 M1 基础上进一步提升，且不触发 protocol 回归红线。

---

## 3. 架构

阶段四是一条配置驱动的训练-评估流水线，四个环节通过文件系统契约衔接：

```
configs/training/llamafactory/{_base_*, sft/, dpo/}   ← 唯一真源（实验矩阵即配置文件集合）
        │
        ├─[本地 CPU]  闸门 1 空跑：0.6B + max_steps=2，验证配置/数据/模板可加载
        │
        └─[云端 4090] 闸门 2 真训练：SFT → models/adapters/<run_id>/
                          │                 DPO（基于对应 SFT）→ models/adapters/<run_id>/
                          ▼
                  [下载] 六组 adapter + manifest + 渲染配置 + 训练日志
                          │
                  [Windows CPU] 闸门 3：LLaMA-Factory base + adapter 顺序启停
                          │                复用 scripts/eval/collect_analysis.py
                          ▼
                  experiments/runs/phase04-<run_id>/  →  reports/m1-sft/、reports/m2-dpo/
                          │
                          ▼
                  [本地] 收尾：获胜版本 merge fp16 → GGUF → llama.cpp 复评 → 阶段五锚点
```

### 核心约定：run_id 即一切

每个 run 有唯一 `run_id`，它同时是训练配置文件名、adapter 目录名、推理配置名与结果目录名：

```
configs/training/llamafactory/sft/qwen3-1.7b-sft.yaml
models/adapters/qwen3-1.7b-sft/
configs/inference/phase04-qwen3-1.7b-sft.yaml
experiments/runs/phase04-qwen3-1.7b-sft/
```

任何一版结果都能由 id 反查到产生它的完整配置。可复现性由命名约定保证，不依赖记忆或事后追溯。

---

## 4. 实验矩阵

共 6 个真训练 run（空跑不计入）：

| run_id | 基座 | 阶段 | 差异参数 |
|---|---|---|---|
| `qwen3-0.6b-sft` | Qwen3-0.6B | SFT | — |
| `qwen3-1.7b-sft` | Qwen3-1.7B | SFT | — |
| `qwen3-0.6b-dpo-b01` | 0.6B + 其 SFT adapter | DPO | `pref_beta=0.1` |
| `qwen3-0.6b-dpo-b03` | 0.6B + 其 SFT adapter | DPO | `pref_beta=0.3` |
| `qwen3-1.7b-dpo-b01` | 1.7B + 其 SFT adapter | DPO | `pref_beta=0.1` |
| `qwen3-1.7b-dpo-b03` | 1.7B + 其 SFT adapter | DPO | `pref_beta=0.3` |

SFT 超参在两档基座间完全一致（LoRA rank 16、lr 1e-4、epoch 3），使「基座档位」成为唯一变量；DPO 同理，`pref_beta` 是唯一变量。每次对比都可归因到单一因素。

**执行顺序不可颠倒**：两个 SFT 必须先于依赖它们的 DPO 完成。

---

## 5. 配置结构

### 分层与渲染

六份配置若各自完整会重复约 90% 内容。采用 base + override 分层：

- `configs/training/llamafactory/_base_sft.yaml` — 共享 SFT 超参、LoRA 设置、`train_on_prompt: false`、`mask_history: true`、`template: qwen3`、no-think 设定、日志与保存策略
- `configs/training/llamafactory/_base_dpo.yaml` — 共享 DPO 超参、`pref_loss: sigmoid`
- `configs/training/llamafactory/sft/<run_id>.yaml` — 仅差异项：`model_name_or_path`、`output_dir`
- `configs/training/llamafactory/dpo/<run_id>.yaml` — 仅差异项：`model_name_or_path`、`adapter_name_or_path`（指向对应 SFT 产物）、`pref_beta`、`output_dir`

LLaMA-Factory 0.9.5 原生不支持 YAML 继承，因此由 `scripts/train/render_config.py` 在运行前合并为完整配置，写入 `configs/training/llamafactory/_rendered/<run_id>.yaml`，并归档一份到 run 目录。这保留了配置的可读性，同时保证喂给训练器的是一份完整、可原样重跑的文件。

### 数据集绑定

沿用 `data/processed/v0.1/dataset_info.json` 中已定义的三个数据集：`phase03_sft_v0_1`、`phase03_dpo_v0_1`、`phase03_dpo_val_v0_1`（sharegpt 格式，含 `tools` 列与 ranking 标记）。

### no-think 的三处一致性

no-think 必须在三个位置同时落实，否则训练与推理分布不匹配：

1. **数据侧** — Phase 3 产物已是纯 JSON 输出，无思考链；
2. **训练侧** — qwen3 模板并关闭 thinking；
3. **推理侧** — `configs/inference/phase04-<run_id>.yaml` 的 chat template 参数。

三处一致性是空跑闸门的一条显式检查项。

### 版本锁定

新增 `requirements-train.txt`，锁定 llamafactory（0.9.5）、transformers、peft、trl、torch 的精确版本。云端开机第一条命令即按它安装。本地与云端版本不一致是「本地空跑通过、云端报错」的最常见根因，因此版本清单是交付物而非附注。

---

## 6. 执行流程与门禁

四道闸门，任一不过即停。

### 闸门 0 — 环境与版本锁定（本地）

产出并验证 `requirements-train.txt`；确认 `configs/training/llamafactory/VERSION` 与之一致。

### 闸门 1 — 本地 CPU 空跑

`scripts/train/dryrun.py` 对每个 run_id 渲染配置、覆盖 `max_steps=2`、强制 CPU/fp32、输出到临时目录，依次跑通 SFT 与 DPO 两条链路。

验证目标不是效果，而是：
- 数据集能被 `dataset_info.json` 正确解析；
- qwen3 模板可套用，`tools` 字段不报错；
- DPO 的 chosen/rejected 列被正确识别；
- adapter 能落盘；
- no-think 三处一致。

空跑用 0.6B，全矩阵约十几分钟。**空跑记录写入 `project-log/phase-04-training/log.md`，属 DoD。**

### 闸门 2 — 云端真训练

`scripts/train/run_matrix.sh` 顺序执行：两个 SFT，再四个 DPO。

**每个 run 结束后立刻回收产物**（adapter + manifest + 训练日志 + 渲染配置 + 依赖快照）到 `experiments/runs/phase04-<run_id>/`。六组训练结束后将这六个目录打包下载，校验完整性后关闭 AutoDL。**云端不启动评估服务。**

### 闸门 3 — 评估与选型

本地 Windows 脚本从每个 run 的 manifest 读取 base 和 adapter，用 LLaMA-Factory CPU API 顺序启动六个 OpenAI 兼容服务。每次必须通过健康检查才调用 `scripts/eval/collect_analysis.py`；完成后可靠停止子进程，再进入下一 run。已有完整 `predictions.jsonl` 和 `scorecard.json` 的 run 可断点跳过。

六组候选的质量分数都来自同一本地后端，因此可用于横向选型。这些 LLaMA-Factory/PyTorch CPU 时延与 M0 llama.cpp 时延不可直接排名；M1/M2 报告必须显式标注后端差异。速度交付线只在获胜版本转为 GGUF 并由 llama.cpp 复评后判定。

#### 选型规则（事先定死，避免事后挑数字）

1. 主指标 `effective_pass`，取最高者；
2. 若差距 ≤ 1 例（统计噪声范围内），取 `task_correctness` 更高者；
3. 仍并列，取参数量更小者（速度优先，服务于 <1.5s 交付线）。

#### DPO 回归红线

DPO 版本相对其 SFT 母版，`protocol` 不得下降超过 **2 个百分点**。DPO 的已知失效模式是为压制幻觉而牺牲格式稳定性，此红线专门拦截。若某档位两个 beta 均触线，记录现象，并以该档位的 SFT 版本作为代表进入最终选型。

### 收尾 — 获胜版本本地同口径复评

仅对最终选出的一版：本地 merge fp16 并用 LLaMA-Factory 复评，随后转换为目标 GGUF，用 llama.cpp 在同一冻结评估集上复评。fp16 与 GGUF 的质量差是量化/转换损失，llama.cpp 时延才与 M0 同类可比。

### 失败处理

任一 run 训练崩溃或评估异常，**不静默跳过**：记入 `log.md` 的 Open Issues，并在最终对比表中标记为 `failed`，而非从表中消失。

---

## 7. 产物留存与可审计性

**原则：选型规则决定「选谁」，留存的证据决定「为什么」与「下一步补什么」。** 规则只输出一个 run_id，但阶段六的数据闭环、阶段七的报告，以及任何「为什么 DPO 在这档反而变差」的追问，依据全在留存记录里。因此每个 run 的产物都完整保留，不因落选而删除。

### 每个 run 目录 `experiments/runs/phase04-<run_id>/` 固定包含

| 内容 | 文件 | 用途 |
|---|---|---|
| 渲染后完整训练配置 | `config.rendered.yaml`、`requirements-train.txt` 快照、base 模型版本/commit | 原样重跑，无需回溯当时改了哪个文件 |
| 训练过程记录 | `trainer_log.jsonl`（loss 曲线、验证 loss；DPO 另含 reward margin 与 chosen-rejected accuracy） | 判断欠拟合/过拟合、beta 是否过紧 —— 只看最终分看不出 |
| 逐条评估明细 | `predictions.jsonl`：51 条样本各自的输入、模型原始输出、解析结果、各维度得分、失败原因标签 | **最关键**。用于回答「哪几条从错变对、哪几条从对变错」 |
| 聚合分数卡 | `scorecard.json`，结构与 M0 `comparison.json` 对齐（总分、Final/ToolCall 分项、六个场景切片、时延吞吐） | 可直接与 baseline 并表 |
| 原始服务日志 | `server.log` | 排查「分低是模型问题还是服务配置问题」 |

### 跨版本对比

新增 `scripts/eval/diff_runs.py`：输入两个 run_id，输出「哪些样本翻正、哪些翻负、按场景切片的净变化」。把「DPO 到底改善了什么」从模糊印象变成可写进报告的具体清单。**落选版本同样参与 diff** —— 更小的模型在哪些切片上反而更强，是选型结论的重要注脚。

### 最终汇总

`reports/m1-sft/README.md` 与 `reports/m2-dpo/README.md` 各含：

- M0 / M1 / M2 并列的完整对比表，**所有 run 都在表内，包括落选与 failed**；
- 选型规则的逐条应用过程与结论；
- 从 diff 中读出的定性观察：典型失败样例、DPO 收紧了哪类边界、还剩哪些系统性错误。

最后一项直接作为阶段六定向补数据的输入。

### 留存边界

- **入 git**：所有配置、训练日志、`predictions.jsonl`、`scorecard.json`、报告 —— 保证结论可追溯；
- **入 git**：adapter 权重（LoRA 体积小，6 版合计约数百 MB）；
- **不入 git**：基座模型、merge 后的 fp16 权重，由 `.gitignore` 排除，路径与获取方式写进 README。

---

## 8. 新增与改动文件清单

### 新增

```
requirements-train.txt
configs/training/llamafactory/_base_sft.yaml
configs/training/llamafactory/_base_dpo.yaml
configs/training/llamafactory/sft/qwen3-{0.6b,1.7b}-sft.yaml
configs/training/llamafactory/dpo/qwen3-{0.6b,1.7b}-dpo-b{01,03}.yaml
configs/inference/phase04-<run_id>.yaml            （6 份）
scripts/train/render_config.py
scripts/train/dryrun.py
scripts/train/run_matrix.sh
scripts/train/collect_artifacts.py
scripts/train/package_cloud_artifacts.py
scripts/eval/run_phase04_local.py
scripts/eval/diff_runs.py
reports/m1-sft/README.md
reports/m2-dpo/README.md
```

### 改动

```
project-log/phase-04-training/log.md    （工作日志、决策、产物、开放问题）
.gitignore                              （排除基座与 merged 权重）
```

### 不改动

```
data/eval/**                            冻结
src/slot_extractor/evaluation/**        评分逻辑冻结，保证 M0/M1/M2 可比
data/processed/**                       v0.1 数据集不变
```

---

## 9. 风险与缓解

| 风险 | 影响 | 缓解 |
|---|---|---|
| 训练数据仅 450 条，SFT 易过拟合 | M1 提升有限或验证 loss 上翘 | 保留 `trainer_log.jsonl` 供判断；若确认为数据量瓶颈，作为阶段六扩数据的量化依据，不在阶段四临时补数据 |
| LLaMA-Factory CPU 时延与 M0 llama.cpp 时延不同口径 | 速度结论被误读 | 六组只比质量，报告标注后端差异；获胜版本转 GGUF 后用 llama.cpp 判定真实部署时延 |
| 云端环境与本地版本不一致 | 空跑通过但云端报错，浪费带卡时间 | `requirements-train.txt` 精确锁版，开机第一步安装 |
| DPO 压幻觉牺牲格式稳定性 | protocol 回退 | 2 个百分点回归红线；触线则回退到 SFT 版本代表该档位 |
| 训练中途机器异常 | 丢失全部产物 | 每个 run 结束即刻回收产物，回收成功再启动下一个 |
| 0.6B 档位可能全面不达标 | 该档位无落地价值 | 仍保留全部结果与 diff，作为「小模型能力边界」的报告素材，不视为失败 |

---

## 10. 开放问题

- SFT 的 epoch 数在 450 条数据上是否需从 3 下调 —— 以闸门 1 后首个 SFT run 的验证 loss 曲线决定，属执行期调整，不阻塞设计。
- GGUF 的具体量化档位在获胜版本确定后按阶段五部署预算选定；不影响六组本地 fp16 LoRA 选型。
