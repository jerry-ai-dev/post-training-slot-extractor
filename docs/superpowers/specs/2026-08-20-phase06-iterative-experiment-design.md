# 阶段六设计：人工驱动的版本化迭代实验闭环

> Status: Approved
> Date: 2026-08-20
> 对应阶段：Phase 06 - Iteration and Data Loop
> 前置：阶段一至五的评估、数据、训练、量化和部署产物

---

## 1. 目标

阶段六建立一套轻量、文件化、可反复执行的实验方法论。它把阶段一至五已有的评估、数据构建、SFT/DPO 训练、模型合并、GGUF 量化和部署评测能力组织成多轮人工迭代闭环。

每轮允许分析多个问题、采用多个策略并设置多个对照变体。每轮必须清晰回答：

1. 分析的问题是什么；
2. 计划的解决策略是什么；
3. 最终结果是什么；
4. 结论是什么。

所有配置、中间参数、模型谱系、原始结果和人工结论必须可追踪、可复现，并能直接作为阶段七报告的证据。

阶段六不是自动调参系统。问题判断、策略审批、远端启动、结果解释、下一轮选择和阶段结束均由人工决定。

---

## 2. 核心原则

1. **人工决策驱动**：系统提供证据、校验和报告骨架，不替代人工判断。
2. **一轮可处理多个问题**：问题数量不受限；每个问题必须有独立证据、假设、策略关联和结论。
3. **策略可以变化**：数据、训练、Prompt、评估、评分、量化和推理均可成为某轮策略。
4. **端到端联动**：训练类变体原则上覆盖训练、合并、量化、质量评估和性能评估。
5. **完整模型矩阵**：适用的训练类变体默认覆盖 Qwen3-0.6B、Qwen3-1.7B 和 Qwen3-4B。
6. **远端迭代、本地分析**：每轮由人工在远端训练和全量评测，结果带回本地校验与分析；最终候选在 Windows CPU 上终验。
7. **版本演进而非覆盖**：训练集、诊断集、评估集、评分器、Prompt 和 schema 均可改变，但每次改变必须创建新版本。
8. **允许分支与回退**：新轮次可从任意历史轮次继承，不默认继承上一轮。
9. **如实限定归因**：允许组合策略；若缺少单变量对照，只能判断组合策略整体效果。
10. **失败也是结果**：失败、跳过、配置偏差和不完整矩阵均保留并解释，不静默删除。
11. **结束由人工决定**：系统不设自动停止条件；人工可继续、补实验、回退或进入终验。
12. **轻量事实源**：Git 中的 YAML、JSON、JSONL 和 Markdown 是事实源，不引入 MLflow、DVC 或实验服务。

---

## 3. 范围

### 3.1 包含

- 多轮实验的版本化目录、注册表和状态；
- Problem、Hypothesis、Strategy、Variant、Run、Result 和 Conclusion 的关联；
- 从任意历史轮次分支和继承；
- 0.6B、1.7B、4B 的 SFT/DPO 及其他适用变体；
- 数据、训练参数、Prompt、评估、评分、量化和推理策略；
- 远端实验包导出和结果包导入；
- 训练、合并、量化、完整质量测试和性能测试；
- 逐样本失败、修复和回归分析；
- 数据及评估合同版本管理；
- 远端跨轮比较和最终本地 Windows CPU 终验；
- M4 定版和阶段七所需跨轮汇总。

### 3.2 不包含

- 自动登录远端、自动提交训练任务或远端作业调度；
- 自动决定根因、策略、下一轮或停止时机；
- 强制每轮只研究一个问题或只改变一个变量；
- MLflow、DVC、数据库、任务队列或实验管理服务；
- 阶段七最终报告正文；
- 真实生产部署、灰度、回滚和线上流量闭环。

---

## 4. 总体工作流

```text
本地收集上一轮证据
        ↓
分析若干问题与原因假设
        ↓
制定综合策略、变体和模型矩阵
        ↓
人工确认问题分析与实验计划
        ↓
本地冻结并导出远端实验包
        ↓
人工上传并在远端执行
        ↓
远端训练 → 合并 → 量化 → 全量质量/性能评测
        ↓
人工下载结果包到本地
        ↓
本地完整性与谱系校验
        ↓
跨模型、跨轮、逐问题和逐样本分析
        ↓
形成分项结论与本轮综合结论
        ↓
人工决定继续、补实验、回退或进入本地终验
```

系统可以生成配置、命令、差分、汇总和报告骨架，但不得自动跨过人工审批点。

---

## 5. 概念模型

```text
Round
├── Problems
│   └── Hypotheses
├── Strategies
├── Variants
│   └── Runs
│       ├── Artifacts
│       ├── Evaluations
│       └── Results
└── Conclusions
```

核心追踪关系是：

```text
Problem ↔ Hypothesis ↔ Strategy ↔ Variant ↔ Run ↔ Result ↔ Conclusion
```

### 5.1 Round

一次从问题分析到人工决策的完整迭代，至少记录：

- `round_id`；
- `parent_round`；
- 从父轮次继承的具体内容；
- 本轮背景和目标；
- 问题、策略、变体和 run；
- 数据、评估、Prompt、schema 和 scorer 版本；
- 远端执行及本地导入信息；
- 分项结论、综合结论和人工决策。

新轮次可以从任意已关闭历史轮次分支。父轮次关系不得形成循环。

### 5.2 Problem 与 Hypothesis

每个问题项有稳定 ID，并记录：

- 现象；
- 指标、样本、日志或产物证据；
- 影响模型和场景；
- 严重度；
- 一个或多个原因假设；
- 尚不确定的事实。

主观观察可以成为待验证线索，但不能冒充已证实事实。

### 5.3 Strategy

策略描述计划如何解决一个或多个问题。允许类型包括但不限于：

- `data_addition`；
- `data_correction`；
- `data_rebalance`；
- `eval_revision`；
- `scorer_revision`；
- `sft_parameter`；
- `dpo_rebuild`；
- `prompt_revision`；
- `quantization`；
- `inference`；
- `control_or_ablation`；
- `other`。

每个策略记录修改项、保持不变项、预期效果、风险及其对应的问题。

### 5.4 Variant

Variant 是可执行的配置组合。每个变体注明：

- 对应的问题和策略；
- 相对父轮次的变更；
- 保持不变的控制项；
- 是否是对照；
- 适用模型及训练阶段；
- 需要该变体的原因；
- 变量耦合和归因限制。

一轮允许多个问题、多个策略和多个变体。关键判断是否增加消融或控制变体，由人工根据成本和证据需求决定。

### 5.5 Run

每个变体展开为实际模型运行。适用的训练类变体默认至少包含：

```text
Qwen3-0.6B
Qwen3-1.7B
Qwen3-4B
```

若本轮包含 SFT 和 DPO，则分别形成独立 run。每个 run 有唯一 ID，并独立保存训练、合并、量化、评估和谱系信息。

如果某轮仅研究评估集、评分器、Prompt、量化或推理参数，可引用历史模型而不重新训练，但必须生成本轮独立评测记录，并在计划中说明为何复用模型。

---

## 6. 轮次状态与人工门禁

```text
draft
  → approved
  → awaiting_remote_run
  → remote_running
  → awaiting_import
  → imported
  → analyzed
  → closed
```

- `draft`：本地完成问题分析、假设、策略和实验矩阵草案；
- `approved`：人工确认问题判断和计划；
- `awaiting_remote_run`：远端实验包已冻结并可上传；
- `remote_running`：人工已在远端启动；
- `awaiting_import`：远端结束，等待结果带回；
- `imported`：本地完成结果包校验；
- `analyzed`：完成结果分析和结论草案；
- `closed`：人工确认分项结论、综合结论和下一步决定。

未经人工确认不能从 `draft` 进入 `approved`，未经人工确认不能从 `analyzed` 进入 `closed`。系统不自动创建下一轮，也不自动结束阶段六。

轮次导入状态另分为：

- `valid`：可用于正式分析；
- `partial`：可探索分析，但不得宣称完整矩阵结果；
- `invalid`：配置、数据或谱系不匹配，需重跑或登记为偏差实验。

---

## 7. 目录与事实源

```text
experiments/phase06/
├── registry.yaml
├── baselines/
├── round-001/
│   ├── round.yaml
│   ├── problems.yaml
│   ├── strategies.yaml
│   ├── variants.yaml
│   ├── reports/
│   │   ├── problem-analysis.md
│   │   ├── strategy-plan.md
│   │   ├── result-analysis.md
│   │   └── conclusion.md
│   ├── package/
│   │   ├── manifest.json
│   │   ├── run-plan.yaml
│   │   ├── configs/
│   │   ├── commands/
│   │   └── checksums.sha256
│   ├── imported/
│   │   ├── import-manifest.json
│   │   └── runs/
│   ├── analysis/
│   │   ├── metrics.json
│   │   ├── case-diffs.jsonl
│   │   ├── regressions.jsonl
│   │   ├── failure-summary.json
│   │   └── comparisons/
│   └── artifacts.yaml
└── summary/
    ├── rounds.json
    ├── model-history.json
    ├── problem-history.json
    └── phase06-summary.md
```

Git 保存配置、manifest、日志、指标、逐样本结果、校验值和报告。大型 merged model、全量 checkpoint 和 GGUF 可以存放在 Git 外，但必须在 `artifacts.yaml` 中登记路径、SHA-256、生命周期和重建方法。

---

## 8. 每轮四份核心报告

### 8.1 `problem-analysis.md`

必须回答：

- 本轮分析哪些问题；
- 各问题的现象和证据是什么；
- 涉及哪些模型、场景、指标和样本；
- 原因假设是什么；
- 哪些事实仍不确定。

### 8.2 `strategy-plan.md`

必须回答：

- 每个问题计划如何处理；
- 修改数据、训练、Prompt、评估、量化或推理的哪些部分；
- 哪些内容保持不变；
- 设置哪些变体和对照；
- 各变体覆盖哪些模型；
- 预期结果、风险和归因边界；
- 人工审批结果。

### 8.3 `result-analysis.md`

必须回答：

- 各 run 是否成功；
- 0.6B、1.7B、4B 的完整质量和性能结果；
- SFT、DPO、量化和其他变体差异；
- 相对父轮次、历史最佳和本轮对照的变化；
- 修复案例、持续失败和新增回归；
- 每个问题和假设获得了什么证据；
- 结果缺失、偏差和异常。

### 8.4 `conclusion.md`

必须回答：

- 各问题的分项结论；
- 假设是被支持、部分支持、反驳还是证据不足；
- 策略是否有效及其副作用；
- 本轮综合结论；
- 保留哪些产物；
- 未解决问题；
- 下一轮建议从哪个历史轮次继承；
- 人工决定继续、补实验、回退或进入终验。

自动工具可以生成事实表格和报告骨架，但原因、归因和最终结论必须经人工确认。关闭后的报告不被后续轮次覆盖。

---

## 9. 数据、评估和合同版本演进

### 9.1 数据角色

每个数据版本必须明确样本角色：

- 训练集：用于 SFT、DPO 或其他训练；
- 开发/诊断集：用于问题分析和策略验证；
- 正式评估集：用于该评估合同下的正式比较。

现有 51 条样本定义为历史 `eval-v0.1`，永久保留，但不被视为永久最终评估集。后续可创建修正和扩充版本。

```text
data/
├── raw/v0.1, v0.2, ...
├── processed/v0.1, v0.2, ...
├── diagnostic/v0.1, v0.2, ...
└── eval/
    ├── v0.1/
    ├── v0.2/
    └── registry.yaml
```

### 9.2 版本规则

下列变化必须创建新版本，不得覆盖旧文件：

- 新增、删除或修正样本；
- 修改输入、expected output 或标签；
- 调整数据划分和场景分布；
- 修改评分器、阈值或语义匹配；
- 修改 Prompt、输出 schema 或工具 schema。

每个版本至少记录：父版本、变更原因、样本变更清单、分布、生成与审核过程、随机种子、SHA-256、分区重叠检查、首次使用轮次及兼容合同。

### 9.3 测试用例和评分错误

若发现测试用例或评分规则错误：

1. 保留原版本和原结果；
2. 把问题登记为本轮 Problem；
3. 创建新的 eval 或 scorer 版本；
4. 记录新增、删除、修正及原因；
5. 标明哪些旧结论受影响；
6. 在新版上重跑必要的历史基线和本轮矩阵；
7. 不回写已关闭轮次的原始结果。

评估体系修订可以成为独立轮次策略，该轮不强制重新训练模型。

### 9.4 跨评估版本比较

同一评估版本可以直接比较。不同版本不得只用总分证明进步，应同时提供：

- 共同样本子集比较；
- 新增、删除、修正样本的影响；
- 新版上的历史模型重测结果；
- 明确的 eval、scorer、Prompt 和 schema 版本。

评估集升级时，建议至少在新版重跑：

- 0.6B、1.7B、4B Base；
- 父轮次三种规模候选；
- 历史最佳候选；
- 本轮全部新变体。

重复候选去重。

### 9.5 数据回流和退役

历史评估样本可以在未来退役并转为训练素材，但必须记录：

- 从哪个评估版本退出；
- 退役时间和原因；
- 首次进入哪个训练版本；
- 哪些后续指标包含模型见过的样本；
- 哪些比较仍可作为泛化证据。

当前训练集与当前正式评估集必须执行内容、结构和语义近似重叠检查。若保留已见样本用于兼容性观察，必须单列指标，不得声称其为独立泛化结果。

### 9.6 评估结果身份

正式结果由以下组合唯一确定：

```text
model artifact
+ training dataset version
+ eval dataset version
+ scorer version
+ prompt/schema version
+ inference config
+ backend/environment
```

这用于区分模型改进、评估难度变化、评分修正、Prompt 变化、量化影响和后端差异。

---

## 10. 远端实验包

问题和策略经人工批准后，本地生成冻结实验包：

```text
round-001/package/
├── manifest.json
├── run-plan.yaml
├── configs/
│   ├── datasets/
│   ├── training/
│   ├── evaluation/
│   ├── quantization/
│   └── inference/
├── scripts/
├── commands/
│   ├── 01-prepare.sh
│   ├── 02-train.sh
│   ├── 03-merge-quantize.sh
│   ├── 04-evaluate.sh
│   ├── 05-package-results.sh
│   └── README.md
├── requirements-train.txt
├── source-commit.txt
└── checksums.sha256
```

`run-plan.yaml` 完整展开 variant × model × stage × quantization 矩阵。每个 run 使用唯一 ID，不能仅靠目录名推测身份。

冻结包必须记录：

- round 与父轮次；
- 问题、策略和变体引用；
- 数据、评估、Prompt、schema 和 scorer 版本；
- 基座模型及 revision；
- 父 adapter/checkpoint；
- rendered config 和 seed；
- 合并、量化、评估和性能测试参数；
- Git commit、工作区状态、依赖文件和所有输入校验值。

---

## 11. 远端人工执行与结果包

远端由人工上传实验包并启动。标准顺序是：

1. 校验实验包；
2. 检查数据、模型和依赖；
3. 保存环境快照；
4. 执行训练；
5. 保存最佳和最后 checkpoint；
6. 合并 adapter；
7. 量化 GGUF；
8. 运行完整质量评估；
9. 运行统一性能评估；
10. 打包结果。

每个 run 的状态为 `pending`、`running`、`succeeded`、`failed` 或 `skipped`。`skipped` 必须填写人工原因。

远端结果包结构：

```text
phase06-round-001-results/
├── result-manifest.json
├── environment/
├── runs/
│   └── <run-id>/
│       ├── status.json
│       ├── config.rendered.yaml
│       ├── trainer_log.jsonl
│       ├── metrics.json
│       ├── predictions.jsonl
│       ├── scorecard.json
│       ├── performance.json
│       ├── lineage.json
│       ├── hashes.json
│       └── logs/
├── selected-artifacts/
├── failures/
└── checksums.sha256
```

每个 run 至少记录：

- 基座、父 checkpoint/adapter 和完整 lineage；
- 数据与评估版本及哈希；
- rendered config、seed 和实际命令；
- Git commit 及工作区状态；
- Python、CUDA、PyTorch、Transformers、PEFT 和 LLaMA-Factory 版本；
- GPU、CPU、内存和操作系统；
- 开始时间、结束时间和耗时；
- 最佳及最后 checkpoint；
- adapter、merged model 和 GGUF 哈希；
- 合并与量化命令；
- backend、采样、线程和上下文参数；
- 质量及性能结果。

长期带回并保存：最终 adapter、入选 checkpoint、GGUF 候选、完整配置、日志、预测、指标和哈希。未入选 merged model 不要求下载，但必须能从基座 revision、adapter、代码、配置和命令重建。

---

## 12. 本地导入门禁

结果包导入后必须先检查：

- round、variant 和 run ID 是否匹配；
- 实验包 checksum 是否一致；
- 计划矩阵是否完成；
- 实际 rendered config 与审批版本是否一致；
- 数据和评估合同版本是否一致；
- 日志、预测、scorecard 和性能结果是否齐全；
- case 数量、ID 唯一性和样本集合是否正确；
- adapter、checkpoint 和 GGUF 校验值是否正确；
- 是否存在未登记参数修改、失败、跳过或重复 run。

远端临时修改不得被隐藏：

- 若只是保持等效语义的资源修复，例如调整 micro batch 并维持 effective batch，可由人工批准为计划内偏差；
- 若改变实验含义，必须登记为新变体；
- 原计划、实际参数和批准说明同时保留。

只有 `valid` 结果可用于完整正式结论；`partial` 必须清楚标记矩阵缺口；`invalid` 不进入正式对比，除非被重新登记为偏差实验。

---

## 13. 本地分析协议

### 13.1 三层比较

**模型内纵向比较**：

- 本轮与父轮次；
- 本轮与历史最佳；
- SFT 与 DPO；
- F16 与不同 GGUF 量化；
- 新策略与本轮对照。

**本轮横向比较**：

- 0.6B、1.7B、4B；
- 策略效果是否跨规模一致；
- 模型质量、体积、内存、延迟和吞吐代价。

**逐问题证据比较**：

- 相关指标与切片；
- 原失败是否修复；
- 是否产生新回归；
- 哪些变体支持或反驳哪些假设；
- 证据是否足够。

### 13.2 最低分析指标

- protocol；
- task correctness；
- effective pass；
- 场景切片；
- 错误类别；
- 修复、持续失败和新增回归；
- 输出长度及协议异常；
- training/eval loss；
- DPO reward 相关指标；
- latency、P50、P95、TTFT 和吞吐；
- GGUF 大小、内存和运行资源。

结果必须注明评估合同和运行环境。Phase 04 HF、Phase 05 本地 llama.cpp、Phase 06 远端和最终本地结果不能在环境不同的情况下被解释为纯模型或纯量化差异。

### 13.3 结论强度

每个问题或假设的结论状态为：

- `supported`；
- `partially_supported`；
- `refuted`；
- `inconclusive`。

组合策略缺少消融时，只能得出组合效果结论。冲突证据必须保留，不以单一总分掩盖不同模型或场景的回退。

---

## 14. 异常处理

### 14.1 训练或量化失败

保存失败阶段、命令、退出码、stdout/stderr、最后 checkpoint、资源状态和恢复说明。相同 checkpoint 与配置的恢复可归入原 run；参数变化则创建新 run。

### 14.2 矩阵不完整

例如 4B 因 OOM 未完成时，本轮标为 `partial`。已完成结果可以分析，但不得声称完成三模型比较。由人工决定补跑、调整后登记新变体，或接受缺口并说明。

### 14.3 文件损坏或结果缺失

校验失败的产物保留在异常区，不进入正式比较。不得通过手工补写汇总掩盖原始文件缺失。

### 14.4 指标异常

以下情况触发人工复核提示，但系统不自动否决：

- 总分大幅变化而逐样本差异无法解释；
- protocol 上升但 task correctness 明显下降；
- 样本数量或 ID 集合不一致；
- 输出异常趋同；
- 耗时或吞吐异常；
- 重复运行差异过大；
- loss 正常但任务性能明显下降；
- 远端与本地排序反转。

### 14.5 评估合同错误

保留旧结果，登记新 Problem，创建新版本并标注受影响结论；必要时重跑历史模型。不得修改关闭轮次的原始事实。

---

## 15. 最终本地终验与 M4

进入终验由人工确认，并记录：

- 为什么可以结束当前迭代；
- 选择哪些候选及其 round/variant/run；
- 为什么未选择其他候选；
- 已知问题；
- 本地终验要回答的问题。

终验原则上覆盖：

- 0.6B 最佳候选；
- 1.7B 最佳候选；
- 4B 最佳候选；
- 必要的 SFT/DPO 对照；
- 必要的 F16/量化对照；
- Phase 05 历史重要基线；
- 具有不同质量—资源权衡的候选。

本地采用固定 Windows CPU、llama.cpp、线程、context、采样、评估合同、预热和顺序执行协议，运行完整质量评估及 short/medium/2K/4K workload，记录逐样本预测、P50/P95、TTFT、吞吐、内存、文件体积和 server log。

若评估集已经升级，同时报告：

- 最新评估版本正式结果；
- 与旧版本的共同样本结果；
- 原 51 条 `eval-v0.1` 的兼容性结果；
- 评估合同变化说明。

远端用于指导趋势，本地目标部署环境用于 M4 定版。若排序不一致，检查 backend、硬件、线程、模板、量化工具和采样参数，保留两套结论并由人工决定；默认以本地部署结果为最终依据。

M4 可以同时给出轻量、平衡和质量优先候选，但必须指定一个默认候选。输出：

```text
reports/m4/
├── README.md
├── selection.json
├── quality-comparison.json
├── performance-comparison.json
├── failure-analysis.json
├── lineage.json
├── reproducibility.md
└── final-validation/
```

M4 必须说明模型规模、来源轮次、数据、训练方法、checkpoint、合并、量化、哈希、评估合同、远端及本地结果、相对 M0/M1/M2/M3 的变化、限制与重建步骤。

---

## 16. 框架验证

### 16.1 单元测试

验证：

- Round、Problem、Strategy、Variant、Run schema；
- ID 唯一性和引用关系；
- 父轮次存在且无循环；
- 版本和 artifact 解析；
- checksum；
- 状态转换；
- 指标及逐样本差分；
- 训练/评估重叠检测；
- lineage 完整性。

### 16.2 Mock 集成测试

使用小型 mock round 完整验证：

```text
建轮 → 多问题/多策略 → 人工批准标记 → 导出包
→ mock 三模型结果 → 导入 → 校验 → 差分
→ 生成四份报告骨架 → 人工结论 → 关闭轮次
```

### 16.3 远端空跑

正式矩阵前用小数据和最少 step 验证：配置加载、三个模型启动、训练、合并、量化、少量评估、结果打包和本地导入。空跑不形成质量结论。

### 16.4 复现验证

至少选择一个代表性 run，验证能从归档资产恢复训练或重建 adapter，重新合并、量化并使用相同评估合同评测。确定性步骤要求一致；非确定性训练记录允许波动范围。

### 16.5 关闭门禁

轮次关闭前必须确认：

- 问题和证据完整；
- 策略、变体和模型映射完整；
- 计划与实际偏差已记录；
- 所有 run 状态明确；
- 导入校验完成；
- 结果分析完成；
- 每个问题有结论状态；
- 未解决问题和下一步决定已填写；
- artifact 与 checksum 清单完整。

此门禁只防止漏记，不替代人工判断实验是否成功。

---

## 17. 阶段七接口

每轮关闭后更新跨轮汇总，但不提前撰写阶段七最终报告。汇总至少包括：

- 每轮问题、策略、结果和结论；
- 问题跨轮状态；
- 数据、评估和合同版本演进；
- 0.6B、1.7B、4B 的质量轨迹；
- SFT/DPO 变化；
- 量化和部署性能变化；
- 修复与回归案例；
- 无效策略和失败实验；
- 远端与本地差异；
- M4 最终谱系。

阶段七可以沿如下结构复用证据：

```text
为什么启动该轮
→ 分析了哪些问题
→ 为什么采用这些策略
→ 实际执行了什么
→ 得到了什么结果
→ 哪些假设成立、被否定或证据不足
→ 该结论如何影响下一轮
```

---

## 18. 完成定义

阶段六设计落实后的最小完成标准是：

1. 能建立、审批、导出、导入、分析和关闭一个多问题、多策略轮次；
2. 能从任意历史轮次分支并追踪完整继承关系；
3. 训练类变体能覆盖 0.6B、1.7B、4B 的适用完整链路；
4. 数据、评估、评分、Prompt 和 schema 能版本化演进且不覆盖历史；
5. 远端结果能通过本地完整性和谱系校验；
6. 每轮均有“问题分析—策略计划—结果分析—结论”四份记录；
7. 能生成跨模型、跨轮和逐样本比较；
8. 能重建至少一个代表性 run；
9. 最终候选完成本地 Windows CPU 终验并形成 M4；
10. 阶段七可直接使用跨轮索引和每轮报告，无需重新猜测实验过程。
