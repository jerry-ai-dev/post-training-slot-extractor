# Phase 04 - SFT, DPO, and Experiment Matrix

## 项目如何完成

Phase 04 仍然使用 Superpowers 的工作方式推进：先通过头脑风暴把目标、约束和关键选择讨论清楚，再形成设计文档；设计确认后，将方案拆成可以逐项验证的实施计划，最后按照计划完成代码、测试和真实训练。这里的重点不是让 AI 一次性“把训练做完”，而是先把整个项目变成一条有明确输入、输出和验收条件的流水线。

本阶段最初被整理为 **4 个大阶段、11 个实施任务**，后来因为评估方式从云端调整为本地，又增加了一份 **5 项任务的修订计划**：

1. **方案设计**：通过头脑风暴确定实验矩阵、训练方法、评估位置、显卡选择、选型规则和失败处理方式，并形成 Phase 04 design。
2. **AI 实施与本地验证**：AI 按实施计划编写训练配置、配置渲染器、dry-run、产物收集、评估、diff、选型和报告生成代码，同时用单元测试、集成测试与本地 CPU 空跑检查整条链路。最初计划中的 Task 1—9 主要属于这一部分。
3. **人工执行远端训练**：代码链路通过后，由人租用 AutoDL 显卡实例，核验环境与数据，顺序运行 2 组 SFT 和 4 组 DPO，共 6 个真实训练 run；训练结束后校验、打包并把 adapter、配置和日志下载回本地。这对应最初计划的 Task 10，也是 AI 无法代替人完成的外部操作节点。
4. **回传数据后继续评估与收尾**：真实训练产物回到本地后，再执行六组评估、版本差异分析、冠军选择、报告生成，以及只对冠军进行 merge、GGUF 转换和 llama.cpp 复评。这对应 Task 11。后来新增的 5 项本地评估修订任务，进一步明确了“AutoDL 只训练和打包，Windows 本地负责评估”的边界。

因此，这个阶段不是一条完全自动、从头跑到尾的流程，而是一次明确的人机接力：

```text
AI：头脑风暴、设计、拆计划、写代码、写测试、本地空跑
  ↓
人：租用远端 GPU、执行真实训练、处理平台操作、下载训练产物
  ↓
AI + 人：读取真实结果、运行本地评估、分析差异、选型并完成报告
```

### 在实施中与 AI 一起学习

无论是让 AI 编写代码，还是自己登录平台完成训练，都建议持续和 AI 讨论，而不是只复制命令、等待结果。AI 在这里不仅是代码生成工具，也可以作为全程的指导老师；即使实际操作由自己完成，也可以不断询问它每个选择背后的原因，并用自己的理解去判断建议是否合理。

例如，在头脑风暴中 AI 会询问评估放在远端还是本地。不要只选择一个答案，还应继续追问：两种方式的成本、速度、环境一致性和可复现性分别怎样？为什么本项目最终选择“远端只训练、本地统一评估”？同样，在租用远端平台时，可以让 AI 根据模型规模、LoRA 方式、精度和 batch size 估算显存，并解释为什么 RTX 4090 24GB 已经足够，而不是直接选择更昂贵的显卡。然后再结合平台价格、预计训练时间和显存余量，自己检查这个结论是否合理。

这种反复提问、理解理由、再做判断的过程，本身就是项目学习的一部分。最终获得的不应只有一套可以运行的代码和几个模型权重，还应包括对训练方案、算力选择、评估口径以及工程取舍的理解。

## 如何手动运行

### 1. 本地 CPU dry-run

在真正租用远端 GPU 之前，先在本机用 CPU 跑一次极小规模的空跑：

```powershell
python -m scripts.train.dryrun
```

这个命令只取一条 SFT 样本和一条 DPO 样本，构造一份最小数据集，把 `max_steps` 压到 2、关闭 bf16/fp16、用 CPU 后端，依次跑通一次 SFT 训练和一次基于该 SFT 产物的 DPO 训练，并检查每个 job 是否成功写出 `adapter_config.json`。

它要解决的问题是：训练配置渲染、数据格式、LoRA 目标模块、DPO 的 chosen/rejected 结构等，只要写错一处，在真正的 GPU 实例上跑起来才会报错——这时候卡是按时间计费的，调试编译错误、路径错误、字段错误纯粹是浪费真金白银。dry-run 用几十秒的 CPU 时间把这些低级错误提前暴露出来，确认“配置能跑通、代码链路没问题”之后，再去 AutoDL 上跑真正的六组训练，可以显著减少在云端排查问题的时间。

dry-run 通过之后，不代表训练效果没问题（毕竟只跑了 2 步、极小数据），但代表整条链路——配置渲染、LoRA 微调启动、DPO 启动、产物写出——是通的。

### 2. AutoDL 租卡训练

dry-run 通过后，再租用真实 GPU 跑六组正式训练（2 组 SFT + 4 组 DPO）。

1. 创建 AutoDL 实例：选择 RTX 4090 24GB、预装 PyTorch 2.x + CUDA 12.x 的镜像，数据盘至少 50GB。
2. 将 `feature/phase04-training` 分支上传或克隆到 `/root/autodl-tmp/` 数据盘。
3. AutoDL 镜像已经包含 CUDA 和 PyTorch，只需安装本项目特有的锁定依赖：

   ```bash
   cd /root/autodl-tmp/slot-extractor-finetune
   pip install -r requirements-train.txt
   ```

   AutoDL 公共镜像通常不会恰好预装本项目锁定版本的 LLaMA-Factory、Transformers、PEFT 和 TRL，因此仍需执行这一步，但不需要手工安装显卡驱动、CUDA 或基础 PyTorch 环境。

4. 检查 GPU：

   ```bash
   python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
   ```

   预期输出包含 `True` 和 `NVIDIA GeForce RTX 4090`。

5. 启动正式六组训练（AutoDL 到此只负责训练）：

   ```bash
   bash scripts/train/run_matrix.sh
   ```

6. 六组全部完成后校验并打包：

   ```bash
   python -m scripts.train.package_cloud_artifacts \
     --runs-root experiments/runs \
     --out /root/autodl-tmp/phase04-training-artifacts.zip
   ```

7. 从 AutoDL 下载 `phase04-training-artifacts.zip`。包内对每个 run 保留：LoRA adapter、`manifest.json`、渲染后训练配置、训练日志和依赖快照。打包成功后即可关闭 AutoDL，不需要在云端起评估服务——评估在下载回本地后统一完成。

### 本小节重点学习内容

#### 1. 整体工作流程

训练链路的核心是“基础配置 + 单 run 覆盖配置 → 渲染成最终配置 → 喂给 LLaMA-Factory 训练 → 收集产物”。SFT 和 DPO 走的是同一套渲染器，区别只在于用哪个 base 文件、以及 DPO 的 run 配置里多了 `adapter_name_or_path`（指向对应的 SFT 产物）和 `pref_beta`：

```text
[SFT 支路]                                   [DPO 支路]
_base_sft.yaml  ┐                                  _base_dpo.yaml  ┐
                ├─▶ deep_merge                                     ├─▶ deep_merge
sft/qwen3-0.6b-sft.yaml │  (render_config.py)       dpo/qwen3-*-dpo-b01.yaml │  (render_config.py)
sft/qwen3-1.7b-sft.yaml ┘                          dpo/qwen3-*-dpo-b03.yaml ┘
  └ run 专属: model_name_or_path / output_dir       └ run 专属: adapter_name_or_path(指向 SFT 产物) + pref_beta
                │                                                    │
                ▼                                                    ▼
   _rendered/<run_id>.yaml  ◀────────────── 真正喂给训练器的配置 ──────────────▶  _rendered/<run_id>.yaml
                │                                                    │
                ▼                                                    ▼
   llamafactory-cli train <rendered_config>          llamafactory-cli train <rendered_config>
   (读 data/processed/v0.1 下 sft train/val)          (读 data/processed/v0.1 下 dpo train/val)
                │                                                    │
                ▼                                                    ▼
   models/adapters/<run_id>/                          models/adapters/<run_id>/
   (adapter_model.safetensors, trainer_log.jsonl...)   (adapter_model.safetensors, trainer_log.jsonl...)
                │                                                    │
                ▼                                                    ▼
         collect_artifacts.py ───────────────────────────────────▶  collect_artifacts.py
                │                                                    │
                ▼                                                    ▼
   experiments/runs/phase04-<run_id>/                 experiments/runs/phase04-<run_id>/
   (adapter/, config.rendered.yaml, manifest.json)     (adapter/, config.rendered.yaml, manifest.json)
```

`run_matrix.sh` 负责把 SFT 支路和 DPO 支路串成一条有先后依赖的顺序：

```text
qwen3-0.6b-sft ──▶ qwen3-1.7b-sft ──▶ qwen3-0.6b-dpo-b01 / b03 ──▶ qwen3-1.7b-dpo-b01 / b03
                                       adapter_name_or_path =        adapter_name_or_path =
                                       models/adapters/qwen3-0.6b-sft models/adapters/qwen3-1.7b-sft
```

也就是说：SFT 体现在 `_base_sft.yaml` + `sft/*.yaml`，产出独立的 LoRA adapter；DPO 体现在 `_base_dpo.yaml` + `dpo/*.yaml`，并且必须先有对应规模的 SFT adapter 存在（DPO 是在 SFT 产物上继续偏好训练，而不是从 base 模型重新训练），这也是六组训练里 SFT 必须先于 DPO 跑完的原因。

评估阶段拿到的是 `experiments/runs/phase04-<run_id>/` 里的 `manifest.json` 和 `adapter/`，对每个 run 依次执行：

```text
experiments/runs/phase04-<run_id>/manifest.json
              │  build_api_config 读取 base_model + adapter_path
              ▼
   起本地 API 服务（HuggingFace 后端加载 base + LoRA）
              │
              ▼
   等待 /v1/models 健康检查通过
              │
              ▼
   跑 51 条冻结评估集 data/eval/test.jsonl
              │
              ▼
   写入 predictions.jsonl / scorecard.json / server.log
              │
              ▼
   关闭本地 API 服务 ──▶ 进入下一个 run
```

六组全部跑完后，再做收尾分析：

```text
                    （六组 predictions.jsonl / scorecard.json 都已就绪）
                                    │
              ┌─────────────────────┴─────────────────────┐
              ▼                                             ▼
     diff_runs.py                                   select_phase04.py
     （四组 SFT→DPO 样本级差异）                        （六组横向比较选出冠军）
              │                                             │
              ▼                                             ▼
 reports/phase04-diffs/*.json                    reports/phase04-selection.json
                                                             │
                                                             ▼
                                                 render_phase04_reports.py
                                                             │
                                                             ▼
                                    reports/m1-sft/README.md, reports/m2-dpo/README.md
```

这六组本地评估跑的是 CPU/HuggingFace 口径，只用于六组之间的横向比较；冠军选出来之后才会 merge 成 fp16、转 GGUF，再用 llama.cpp 重新评估一遍，那个数字才能和 M0 的 llama.cpp 基线直接比。

#### 2. 重点文件

训练参数配置（`configs/training/llamafactory/`）：

- `_base_sft.yaml` / `_base_dpo.yaml`：SFT、DPO 各自的共享超参数，例如 `lora_rank`、`learning_rate`、`num_train_epochs`、`pref_loss`（DPO 用 sigmoid loss）。
- `sft/<run_id>.yaml`：每个 SFT run 独有的部分，主要是 `model_name_or_path`（0.6B 还是 1.7B）和 `output_dir`。
- `dpo/<run_id>.yaml`：每个 DPO run 独有的部分，除了 model/output_dir，还有 `adapter_name_or_path`（指向对应规模的 SFT adapter 目录）和 `pref_beta`（b01 = 0.1，b03 = 0.3）。
- `_rendered/<run_id>.yaml`：`render_config.py` 把上面两层 `deep_merge` 之后落盘的最终配置，也是真正传给 `llamafactory-cli train` 的文件，出问题时应该看这份而不是原始 base/run 配置。

训练后产物（`experiments/runs/phase04-<run_id>/`，每个 run 一个目录）：

- `adapter/`：这个 run 真正要用的**模型权重**——`adapter_model.safetensors` 是 LoRA 权重增量，`adapter_config.json` 记录 rank、target modules 等元数据；下面的 `checkpoint-*/` 是训练中间存档，一般不需要用它，只用最外层这份最终权重。权重是 LoRA 增量而不是完整模型，必须和对应的 base 模型（`Qwen/Qwen3-0.6B` 或 `Qwen/Qwen3-1.7B`）一起加载才能用，例如：

  ```powershell
  llamafactory-cli chat `
    --model_name_or_path Qwen/Qwen3-1.7B `
    --adapter_name_or_path experiments/runs/phase04-qwen3-1.7b-sft/adapter `
    --template qwen3 --finetuning_type lora
  ```

- `manifest.json`：run 的元信息（`base_model`、`adapter_path` 等），评估脚本靠它找到该用哪个 base 模型和哪份 adapter，是训练产物和评估脚本之间的“索引文件”。
- `config.rendered.yaml`：这个 run 实际训练时使用的最终配置快照，出结果异常时优先回查这里而不是原始 yaml。
- `trainer_log.jsonl`：**训练日志**，逐 step 记录 loss 等指标，用来判断训练是否收敛、是否有异常波动。
- `requirements-train.txt`：训练当时的依赖快照，用于复现环境。
- `predictions.jsonl` / `scorecard.json` / `server.log`：不是训练产物，是本地评估阶段追加进来的——分别是模型在 51 条测试集上的逐条输出、汇总分数卡、评估服务的运行日志。

#### 3. 重点参数分析

一次训练里最值得花心思的不是把代码跑通，而是超参数怎么设计：哪些参数固定住作为对照，哪些参数是本次实验真正想验证的变量，以及每个数值背后为什么这么选。本次实验矩阵里，值得关注的参数和选择理由如下。

**哪些是"变量"，哪些是"控制项"**

六组 run 的差异被刻意压缩到最少几个变量，方便做归因：

- SFT 阶段唯一变量是**基座规模**（0.6B / 1.7B），其余超参数（`lora_rank`、`learning_rate`、`num_train_epochs` 等）在两档基座上完全一致，写在共享的 `_base_sft.yaml` 里，run 专属文件里只放 `model_name_or_path` 和 `output_dir`。这样两档 SFT 的效果差异就能完全归因到"模型大小"，而不会被超参数不一致干扰。
- DPO 阶段唯一变量是 **`pref_beta`**（0.1 / 0.3），同样固定在两档基座、两个 beta 共 4 组配置里，其余 DPO 超参数也全部收在共享的 `_base_dpo.yaml` 里。`pref_beta` 之所以被选为唯一要扫的旋钮，是因为它直接决定 DPO"压制幻觉 vs 过度优化、丢失原有能力"之间的平衡，是这个任务里最值得花预算反复试的参数；其余超参数则没有必要在 6 组里重复试错，先固定一版让流程跑通，效果不够再迭代。

**LoRA 相关参数**

- `finetuning_type: lora`：这里其实是两层独立的决策，跟显存够不够无关。第一层，选 LoRA 而不是全参数微调，是因为训练数据很小（SFT 450 条、DPO 135 条），全参数微调在这么小的数据集上很容易过拟合、甚至冲掉模型原有能力；而且这次要跑 6 组实验，LoRA adapter 体积远小于完整模型（六份合计几百 MB vs 六份完整权重），管理和下载成本低很多；DPO 还需要"在 SFT 基础上续训"，LoRA 天然支持这种衔接，全参微调没有对应的干净做法。第二层，在"用 LoRA"已经定下来之后，才轮到"用 bf16 LoRA 还是 QLoRA/NF4 量化 LoRA"——这一步才跟显存有关：24GB 对 0.6B/1.7B 做 bf16 LoRA 完全充裕，没必要为了进一步省显存去牺牲精度上 QLoRA（QLoRA 只是显存不够时的退路，这里不存在这个压力）。
- `lora_rank: 16`、`lora_alpha: 32`（alpha = 2×rank，是 LoRA 里常见的经验搭配）：rank 决定了新增参数量和可学习容量，16 是中等偏小的取值——对于本项目这种目标单一、数据量不大（SFT 450 条、DPO 135 条）的槽位抽取任务，没必要用更大的 rank 去追求更强表达力，rank 太大反而在小数据集上更容易过拟合。
- `lora_target: all`：正式训练对所有线性层（注意力 + MLP）都插入 LoRA，而不是只挑 `q_proj,v_proj`（那是 CPU dry-run 为了跑得快才做的简化，见上面"重点文件"里 dry-run 的配置）。真训练要看真实效果，所以放开到全部目标模块。

**学习率、epoch、batch size：SFT 和 DPO 明显不同**

- SFT：`learning_rate: 1e-4`、`num_train_epochs: 3`、`per_device_train_batch_size: 2` + `gradient_accumulation_steps: 8`（等效 batch size 16）。3 个 epoch、相对较高的学习率，目的是让模型在小数据集上充分学到目标 JSON 输出格式和抽取任务本身的能力。
- DPO：`learning_rate: 5e-6`（比 SFT 低了整整 20 倍）、`num_train_epochs: 1`、`per_device_train_batch_size: 1` + `gradient_accumulation_steps: 8`（等效 batch size 8）。DPO 是在已经训练好的 SFT 权重上做偏好微调，学习率必须远低于 SFT，否则很容易在很少的偏好对（135 条）上把模型已经学到的能力"冲掉"，导致格式跑偏、幻觉增多；只跑 1 个 epoch 也是同样的考虑——偏好数据量小，多轮很容易过拟合到这 135 条上。
- 两个阶段都用 `lr_scheduler_type: cosine` + `warmup_ratio: 0.1`：先热身再余弦衰减，是小数据量微调场景下比较稳的常规选择，避免训练一开始学习率就冲太猛。

**其他影响效果和可比性的关键开关**

- `enable_thinking: false`（no-think）：这是本任务特有的关键决定——槽位抽取是固定 Schema 的结构化输出任务，不需要长链路推理；如果开启 thinking，输出 token 数会翻倍，CPU 推理延迟会严重超出 <1.5s 的交付线，所以训练和推理口径统一关闭思考模式。
- `bf16: true` / `fp16: false`：用 bf16 而不是 fp16，是因为 bf16 数值范围更大、训练更稳定，现代 GPU（RTX 4090）原生支持，没有理由退回 fp16。
- `train_on_prompt: false` + `mask_history: true`（SFT）：只对助手回复部分计算 loss，不让模型去学习用户输入或历史对话的内容，这是 SFT 任务型微调的标准做法，避免模型"背" prompt 而不是学任务本身。
- `pref_loss: sigmoid`（DPO）：DPO 损失函数选最经典的 sigmoid 形式（也就是原始 DPO 论文的形式），没有引入更激进的变体，保持这一步方法本身简单可控，把预算留给 `pref_beta` 的对比。
- `metric_for_best_model: eval_loss` + `load_best_model_at_end: true`：训练过程中自动挑 eval loss 最低的 checkpoint 保存，而不是简单用最后一步的权重，避免训练后期过拟合的 checkpoint 被当成最终产物。
- 评估阶段 `select_phase04.py` 里的 **`protocol` 回归红线（2 个百分点）**：这不是训练超参数，而是选型规则里的一个关键参数——DPO 已知的失效模式是"为了压幻觉牺牲输出格式稳定性"，所以专门设了这条红线：DPO 版本的 `protocol` 分数只要比它的 SFT 母版低超过 2%，就直接判定不合格，不参与冠军竞争。这也是本次实验里 SFT 和 DPO 效果对比时首先要看的门槛，而不是只看 `effective_pass` 一个数字。

#### 4. 实验结果分析

六组候选均使用同一套 51 条冻结评估集完成本地评估。候选模型通过
LLaMA-Factory Hugging Face CPU 后端直接加载 base + LoRA；因此下表中的候选模型质量
指标可以相互比较，但其 CPU 时延不与 M0 的 llama.cpp 时延直接比较。

| 模型 | protocol | task correctness | effective pass |
|---|---:|---:|---:|
| Qwen3-0.6B 基础模型（M0） | 39.2% | 37.7% | 2/51 |
| Qwen3-0.6B SFT | 82.4% | 77.3% | 24/51 |
| Qwen3-0.6B DPO β=0.1 | 80.4% | 75.2% | 22/51 |
| Qwen3-0.6B DPO β=0.3 | 80.4% | 76.5% | 22/51 |
| Qwen3-1.7B 基础模型（M0） | 72.5% | 64.7% | 6/51 |
| Qwen3-1.7B SFT | 88.2% | 92.1% | 29/51 |
| Qwen3-1.7B DPO β=0.1 | 88.2% | 91.9% | 29/51 |
| Qwen3-1.7B DPO β=0.3 | 88.2% | 89.8% | 27/51 |

每个 run 训练时都开了 `plot_loss: true`，LLaMA-Factory 训练完会自动把曲线图存进对应 run 的 `experiments/runs/phase04-<run_id>/adapter/` 目录：6 组都有 `training_loss.png`（训练 loss）和 `training_eval_loss.png`（验证 loss），4 组 DPO 另外多一张 `training_rewards_accuracies.png`（chosen 相对 rejected 的判别准确率）。以冠军 `qwen3-1.7b-sft` 和其 DPO β=0.1 版本为例：

![qwen3-1.7b-sft training loss](../../experiments/runs/phase04-qwen3-1.7b-sft/adapter/training_loss.png)

*Qwen3-1.7B SFT 训练 loss：87 步内从 ~1.0 稳定降到 ~0.1 左右并收敛，没有震荡或反弹，说明 3 个 epoch、1e-4 学习率这组超参数对这份数据是合适的。*

![qwen3-1.7b-dpo-b01 rewards accuracy](../../experiments/runs/phase04-qwen3-1.7b-dpo-b01/adapter/training_rewards_accuracies.png)

*Qwen3-1.7B DPO β=0.1 的 chosen/rejected 判别准确率：只有 17 个训练 step，准确率先从 0.5 冲到 step 10 的 ~0.68，随后又滑回 step 15 的 ~0.5（等于随机猜）。这张图是"数据量/训练步数太少、没能稳定学到偏好"的直接证据，和下面结论里"DPO 未形成稳定收益"是同一件事的两种呈现方式——分数卡看到的是结果，这张图看到的是训练过程中就已经不稳。*

这种"先涨后跌"在 DPO 里是有名字的：**reward over-optimization / reward hacking（奖励过优化）**，DPO 原作者团队 2024 年的后续论文专门指出，这类直接对齐算法甚至会在**一个 epoch 还没跑完时**就开始退化；更细一层的机制叫 **likelihood displacement（似然位移）**，指 chosen 和 rejected 语义相近时，抬升 chosen 概率的梯度会"溢出"到其他相似回答上，导致判别力不增反降。可能原因：① 数据量和步数太小（135 条、17 步），梯度噪声主导，还没形成稳定规律；② `load_best_model_at_end` 只按 `eval_loss` 挑 checkpoint，没有针对 reward accuracy 做提前停止，退化后的权重也会被保留；③ `pref_beta=0.1` 相对这么小的数据集偏松，模型更容易越过"学到偏好"和"被少数样本带偏"之间的临界点。

其余 4 组的曲线图逻辑相同，需要时直接去对应 run 的 `adapter/` 目录里看即可。

结合 `trainer_log.jsonl`（训练是否收敛）和 `scorecard.json`/样本级 diff（效果是否提升），可以得出以下初步结论：

- SFT 效果显著。0.6B 从 2/51 提升至 24/51，1.7B 从 6/51 提升至 29/51。
- 本轮 DPO 未带来进一步提升。0.6B 两个 DPO 版本均比 SFT 少通过 2 条；1.7B
  β=0.1 与 SFT 同为 29/51，但任务正确率略低；β=0.3 少通过 2 条。
- 样本级 diff 显示，0.6B DPO 各翻正 1 条、翻负 3 条；1.7B β=0.1 没有翻正或
  翻负，β=0.3 翻负 2 条，未形成稳定的场景收益。
- 第一轮冠军为 `qwen3-1.7b-sft`。当前结果说明 SFT 已学到主要任务能力，而首版
  DPO 数据（135 train / 15 val、17 个训练 step）没有对 SFT 的真实残余错误形成有效增益。
- 后续若重训 DPO，应优先根据 SFT 的真实失败输出重建偏好数据、扩大偏好对和验证集，
  再尝试更低学习率与较小 β；不建议只在原数据上增加 epoch 或提高 β。

以上关于 DPO 为什么没提升的分析（reward over-optimization / likelihood displacement 等）只做到"定位现象、给出初步猜测"为止，不在 phase-04 阶段深入下去。针对失败案例做系统性的根因分析、定向补数据、重训和反复迭代评估，是 `project-log/phase-06-iteration/` 的目标（见该阶段 `log.md` 的 Goal：「Use scorecard failures to drive targeted data repair, retraining, and repeated evaluation」），phase-04 到"选出第一轮冠军、留下足够的证据（分数卡、diff、训练曲线）"就算完成任务。


