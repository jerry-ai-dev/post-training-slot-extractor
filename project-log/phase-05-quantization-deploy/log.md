# Phase 05 - 模型量化与本地部署

## Goal

将 Phase 04 产出的 LoRA adapters 与对应的 Qwen3 基座模型合并，转换并量化为适合本地 CPU 推理的 GGUF 模型；随后使用统一评估集验证量化前后的质量、时延和吞吐差异，并提供一个可交互的双模型对比 App，便于直观比较不同模型的工具调用与回复效果。

## 开发方法

和之前阶段一致：先使用 Superpowers 编写设计文档和实施计划，再按照计划完成代码实现、测试与验证。相关设计和计划位于 `docs/superpowers/specs/` 与 `docs/superpowers/plans/`。

## 运行方法

以下命令均在项目根目录执行。模型清单、基座模型、adapter 与产物路径统一定义在 `configs/quantization/phase05.yaml`；最终 adapters 位于 `models/adapters/`，构建代码位于 `scripts/quantize/`。

### 1. 合并 adapters 得到模型

构建脚本会根据模型注册表找到对应的 Qwen3 基座模型和 Phase 04 adapter，使用 PEFT 的 `merge_and_unload()` 完成合并，并将中间模型写入 `models/merged/`。通常无需单独执行合并命令，直接运行完整构建入口即可：

```powershell
uv run python -m scripts.quantize.build_phase05_real --model-id qwen3-1.7b-sft-q4-k-m
```

如需构建注册表中的全部量化目标，省略 `--model-id`。具体合并实现见 `scripts/quantize/build_phase05_real.py`。

### 2. 量化模型

同一构建入口会继续将合并后的 Hugging Face 模型转换为 F16 GGUF，使用 `data/calibration/phase05-v1.txt` 生成 imatrix，再通过 llama.cpp 量化为 `Q4_K_M`：

```powershell
uv run python -m scripts.quantize.build_phase05_real
```

量化结果位于 `models/gguf/`，F16 anchor、中间文件和 manifest 分别位于 `models/merged/` 与 `models/quantization/`。llama.cpp 的转换、imatrix 和量化工具路径见 `configs/quantization/phase05.yaml`。

### 3. 评估模型

本地评估入口会按顺序启动 llama-server，对注册表中的 Q4 模型和 F16 anchors 运行冻结质量集及 short、medium、2K、4K 性能负载：

```powershell
uv run python -m scripts.eval.run_phase05_local --config configs/evaluation/phase05.yaml --skip-complete
```

去掉 `--skip-complete` 可重新执行已有结果的模型；评估输出写入 `reports/phase05/`。评估实现位于 `scripts/eval/run_phase05_local.py`，评估配置位于 `configs/evaluation/phase05.yaml`。

### 4. 运行双模型对比 App

先确保 `deployment/llama_cpp/bin/llama-server.exe` 存在，并已按前述步骤生成要比较的 GGUF 模型。App 会读取注册表，并在选择模型时自动启动和管理左右两侧的 llama-server，无需手动分别部署：

```powershell
uv run python -m uvicorn slot_extractor.tool_loop.app:create_app --factory --host 127.0.0.1 --port 8000
```

随后打开 `http://127.0.0.1:8000`。App 配置位于 `configs/tool_loop/phase05.yaml`，后端实现位于 `src/slot_extractor/tool_loop/app.py`；如果 8000 端口被占用，可将命令中的端口改为 8001。

### 阶段五完整流程与技术架构

#### 1. 模型合并与量化

```text
Qwen3 Base（0.6B / 1.7B）
        │
        ├── Base 实验：跳过 Adapter 合并
        │
        └── SFT / DPO 实验 + 对应 LoRA Adapter
                       │
                       ▼
              PEFT merge_and_unload()
              PyTorch float16 · low_cpu_mem_usage
                       │
                       ▼
              Hugging Face 合并模型
                       │
                       ▼
       llama.cpp convert_hf_to_gguf.py --outtype f16
                       │
                       ▼
                   F16 GGUF
                       │
          ┌────────────┴────────────┐
          │                         │
          ▼                         ▼
  保留 2 个 SFT F16 Anchor     llama-imatrix
  用于观察后端差异            校准文本：phase05-v1.txt
                              threads=8 · ctx=512 · batch=128
                                        │
                                        ▼
                              llama-quantize --imatrix
                              方法：Q4_K_M · threads=8
                                        │
                                        ▼
                              8 个 Q4_K_M GGUF 模型
```

`Q4_K_M` 属于 llama.cpp 的 K-Quant 4-bit 混合精度量化方法。每个模型使用同一份领域校准文本，但分别生成自己的 imatrix。最终产物写入 Manifest，并记录 Base revision、Adapter、构建命令和 SHA256。

#### 2. 本地评估

```text
8 个 Q4_K_M 模型 + 2 个 SFT F16 Anchor
                    │
                    ▼
       llama-server（Windows CPU · 8 threads）
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  冻结质量集             性能负载
  51 cases / 模型        short / medium / 2K / 4K
  temperature=0          1 次 warmup + 5 次 hot run
          │                   │
          ▼                   ▼
  协议遵循、任务正确性    时延、P95、TTFT、吞吐、内存、体积
          └─────────┬─────────┘
                    ▼
             reports/phase05/
```

#### 3. 双模型对比 App

```text
浏览器双栏界面
      │ 选择左右模型、发送预约请求
      ▼
Uvicorn + FastAPI
      │
      ▼
LlamaServerManager
      │ 自动启动、健康检查、复用和停止
      ▼
左右两个 llama-server
      │
      ▼
独立 ConversationOrchestrator
      │ 模型 → find_technicians Mock 工具 → 最终回复
      ▼
状态、Tool Trace 和回复返回双栏界面
```

网页与 API 由同一个 FastAPI 应用提供；用户选择模型后，应用才通过 `LlamaServerManager` 启动对应的推理后端，因此无需提前手动部署两个 llama-server。

### 本小节重点理论知识

本小节不要求推导底层公式，但需要建立“Adapter 如何变成完整模型、完整模型如何变成可在 CPU 上运行的量化模型”这一条完整认识。

#### 1. 模型合并

- **主要工具库**：Hugging Face `Transformers` 负责加载基座模型，`PEFT` 负责加载 LoRA Adapter，PyTorch 提供张量计算和数据类型支持；本项目使用 `PeftModel.from_pretrained()` 和 `merge_and_unload()`。
- **LoRA Adapter 的含义**：Adapter 不保存完整模型，只保存训练得到的低秩增量参数；使用时必须与训练时对应的 Base Model、模型结构和 revision 匹配。
- **合并原理**：LoRA 将权重更新表示为低秩矩阵乘积，合并时把这部分增量写回基座权重，得到不再依赖 PEFT Adapter 的完整模型权重。
- **大致方法**：加载 Base Model → 挂载 Adapter → 执行 `merge_and_unload()` → 保存合并模型与 tokenizer → 进行推理一致性检查。
- **需要关注的问题**：Base 与 Adapter 是否配套、加载和合并时的 dtype、峰值内存、权重分片、SFT/DPO Adapter 的正确 lineage，以及合并前后输出是否一致。

#### 2. 模型量化

- **量化的含义**：把 F16/F32 权重压缩为更低位宽表示，以减少模型体积和内存带宽，并提高 CPU 推理速度；代价是引入近似误差，可能影响输出质量。
- **GGUF**：llama.cpp 使用的模型文件格式，除权重外还保存模型结构、tokenizer 和运行所需元数据。GGUF 是文件格式，`Q4_K_M` 是其中权重采用的量化类型，两者不是同一概念。

量化名称中混合了三个维度：**位宽**、**算法家族**和**家族内部配置**。它们的对应关系如下：

```text
llama.cpp 量化类型
│
├── 传统量化家族
│   ├── Q4_0、Q4_1
│   └── Q8_0
│
├── K-Quant 家族
│   ├── Q2_K
│   ├── Q3_K_S / Q3_K_M / Q3_K_L
│   ├── Q4_K_S / Q4_K_M       ← 本项目选择
│   ├── Q5_K_S / Q5_K_M
│   └── Q6_K
│
└── IQ 家族
    ├── IQ1_S / IQ1_M
    ├── IQ2_XXS / IQ2_XS / IQ2_S / IQ2_M
    ├── IQ3_XXS / IQ3_XS / IQ3_S / IQ3_M
    └── IQ4_XS / IQ4_NL
```

因此，**IQ 与 K-Quant 是并列的两个量化家族，不是上下级关系**。同一个目标位宽可以存在不同家族的实现，例如 `Q4_K_M` 和 `IQ4_XS` 都属于约 4-bit 量化，但使用的表示方法不同。

| 名称部分 | 表示什么 | 示例 |
|---|---|---|
| `Q4`、`Q5`、`IQ2`、`IQ3` | 大致目标位宽；数字越小通常体积越小、质量风险越高 | `Q4_K_M`、`IQ2_XS` |
| `_K` | 使用 K-Quant 的 K-block 分块量化方案 | `Q4_K_M` |
| `IQ` | 使用 IQ 家族的码本、非线性表示及重要性感知方法 | `IQ3_M`、`IQ4_NL` |
| `_S/_M/_L` | 同一家族内部的 Small、Medium、Large 取舍 | `Q3_K_S`、`Q3_K_M`、`Q3_K_L` |
| `_XS/_XXS` | 更小体积的配置 | `IQ2_XS`、`IQ3_XXS` |
| `_NL` | Non-Linear，采用非线性量化表示 | `IQ4_NL` |

- **传统 Q 家族**：如 `Q4_0`、`Q4_1`，结构较简单，是较早的量化方案。
- **K-Quant 家族**：使用 K-block 分块和混合精度策略，在不同张量间分配不同精度，通常具有较成熟的速度、兼容性和质量表现。
- **IQ 家族**：更侧重极低位宽下的表示效率，常使用重要性感知、非线性量化或更复杂的码本；压缩潜力更高，但对 llama.cpp 版本、硬件算子和校准数据更敏感。
- **`Q4_K_M` 的完整含义**：主体权重约为 4-bit，属于 K-Quant 家族，采用 Medium 配置；部分重要张量会使用更高精度，因此它不是“所有参数都严格占 4 bit”。
- **imatrix 不属于任何量化家族**：它是量化前生成的校准统计，可以辅助量化器判断哪些部分更重要。K-Quant 和 IQ 类型都可能使用 imatrix，但是否必需、如何使用取决于具体量化类型与 llama.cpp 版本。
- **本阶段为何不选 IQ**：阶段五首先需要建立稳定、成熟、容易复现的 CPU 量化基线，因此统一采用 `Q4_K_M`。IQ2、IQ3、IQ4 可作为后续扩展矩阵，用来研究更激进压缩下的质量、速度和兼容性变化。
- **选择方法**：不能只按位宽判断，应在同一评估集、Prompt、采样参数和推理后端下，同时比较质量、体积、内存、时延和吞吐。

> **扩展：量化远不止选择一个 Q4 或 Q8 档位。** 完整的量化体系还可以从多个维度理解：按发生时间可分为训练后量化（PTQ）和量化感知训练（QAT）；按量化对象可分为仅权重量化、权重与激活量化、KV Cache 量化；按数据依赖可分为无校准量化、校准量化和重要性感知量化；具体算法生态还包括 llama.cpp 的传统 Q/K-Quant/IQ、GPTQ、AWQ、SmoothQuant、bitsandbytes 等。不同方案服务于不同硬件、精度和吞吐目标，不能只比较名称或位宽。本项目采用的是一条明确的子路径：**训练后量化（PTQ）→ 仅权重量化（Weight-only）→ 使用领域数据生成 imatrix 校准 → llama.cpp K-Quant → Q4_K_M → GGUF/CPU 部署**。本阶段只建立这条稳定基线，其余方向可作为后续量化实验的扩展入口。

#### 3. imatrix

- **基本含义**：imatrix 是 Importance Matrix，即重要性矩阵。llama.cpp 用校准文本运行模型，收集中间激活统计，用来估计不同权重或通道对模型输出的重要程度。
- **作用**：量化器利用 imatrix 在有限位宽下更谨慎地处理重要部分，从而降低量化误差；它是量化过程的辅助校准信息，不是模型权重，也不是训练产生的 Adapter。
- **生成方式**：准备具有代表性的领域文本 → 使用目标 F16 GGUF 运行 `llama-imatrix` → 得到该模型对应的 imatrix 文件 → 通过 `llama-quantize --imatrix` 参与量化。
- **需要关注的问题**：校准文本应贴近真实输入分布但不能混入冻结评估集；不同模型应分别生成 imatrix；校准文本版本、参数和哈希需要记录，保证结果可追溯。
- **本项目参数**：领域文本为 `data/calibration/phase05-v1.txt`，`threads=8`、`context=512`、`batch=128`，最终通过 `llama-quantize` 生成 `Q4_K_M` 模型。

## 本地评估结果

完整的本地实验矩阵运行约 2 小时 33 分钟，覆盖 8 个 `Q4_K_M` 模型和 2 个 SFT F16 对照模型。每个模型均运行 51 条冻结质量用例，以及 short、medium、2K、4K 四档性能负载；每档负载包含 1 次冷启动请求、1 次预热请求和 5 次热运行请求。

### 评估指标与计算口径

本阶段复用阶段二的冻结评估集、Prompt 组装方式和确定性评分器，因此质量指标与阶段二完全同口径：

1. **协议遵循 `protocol`**：检查原始输出是否为可解析 JSON，`action` 是否合法，以及 Tool Call / Final 的字段名、字段数量、字段类型和枚举值是否符合 Schema。每条样本完全合法得 1 分，否则得 0 分；表中为 51 条样本的平均分。
2. **任务正确性 `task_correctness`**：Tool Call 检查动作、工具名和参数；Final 检查结构化状态、`reply_type`、必要/禁止语义动作及回复与工具事实的一致性。Final 的结构化结果占 70%，自然语言回复占 30%；表中为 51 条样本的平均分。
3. **有效通过率**：单条样本必须同时满足“协议检查通过”且“任务正确性不低于 0.95”。报告使用 `通过条数/51`，与阶段二定义一致；任务分严格等于 1 的数量不作为主指标。
4. **时延与吞吐**：直接报告质量集的总时延均值、P95、平均首字时延和平均生成吞吐，不把速度折算成质量分。

多轮、工具调用、工具结果、确认、无关输入和信息缺失仍作为场景切片保存在逐样本日志中，不重复构造一级总分。具体字段、回复和工具错误也继续保留在每条记录的错误标签中。

### 评测结果和结论

以下结果由现有 `reports/phase05/<model-id>/quality.json` 中的 510 条真实逐样本记录重新聚合得到，没有重新运行模型推理。

| 模型 | 协议遵循 | 任务正确性 | 有效通过率 | 平均时延 | P95 | 平均首字时延 | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b-base-q4-k-m | 68.6% | 43.2% | 2/51 | 5.09s | 6.32s | 2.70s | 18.0 tok/s |
| qwen3-0.6b-sft-q4-k-m | 78.4% | 78.3% | 22/51 | **4.97s** | **7.78s** | **2.98s** | **18.3 tok/s** |
| qwen3-0.6b-dpo-b01-q4-k-m | 86.3% | 79.7% | 24/51 | 10.10s | 15.10s | 7.79s | 9.3 tok/s |
| qwen3-0.6b-dpo-b03-q4-k-m | 80.4% | 77.0% | 23/51 | 18.64s | 60.25s | 15.41s | 6.7 tok/s |
| qwen3-0.6b-sft-f16 | 86.3% | **81.7%** | **25/51** | 12.19s | 18.31s | 9.54s | 7.1 tok/s |
| qwen3-1.7b-base-q4-k-m | 68.6% | 58.1% | 7/51 | 11.25s | 17.37s | 8.07s | 6.8 tok/s |
| qwen3-1.7b-sft-q4-k-m | 94.1% | 91.9% | 29/51 | **12.08s** | 17.64s | **8.99s** | **7.6 tok/s** |
| qwen3-1.7b-dpo-b01-q4-k-m | **96.1%** | 90.5% | **30/51** | 12.49s | **17.22s** | 9.34s | 7.5 tok/s |
| qwen3-1.7b-dpo-b03-q4-k-m | 92.2% | **92.1%** | 29/51 | 12.61s | 17.75s | 9.42s | 7.3 tok/s |
| qwen3-1.7b-sft-f16 | 88.2% | 91.8% | 29/51 | 20.09s | 29.07s | 16.10s | 4.5 tok/s |

粗体表示同一参数规模的后训练模型中该列的最好结果，不用于自动选择模型。0.6B 组中，SFT F16 的质量最高，但 SFT Q4 的速度和体积优势明显；两个 DPO Q4 的质量增益伴随较大的时延波动。1.7B 组中，DPO b01 的协议遵循和有效通过数最高，DPO b03 的平均任务正确性最高；SFT Q4 与 F16 的任务正确性几乎相同，但本地推理更快。两个 Base 模型均明显弱于对应的后训练模型。

### 性能与模型体积

下表记录 Windows CPU 本地实测的性能数据，不表示额外的质量得分。各列定义如下：

- **质量集平均时延**：51 条冻结质量用例的端到端生成时间均值，包含请求、Prompt 处理和输出生成时间。
- **质量集 P95 时延**：上述 51 次端到端时延的第 95 百分位数，约 95% 的请求可以在该时间以内完成。
- **生成吞吐（tok/s）**：运行质量集时的平均输出 Token 生成速度，数值越高越快；不包含 Prompt 处理吞吐。
- **Short 热运行均值**：完成 1 次不计入统计的预热后，连续 5 次 short Prompt 请求的端到端平均时延；此时服务和模型均已启动。
- **4K 热运行均值**：使用名义 4K 长度负载进行 5 次热运行的平均时延，主要反映长 Prompt 处理和生成耗时。
- **模型体积（GiB）**：GGUF 文件的二进制体积，计算方式为 `bytes / 1024^3`。

对 short、medium、2K、4K 每档负载，评估器都执行 1 次首次请求、1 次不计入统计的预热请求和 5 次计时热请求。表中只展示热运行均值，因为它更接近本地服务启动后的重复使用场景。`2K` 和 `4K` 是通过重复中文文本构造的固定 Prompt 规模档位，不代表每种 tokenizer 都会产生精确的 2,000 或 4,000 个输入 Token。因此，这些时延主要用于比较同一次实验、同一硬件环境下不同模型的相对表现。

| 模型 | 质量集平均时延 | 质量集 P95 时延 | 生成吞吐（tok/s） | Short 热运行均值 | 4K 热运行均值 | 模型体积（GiB） |
|---|---:|---:|---:|---:|---:|---:|
| qwen3-0.6b-base-q4-k-m | 5.09s | 6.32s | 18.02 | 0.99s | 5.17s | 0.45 |
| qwen3-0.6b-sft-q4-k-m | 4.97s | 7.78s | 18.26 | 0.76s | 3.90s | 0.37 |
| qwen3-0.6b-dpo-b01-q4-k-m | 10.10s | 15.10s | 9.31 | 2.22s | 5.42s | 0.37 |
| qwen3-0.6b-dpo-b03-q4-k-m | 18.64s | 60.25s | 6.70 | 1.23s | 3.58s | 0.37 |
| qwen3-0.6b-sft-f16 | 12.19s | 18.31s | 7.08 | 2.49s | 5.50s | 1.12 |
| qwen3-1.7b-base-q4-k-m | 11.25s | 17.37s | 6.82 | 2.58s | 24.34s | 1.19 |
| qwen3-1.7b-sft-q4-k-m | 12.08s | 17.64s | 7.55 | 2.36s | 21.45s | 1.03 |
| qwen3-1.7b-dpo-b01-q4-k-m | 12.49s | 17.22s | 7.46 | 6.03s | 24.16s | 1.03 |
| qwen3-1.7b-dpo-b03-q4-k-m | 12.61s | 17.75s | 7.33 | 2.19s | 24.13s | 1.03 |
| qwen3-1.7b-sft-f16 | 20.09s | 29.07s | 4.51 | 8.76s | 35.00s | 3.21 |

### 结果观察

- `qwen3-1.7b-dpo-b01-q4-k-m` 的协议遵循得分最高，为 96.1%。
- `qwen3-1.7b-dpo-b03-q4-k-m` 的任务正确性最高，为 92.1%。
- 1.7B SFT Q4 与对应 F16 对照模型的任务正确性基本持平（91.9% 对 91.8%），同时速度更快、体积更小。
- 0.6B SFT Q4 相比对应 F16 对照模型损失约 3.4 个百分点的任务正确性。
- 两个 Base 模型均明显弱于对应的后训练模型。
- 以上结果只作为实验观察，本阶段不设置自动胜出模型或质量验收阈值。

## 阶段结论

走到这一步，我们已经获得了完整的质量、性能和模型体积数据，但这些实验结果还不足以支持“哪个模型最好”或“最终应该使用哪个模型”的结论。本阶段没有设置统一的业务验收阈值，也没有在质量、速度和资源占用之间定义固定权重，因此不在这里进行最终模型选型。

第五阶段能够确认的核心结论是：**从 Adapter 合并、GGUF 转换、imatrix 校准、Q4_K_M 量化，到 llama.cpp 本地推理、统一评估和双模型 App 对比，整条工程流程已经跑通。** 无论从冻结数据集的评测结果，还是 App 中的直观使用效果来看，后训练模型整体上都明显优于对应的 Base 模型，说明前面阶段的 SFT/DPO 训练已经产生了可以观察到的正向效果。

与此同时，当前结果也暴露出下一阶段需要重点回答的问题：

- 为什么 DPO 相比 SFT 的提升并不稳定，部分模型的收益很小，甚至伴随指标回退？
- 为什么协议遵循和任务正确性已经较高，但严格口径下的有效通过率仍然偏低？
- 失败样本主要集中在哪些场景、字段和决策边界，现有训练数据是否覆盖不足或分布失衡？
- 如何在保持质量的同时进一步降低平均时延、P95 和首字时延，提高本地 CPU 推理吞吐？
- 模型规模、量化档位、Prompt、采样参数和推理后端分别对质量与性能产生了多大影响？

前几个阶段更侧重于建立数据、训练、量化、评估和应用的完整闭环，目标是证明流程可执行、结果可记录、实验可复现。下一阶段将从“跑通流程”转向“基于数据和指标进行迭代”：分析具体失败样本，定位误差来源，调整数据配比和训练策略，必要时重新执行 SFT/DPO，并持续验证质量与推理性能是否得到真实提升。
