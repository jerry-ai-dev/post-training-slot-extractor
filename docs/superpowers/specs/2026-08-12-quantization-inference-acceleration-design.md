# 阶段五设计：量化、CPU 推理评测与多轮工具对话应用

> Status: Approved
> Date: 2026-08-12
> 对应阶段：Phase 05 - Quantization and Local Deployment
> 前置：阶段四六组训练 run、本地冻结集评测与既有 llama-server inference harness

---

## 1. 目标与原则

阶段五把阶段四产出的 Qwen3 Base/SFT/DPO 模型转换为 GGUF、生成 Q4_K_M 量化产物，在当前 CPU-only 机器上按统一口径评测质量、速度、内存和体积，并提供一个可切换任意已注册模型的双栏多轮对话应用。

本阶段不是只处理阶段四候选冠军，也不预设质量回退阈值。它是一轮探索性评测：完整保留原始指标、逐样本变化和人工体验证据，为后续补做 Q5_K_M/Q8_0 或继续迭代提供依据。

关键原则：

1. **配置驱动**：模型清单、量化矩阵和 Demo 事实均由版本化配置描述，不把首批八个模型写死在代码或 UI 中。
2. **可追溯**：每个 GGUF 可追溯到基座 revision、adapter、合并顺序、转换参数、imatrix 数据、llama.cpp 版本和文件哈希。
3. **同尺对比**：正式性能测试顺序运行，固定机器、运行参数、数据和后端；并行模式仅用于体验。
4. **忠于协议**：对话应用复用现有 prompt、输出 schema、消息历史和业务规则，并真实执行 Host 侧 tool loop。
5. **Mock 事实而非伪造链路**：外部技师系统的数据是 Mock 的；工具参数校验、工具分派、结果回填和模型续推是真实执行的。
6. **不掩盖失败**：不自动修复非法 JSON，不把 Mock 数据未覆盖伪装成业务 `no_match`，不静默删除失败样本。
7. **YAGNI**：首版不引入数据库、任务队列、容器平台、复杂前端框架、公网部署或登录系统。

### 1.1 当前部署基准机

设计基于 2026-08-11 实机采集结果：

| 项目 | 值 |
|---|---|
| 运行环境 | Hyper-V Virtual Machine |
| OS | Windows 11 Enterprise，Build 26200 |
| CPU | AMD EPYC 7763 |
| 分配资源 | 8 physical cores / 16 logical processors |
| 内存 | 63.95 GiB |
| GPU | 无可用计算 GPU |
| Python | 3.12.10 |
| PyTorch | 2.13.0+cpu |
| CUDA | unavailable |
| C 盘 | 约 2 TiB，总空闲约 1.46 TiB（采集时） |

因此主部署与基准路径为 Windows CPU + 原生 llama.cpp/`llama-server.exe`。

### 1.2 本阶段不设门槛

本阶段不定义：

- 量化质量 pass/fail 阈值；
- 自动淘汰规则；
- Q5_K_M/Q8_0 自动晋级规则；
- 强制统一的最终量化档位。

报告只陈述数据、变化和观察。首轮完成后，再由人工选择值得补做的模型与档位。

---

## 2. 交付范围

### 2.1 包含

1. 可扩展模型注册表；
2. merge、HF→GGUF、imatrix、quantize、verify 流水线；
3. 首批 8 个 Q4_K_M 模型；
4. 8 个 Q4_K_M 的冻结集评测；
5. 0.6B SFT 与 1.7B SFT 的 F16 GGUF 同后端锚点评测；
6. 质量、速度、内存、体积和逐样本对比报告；
7. 统一 llama-server 生命周期管理器；
8. FastAPI + 原生 HTML/CSS/JavaScript 双栏对话应用；
9. 顺序公平模式与并行体验模式；
10. 严格的多轮 tool loop；
11. 忠于现有数据集语义的 Mock `find_technicians`；
12. 主区突出 `final.reply`，浅色可展开过程区展示模型 JSON、工具调用和工具结果；
13. 完整对比记录导出与复现元数据；
14. 为新模型、新量化档位、新工具和真实数据源保留明确接口。

### 2.2 不包含

- GPTQ/AWQ 或 GPU 部署；
- 自动质量阈值与模型排名淘汰；
- 第一轮自动批量生产全部 Q5_K_M/Q8_0；
- 真实技师数据库、真实排班 API 或实际创建预约；
- 生产级排班系统、跨日规则、周期班表、请假、预约写入和并发冲突处理；
- 公网部署、鉴权、多用户隔离；
- 数据库持久化聊天历史；
- 盲测 Arena；
- React/Vue 等前端构建链；
- 自动改写模型非法 JSON 或不忠实回复。

---

## 3. 整体架构与边界

```text
                         configs/models/registry.yaml
                                    │
                    ┌───────────────┼────────────────┐
                    ▼               ▼                ▼
           量化生产流水线       正式评测编排器      双栏对话应用
                    │               │                │
                    ▼               ▼                ▼
      merged / gguf / imatrix   scorecards      多轮 Tool Loop
                    │               │                │
                    └───────────────┴────────────────┘
                                    │
                             llama-server 管理器
                                    │
                                    ▼
                           本地 llama-server.exe
```

组件职责：

- **模型注册表**：描述模型身份、lineage、产物、runtime 和能力；是 UI、评测和服务启动的共同模型目录。
- **量化流水线**：生产模型产物，不负责聊天或评分。
- **llama-server 管理器**：启动、健康检查、复用和停止进程，不负责业务 tool loop。
- **评测编排器**：顺序执行冻结集和性能 workload，复用已有 scorer。
- **ConversationOrchestrator**：解析模型 JSON、驱动工具循环、维护每侧会话状态。
- **ToolRegistry/Executor**：校验并执行已知工具，不负责模型推理。
- **FastAPI UI**：展示、控制和导出，不复制模型、工具或评分逻辑。

核心约束是：**注册表描述模型，流水线生产模型，服务管理器运行模型，Orchestrator 驱动业务循环，评测器测模型，UI 只投影事件。**

---

## 4. 首批模型矩阵

第一轮统一生产 8 个 Q4_K_M：

| 参数规模 | 阶段 | 初始模型 ID |
|---|---|---|
| 0.6B | Base | `qwen3-0.6b-base-q4-k-m` |
| 0.6B | SFT | `qwen3-0.6b-sft-q4-k-m` |
| 0.6B | DPO β=0.1 | `qwen3-0.6b-dpo-b01-q4-k-m` |
| 0.6B | DPO β=0.3 | `qwen3-0.6b-dpo-b03-q4-k-m` |
| 1.7B | Base | `qwen3-1.7b-base-q4-k-m` |
| 1.7B | SFT | `qwen3-1.7b-sft-q4-k-m` |
| 1.7B | DPO β=0.1 | `qwen3-1.7b-dpo-b01-q4-k-m` |
| 1.7B | DPO β=0.3 | `qwen3-1.7b-dpo-b03-q4-k-m` |

Base 用于训练前后对照；SFT/DPO 用于训练策略对照。阶段四的 `phase04-qwen3-1.7b-sft` 仍是当前候选冠军，但本阶段不因此跳过其他候选。

DPO lineage 不依靠命名猜测。注册配置必须显式记录实际 base、训练 run、adapter 输入和最终 adapter；流水线根据阶段四 manifest 解析真实合并输入，并拒绝无法证明的 adapter 顺序。

---

## 5. 配置契约

### 5.1 模型注册表

示意：

```yaml
models:
  - id: qwen3-1.7b-sft-q4-k-m
    display_name: Qwen3 1.7B · SFT · Q4_K_M
    family: qwen3
    parameter_size: 1.7b
    stage: sft
    quantization: Q4_K_M
    lineage:
      base_model: Qwen/Qwen3-1.7B
      training_run: phase04-qwen3-1.7b-sft
    artifacts:
      gguf: models/gguf/qwen3-1.7b-sft/qwen3-1.7b-sft-Q4_K_M.gguf
      manifest: models/gguf/qwen3-1.7b-sft/manifest.json
    runtime:
      chat_template: qwen3
      context_size: 4096
      threads: 8
      batch_size: 512
    capabilities:
      evaluation: true
      interactive: true
      tool_loop: true
```

加载时严格校验：

- ID 唯一；
- lineage 可解析；
- 必需路径存在；
- GGUF 哈希与 manifest 一致；
- quantization、chat template、context、threads、batch 合法；
- 未知字段报错，不静默忽略。

UI 调用模型目录 API 动态发现模型，不保存固定清单。新增模型的常规流程是添加注册配置、准备 GGUF、通过校验、重载目录。

### 5.2 量化矩阵

量化实验配置与长期模型注册表分开：

```yaml
experiment: phase05-q4-matrix
targets:
  - qwen3-0.6b-base
  - qwen3-0.6b-sft
  - qwen3-0.6b-dpo-b01
  - qwen3-0.6b-dpo-b03
  - qwen3-1.7b-base
  - qwen3-1.7b-sft
  - qwen3-1.7b-dpo-b01
  - qwen3-1.7b-dpo-b03
quantizations: [Q4_K_M]
```

以后补做 Q5_K_M/Q8_0 时创建新矩阵，不修改首轮实验配置或结果。

### 5.3 Demo 技师库配置

Demo 配置保存：

- 固定、可查看的技师目录；
- 姓名、性别和合同支持的专长；
- 固定日期上的可用时间段；
- 固定演示时钟、营业时间和日期范围；
- fixture 版本与内容哈希；
- Demo 匹配规则版本。

`find_technicians` 不读取预设问答结果，而是根据这份技师库实时执行确定性匹配。查询日期超出 Demo 日历、偏好无法按已声明专长解释或数据本身不完整时，返回独立的 `mock_coverage_miss`，不能冒充业务 `no_match`。

---

## 6. 量化生产流水线

### 6.1 步骤

```text
resolve → merge → convert-f16 → build-imatrix → quantize → verify
```

- Base 模型跳过 adapter merge；
- SFT/DPO 根据 manifest 解析输入；
- 每一步生成结构化 manifest、命令、日志、版本和哈希；
- 输入与参数哈希一致时可跳过；
- 哈希变化使缓存失效；
- 单模型失败不阻断其他矩阵项；
- 大文件先写临时路径，校验成功后原子移动；
- 失败中间产物保留在明确的 failed 路径供分析，不冒充成功缓存。

### 6.2 imatrix 校准数据

imatrix 使用版本化领域文本：

- 来源是训练侧领域语料；
- 不读取冻结评测集，防止泄漏；
- 覆盖工具调用、字段提取、澄清、拒绝和闲聊；
- 固定选择规则、顺序和随机种子；
- 保存数据清单与内容哈希；
- 八个模型使用同一份校准文本，但每个最终模型单独计算 imatrix。

### 6.3 产物组织

```text
models/
├── merged/<model-id>/
│   ├── model files
│   └── manifest.json
├── imatrix/<model-id>/
│   ├── calibration.txt
│   ├── imatrix.dat
│   └── manifest.json
└── gguf/<model-id>/
    ├── <model-id>-F16.gguf
    ├── <model-id>-Q4_K_M.gguf
    └── manifest.json

experiments/runs/phase05-<model-id>-q4-k-m/
├── config.rendered.yaml
├── quantization.json
├── server.json
├── predictions.jsonl
├── scorecard.json
├── comparison.json
└── logs/
```

大型权重继续排除在 Git 外；Git 只保留配置、manifest、指标、摘要和可复现命令。

### 6.4 可追溯字段

每个最终 GGUF 至少记录：

- Hugging Face repo 与 revision；
- adapter run、路径和哈希；
- 合并顺序与 dtype；
- Transformers/PEFT/llama.cpp 版本；
- HF→GGUF 参数；
- imatrix 数据版本与哈希；
- quantize 命令；
- 输出 GGUF SHA-256；
- 当前硬件和 runtime 参数。

---

## 7. llama-server 生命周期管理

统一管理器负责：

- 从注册配置渲染启动参数；
- 动态端口分配；
- 进程启动、健康检查、预热、超时和优雅停止；
- 记录 threads、context、batch、版本和完整命令；
- 请求取消与异常进程回收；
- 应用退出时仅终止自己创建的进程。

### 7.1 顺序公平模式

默认模式在同一时间只让一个模型占用正式 benchmark CPU 资源：

```text
左侧完整 turn → 停止/释放 → 右侧完整 turn
```

左右使用相同用户输入、生成参数、Demo 技师库快照和时钟。实际执行顺序写入导出记录，可固定或交替首发。只有该模式的数据标记为可比较。

### 7.2 并行体验模式

并行模式启动两个独立端口，并按配置拆分 threads。它用于同时观看回答流，不用于正式速度比较。UI 持续提示资源竞争，导出写入 `comparable: false`，结果不进入正式 benchmark 报告。

---

## 8. 探索性评测协议

### 8.1 质量评测

8 个 Q4_K_M 全部运行现有冻结集，复用当前 scorer 和口径，包括：

- `effective_pass_count`；
- protocol compliance；
- task correctness；
- JSON 合法率；
- 字段和工具准确性；
- 回复语义与忠实度；
- 场景切片；
- 逐样本预测、失败原因、翻正与翻负。

每个量化模型对比：

1. 同规模 Base Q4_K_M，用于观察训练前后变化；
2. 对应 Phase 4 run，用于观察转到 GGUF/Q4 后的变化。

Phase 4 是 Hugging Face CPU backend，Phase 5 是 llama.cpp。报告必须明确：两阶段 delta 可能同时包含量化、模板、采样和后端差异，不能全部归因于量化。

为分离后端与量化影响，首轮至少对 0.6B SFT 和 1.7B SFT 运行 F16 GGUF 锚点评测：

```text
HF merged → F16 GGUF：主要观察后端/模板差异
F16 GGUF → Q4_K_M：主要观察量化差异
```

其他 F16 GGUF 是否完整评测，由首轮数据后人工决定。

### 8.2 性能协议

固定：

- 当前基准机；
- llama.cpp 版本；
- threads、context、batch、temperature、seed、最大输出；
- 单模型顺序执行；
- 预热策略和样本顺序；
- 冷启动与热运行分开；
- 失败与超时保留。

记录：

- 文件体积；
- 模型加载耗时；
- prompt/generated tokens；
- prompt processing time/tokens per second；
- TTFT；
- decode tokens per second；
- 请求总耗时；
- 进程峰值工作集内存；
- 成功、超时、取消和异常状态。

汇总提供 count、mean、median、P90、min/max；样本不足时明确标注，不夸大 P90。

### 8.3 workload

1. **任务真实负载**：冻结集端到端请求。
2. **固定长度基准**：短上下文、中上下文、2K、4K，固定输出预算。

8K 属于可选压力测试，不纳入首轮八模型强制矩阵。

### 8.4 Tool Loop 的三层性能

- **模型调用**：load、TTFT、prefill、decode、tokens、request total；
- **工具调用**：校验、Mock 查询、序列化耗时；
- **完整用户 turn**：从发送到最终 `reply`，包含一至多次模型调用和工具执行。

正式报告不能用某一层替代其他层。

### 8.5 报告

```text
reports/phase05/
├── model-scorecards/
├── comparisons/
│   ├── base-vs-sft.json
│   ├── sft-vs-dpo.json
│   └── phase04-vs-q4.json
├── benchmark-summary.json
├── sample-regressions.jsonl
└── report.md
```

总报告展示质量、速度、内存、体积、训练策略、F16/Q4 锚点、逐样本索引、环境、命令和限制；不输出自动 pass/fail。人工对话导出与冻结集正式报告分开。

---

## 9. 多轮 Tool Loop 对话应用

### 9.1 技术选择

采用 FastAPI + 原生 HTML/CSS/JavaScript：

- 后端与现有 Python schema、prompt、inference 代码容易复用；
- SSE 或流式 HTTP 支持生成事件；
- 无 React/Vue 构建链；
- 双栏状态、进程生命周期和导出格式不受低代码 UI 框架限制。

应用默认仅监听 `127.0.0.1`。

### 9.2 每侧执行状态机

```text
用户消息
    ▼
PromptBuilder(system + history + current_state + tools + user)
    ▼
模型生成原始文本
    ▼
解析并严格校验 JSON
    ├── action=final
    │      ├── 保存完整 final JSON
    │      ├── 运行 assertions
    │      ├── 主区展示 final.reply
    │      └── 等待用户
    └── action=tool_call
           ├── 校验已知工具和参数
           ├── 写 assistant tool_calls 历史
           ├── ToolRegistry 分派 executor
           ├── 写 tool result 历史
           ├── 更新 current_state
           └── 再次调用模型
```

循环终止条件：

- 返回 `final`；
- 每个用户 turn 达到 3 次工具调用保护上限；
- JSON/工具参数校验失败；
- 工具内部失败；
- 用户取消；
- 模型或服务超时。

3 次上限是防无限循环保护，不是质量阈值。

### 9.3 复用既有协议

复用：

- `PromptBuilder.build_messages`；
- `validate_tool_call_output` / `validate_final_output`；
- user/assistant/tool 历史格式；
- `tool_call_id` 匹配规则；
- current state 字段和失效规则；
- hallucination/faithfulness assertions。

新增：

- `ConversationOrchestrator`；
- `ToolRegistry`；
- `FindTechniciansExecutor`；
- Demo fact repository；
- 旧 tool result/time 兼容归一化层；
- UI event projection 和导出层。

现有 `MockBackend` 只按 sample ID 返回预制模型输出，不能作为工具执行器。

### 9.4 双栏 UI

左右各可选择任意 interactive 模型，展示：

- 模型分组、参数规模、stage、量化档位和 lineage；
- 独立多轮历史；
- 主聊天 reply；
- 浅色可展开过程事件；
- 每次模型调用和完整 turn 的性能；
- schema/assertion 状态；
- 清空、停止、重试、复制和导出。

顶部提供：

- 顺序公平/并行体验切换；
- 共享 temperature、seed、max tokens、context；
- 同步输入；
- 交换模型；
- 清空双方；
- 服务和模型状态；
- 可展开的 Mock 技师库面板，显示姓名、性别、专长、固定日期可用时段、fixture 版本和演示时钟。

每次工具调用的过程区展示逐技师匹配解释：姓名、性别、专长和请求时间分别为何命中或排除，以及最终状态。该解释只供人核对，不回填给模型。

切换某侧模型时默认提示清空该侧上下文；允许明确保留，但必须记录切换边界。浏览器刷新后默认不持久化，用户可主动导出 JSON。

### 9.5 过程事件

每轮时间线：

```text
用户消息
├── 模型原始 tool_call JSON
├── schema 校验
├── Host 执行 find_technicians(...)
├── 规范 tool result
├── 模型原始 final JSON
├── assertions
└── 主回复 final.reply
```

视觉约定：

- reply 使用主要聊天气泡；
- 模型 JSON 浅灰；
- 工具调用/结果浅黄；
- 错误浅红；
- 默认摘要，可展开完整 JSON；
- 原始文本即使解析失败也保留；
- 每个事件显示轮次、耗时和状态。

---

## 10. 忠于数据集的 Mock `find_technicians`

### 10.1 Mock 边界

- **真实执行**：解析工具请求、严格校验五个参数、判断 specific/search、读取技师库、计算预约结束时间、逐项匹配、构造规范结果、写回历史并再次调用模型。
- **Mock 内容**：技师姓名、性别、专长以及固定日期的可用时间段。

原数据集定义了工具参数、技师基本事实和离散结果状态，但没有给出完整日历或时间区间算法。经本阶段设计确认，Demo 增加一套简单、确定性、可见的固定日期日历；它是演示扩展，不冒充原始数据事实，也不扩展为生产级排班系统。

### 10.2 数据集事实与 Demo 扩展

数据集合同明确：

- 王芳：female，肩颈/精油/足部；
- 李明：male，泰式/足部/精油；
- 营业时间事实：09:00–21:00；
- 工具五参：`technician_name`、`start_time`、`duration_minutes`、`gender_preference`、`preferences`；
- specific 查询返回 `available/unavailable/not_found`；
- search 查询返回 `matched/no_match`；
- 指定技师失败时不得自动替换；条件变化后必须重新查询。

Demo 明确增加：

- 固定日期上的可用时间段；
- 预约区间必须完整包含于某个可用时间段；
- 姓名、性别、可识别专长和时间区间的确定性匹配顺序；
- UI 可见的逐技师匹配 trace。

首版不增加周期班表、休假、已占用预约写入、并发锁或跨时区逻辑。

### 10.3 固定日期技师库

示意 fixture：

```yaml
version: phase05-demo-v1
demo_clock: "2026-08-12 10:00"
business_hours: {start: "09:00", end: "21:00"}
calendar_range: {start: "2026-08-13", end: "2026-08-15"}
technicians:
  - id: tech-wang-fang
    name: 王芳
    gender: female
    specialties: [肩颈, 精油, 足部]
    availability:
      "2026-08-13":
        - {start: "09:00", end: "12:00"}
        - {start: "14:00", end: "18:00"}
      "2026-08-14":
        - {start: "10:00", end: "17:00"}
  - id: tech-li-ming
    name: 李明
    gender: male
    specialties: [泰式, 足部, 精油]
    availability:
      "2026-08-13":
        - {start: "10:00", end: "14:00"}
      "2026-08-14":
        - {start: "13:00", end: "20:00"}
```

fixture 加载时校验 ID/姓名唯一、专长非空、日期在 calendar range 内、时段格式合法且不重叠、时段位于营业时间内。界面“Mock 技师库”面板直接读取同一份已校验快照，不能维护另一套展示数据。

### 10.4 确定性匹配规则

输入归一化后按固定顺序执行：

1. 解析 `start_time`，以 `duration_minutes` 计算半开预约区间 `[start, end)`；
2. 校验预约不跨日、日期在 Demo 日历范围内、区间在营业时间内；
3. `technician_name != null` 时只保留精确姓名匹配者；姓名不存在返回 `not_found`，绝不寻找替代者；
4. `gender_preference != null` 时要求性别完全相同；
5. 对能与技师库专长精确对应的正向偏好，要求全部包含于 `specialties`；
6. 要求预约区间完整包含于该技师某个 availability 区间；边界相接不算重叠，恰好在可用时段末端结束有效；
7. 以技师 fixture 顺序产生稳定结果和 trace。

`preferences` 是开放自然语言字段。首版只计算可与声明专长精确对应的正向项（肩颈、精油、足部、泰式）；“轻柔、安静、不要精油、会英语”等未建模条件不能被静默忽略，也不能擅自映射。遇到这类条件时返回 `mock_coverage_miss` 并在 UI 说明未建模项。

为忠于现有模型协议，Demo fixture 应设计成首批受支持查询最多命中一个候选。fixture 校验或测试发现同一受支持查询可命中多个候选时，应明确报出歧义并补充区分数据；不能静默挑第一个。

### 10.5 两种结果模式

#### specific：`technician_name != null`

- 姓名不存在：`not_found`；
- 姓名存在，但性别、可识别专长或时间任一不符合：`unavailable`；
- 所有条件符合：`available`；
- 不返回替代技师。

#### search：`technician_name == null`

- 唯一技师满足性别、可识别专长和完整时间区间：`matched`；
- 所有技师均被明确条件排除：`no_match`；
- 查询包含未建模条件、超出 Demo 日期范围或出现多候选歧义：`mock_coverage_miss`，不伪装成 `no_match`。

### 10.6 模型可见结果与 UI trace 分离

回填模型历史仍只使用训练协议的规范结果：

```json
{
  "mode": "search",
  "status": "matched",
  "requested_technician": null,
  "candidates": [
    {"name": "王芳", "gender": "female"}
  ]
}
```

UI 则展示同一查询的解释性 trace，例如：

```text
查询：2026-08-13 15:00–16:00，female，肩颈
王芳：性别 ✓  专长 ✓  时间 ✓  → matched
李明：性别 ✗  专长 ✗  时间 ✗  → excluded
```

trace 包含归一化参数、逐技师姓名/性别/专长/时间判断、最终状态和 fixture 哈希，只供人核对，不回填给模型。这样用户可同时看到技师库、工具请求和工具为何返回该结果。

### 10.7 条件变化与状态失效

时间、时长、指定姓名、性别偏好或 preferences 任一变化时：

- 旧工具证据失效；
- `technician_status` 回到 `not_checked`；
- 旧 `technician_gender` 不继承为筛选条件；
- 模型必须重新查询；
- final 只能引用最新工具结果支持的技师。

左右共享同一个只读技师库快照、固定时钟和匹配规则版本，但历史、current state、call ID、草稿和错误独立。首版工具只查询，不修改 availability，避免左右相互污染。

### 10.8 合同漂移处理

现有原始数据包含旧 `{status, technicians}` 结果、ISO 时间和其他变体。兼容层可以读取并归一化，但新 App 只生成 canonical specific/search 结果和 `YYYY-MM-DD HH:MM` 时间。

`not_found` 的姓名在个别旧 eval 样本与当前 generator/validator 之间不一致。新 App 采用当前严格契约：

```json
{
  "mode": "specific",
  "status": "not_found",
  "requested_technician": "陈静",
  "technician": null
}
```

结构化 final 中 `technician_name` 为 `null`；请求名保留在 `requested_technician` 和对话语义中。旧差异写入兼容性说明，不继续扩大格式漂移。

---

## 11. 错误处理与恢复

### 11.1 模型协议错误

- 非 JSON 或 schema 错误：保存原文，展示错误，不执行工具，本侧停止；
- 未知工具/非法参数：拒绝执行并记录 tool protocol failure；
- final 无最新工具证据、使用不存在技师或沿用失效结果：保留 reply 供观察，同时显示 assertion warning；
- 不自动猜测修复；提供显式“重试本侧”。

### 11.2 工具错误

- `mock_coverage_miss` 与 `no_match` 分开；
- 取消、超时、内部错误分开记录；
- 内部错误可回填一次规范 error result 让模型解释，但必须标记非正常业务结果；
- 达到循环上限时标记 `tool_loop_limit_exceeded`。

### 11.3 流水线恢复

- manifest + 输入哈希控制断点续跑；
- 来源或哈希不一致的缓存不复用；
- 单模型失败不阻断矩阵；
- 大产物原子落盘；
- 正式评测、人工对话和并行体验分别存储。

---

## 12. 测试策略

### 12.1 量化流水线

- Base 跳过 merge；
- SFT/DPO lineage 和合并输入解析；
- 输入哈希变化使缓存失效；
- 已完成步骤安全跳过；
- 单模型失败隔离；
- imatrix 不含冻结集；
- llama.cpp 命令渲染；
- manifest 完整；
- 量化档位通过配置扩展。

普通测试使用 fixture/dry-run，不转换大型模型。

### 12.2 注册表与服务管理

- 注册、过滤、分组；
- 重复 ID、缺失文件、错误哈希；
- 动态端口、health、启动超时；
- 顺序互斥与并行参数拆分；
- 取消、异常退出、进程回收；
- 只停止管理器创建的进程。

### 12.3 Tool Loop

- 直接 final；
- `tool_call → result → final`；
- 连续合法工具调用；
- 非 JSON、schema 错误、未知工具；
- tool_call_id 匹配；
- 循环上限；
- 取消和超时；
- 条件变化使旧结果失效；
- final 非法引用触发 warning；
- 解析失败仍保留原文。

### 12.4 Mock 业务语义与匹配引擎

至少覆盖：

1. 技师 fixture 的 ID/姓名唯一、日期范围、营业时间和时段不重叠校验；
2. 指定王芳，在支持的专长和可用区间内 → `available`；
3. 指定李明，但请求区间不完整落在可用时段内 → `unavailable`；
4. 指定陈静 → `not_found`，且不自动返回替代者；
5. female + 肩颈 + 王芳可用时段 → 唯一 `matched` 王芳；
6. female + 泰式 → `no_match`；
7. 放宽性别且李明时间可用 → `matched` 李明；
8. 预约恰好在 availability 末端结束有效，超出一分钟无效；
9. 预约跨日、超出固定日期范围或营业时间 → `mock_coverage_miss`；
10. 包含“轻柔、安静、不要精油”等未建模偏好 → `mock_coverage_miss`；
11. 同一查询出现多个候选时报告歧义，不静默选择首位；
12. 五个查询条件任一变化必须重查；
13. 新结果始终使用 canonical schema；
14. UI 技师库快照、匹配 trace 和 executor 使用相同 fixture 哈希；
15. 接受可用结果 → `booking_authorized`；
16. 拒绝可用结果 → `appointment_paused`；
17. 知悉失败结果 → `acknowledge_result`。

### 12.5 API/UI 集成

- `/api/models` 仅返回可用 interactive 模型；
- 双栏同步发送与状态隔离；
- 顺序公平执行顺序；
- SSE 事件顺序；
- 主气泡仅取 `final.reply`；
- 中间事件可展开；
- Mock 技师库面板显示 executor 实际使用的姓名、性别、专长和固定日期 availability；
- 每次工具调用显示逐技师匹配/排除 trace，且与规范 tool result 一致；
- 单侧失败不阻止另一侧；
- 模型切换边界；
- 导出含 lineage、fixture、原始事件和性能；
- 并行模式为 `comparable: false`。

### 12.6 本地真实冒烟

至少选一个 0.6B Q4_K_M 和一个 1.7B Q4_K_M 跑通：

```text
加载 GGUF → 用户多轮输入 → tool_call → Mock 工具 → 回填 → final → UI → 导出
```

该测试标记 local/integration，不放入默认快速测试。

---

## 13. 完成标准

1. 8 个 Q4_K_M 均有可追溯 manifest，或有明确可复现的失败记录；
2. 每个成功模型通过 llama-server 启动和生成冒烟；
3. 所有成功 Q4 模型产生冻结集 scorecard；
4. 两个 SFT F16 GGUF 锚点完成同后端评测；
5. 总报告不设 pass/fail，只记录指标、变化和限制；
6. App 从注册表动态加载模型，无硬编码模型清单；
7. 至少一个 0.6B 和一个 1.7B 跑通完整多轮工具循环；
8. 左右模型对相同输入、相同 fixture 和相同生成参数独立比较；
9. 主界面突出 reply，中间模型 JSON/tool 事件可审计；
10. 界面可查看 executor 实际使用的 Mock 技师姓名、性别、专长和固定日期可用时段；
11. 每次工具调用可查看逐技师匹配 trace，并能从技师库人工核对结果；
12. Mock 工具协议由数据集语义派生，固定日期日历扩展由确定性测试锁定；
13. 默认测试通过，真实本地冒烟有运行记录；
14. 文档能指导未来通过配置加入新模型和量化档位。

---

## 14. 已知限制与后续入口

- Mock availability 是固定日期、只读、确定性的 Demo 日历，不是生产级排班引擎；
- 现有数据存在 tool result 和 `not_found` 姓名语义的历史漂移；
- HF Phase 4 与 llama.cpp Phase 5 的直接 delta 不是纯量化影响；
- 当前机器为虚拟 CPU 环境，结果不能直接外推到其他 CPU；
- 首轮仅强制 Q4_K_M，Q5_K_M/Q8_0 由结果后人工选择；
- 未来接真实业务系统时，实现 `TechnicianRepository`/availability provider adapter，并保持 ToolRegistry、Orchestrator、模型协议和 UI 事件格式不变；
- 未来增加工具时，通过 ToolRegistry 注册新的 schema 和 executor，不在 UI 或 ConversationOrchestrator 中添加工具特判。
