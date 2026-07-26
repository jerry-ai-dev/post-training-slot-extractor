# 阶段三：构建训练数据集

## 1. 项目如何完成的

本阶段仍然使用 Superpowers 的完整工作流推进：先围绕第三阶段目标编写具体的 Specs，再根据 Specs 生成可执行的 Plans，最后由 AI 按照计划完成数据集生成与验证。

第三阶段与第二阶段有一定相似之处，核心工作都包含数据集的构建。第二阶段的经验表明，如果在生成数据之前，作者自己还没有想清楚业务规则，或者文档没有把要求定义清楚，那么 AI 即使能够快速生成大量数据，最终的数据集也很容易出现业务逻辑错误、场景覆盖不足或样本标准不一致等问题，后续往往需要反复修改。

因此，这一次在编写 Specs 时，我没有直接让 AI 批量生成数据，而是先与 AI 进行了比较充分的讨论，并重点把以下内容定义清楚：

- 每条数据应该采用什么结构，最终应该呈现为什么样子。
- 数据集需要覆盖哪些真实业务场景。
- 不同业务场景分别对应怎样的输入、输出和处理逻辑。
- 整体业务流程如何推进，各类场景之间如何衔接。
- 数据生成和验收需要遵守哪些规则。

这些关键要求经过确认并写入 Specs 后，再由 AI 根据 Specs 生成具体的实施 Plan，最后按照 Plan 执行数据集构建。

在数据生成过程中，我采用了“小批量验证，再扩大规模”的方式。AI 首先只生成 25 条数据，我先检查这些样本的结构、业务逻辑和场景表达是否符合预期。确认这批数据没有明显问题后，再让 AI 按照同样的规则继续生成 500 条数据，避免在规则存在问题时一次性生成大量错误样本。

500 条数据生成完成后，我又与 AI 一起进行了人工抽检。抽检并不是随机阅读几条数据，而是由我指定某一类具体业务场景，再让 AI 从完整数据集中找出对应的样本，交给我逐条检查。通过这种方式，可以有针对性地确认不同业务分支是否真实存在、样本是否符合业务逻辑，以及输入输出是否满足 Specs 中的定义。

最终，抽检的各类业务场景均顺利通过。相比第二阶段数据生成后经历的多轮修改，本阶段由于前期对 Specs、业务场景、数据结构和整体流程把关得更充分，生成结果基本符合预期，没有出现太多返工。这也进一步说明：使用 AI 批量生成数据时，真正决定数据质量的关键，并不是一次生成多少条，而是生成前是否已经把业务和数据标准定义清楚，以及生成后是否采用了分批验证和场景化抽检。

---

## 2. 整体业务目标和核心逻辑

### 2.1 本阶段的核心目标

第三阶段的核心目标很明确：为下一阶段的模型训练生成可以直接使用的 **SFT 数据集和 DPO 数据集**。

在完成数据生成的同时，还要保证数据集能够覆盖按摩预约 Agent 的各类核心业务场景，避免样本只集中在少数简单情况。最终数据需要符合统一的数据合同和 LLaMA-Factory 所需的 ShareGPT 格式，并通过质量校验、训评隔离和场景覆盖检查，为后续 SFT 与 DPO 训练提供可靠的数据基础。

最终交付的数据集版本为 `v0.1`：包含 500 条 raw 原始样本，由此构建出 450 条 SFT 训练样本、50 条 SFT 验证样本，以及 135 条 DPO 训练样本和 15 条 DPO 验证样本。

### 2.2 整体流水线和核心逻辑

阶段三通过一个统一的 CLI 命令运行整套数据集构建管线：

```powershell
uv run slot-build-dataset --generate `
  --config configs/data/phase03.yaml `
  --output-root data `
  --strict-audit
```

其中，`--output-root data` 明确指定所有生成结果写入仓库的 `data/` 目录。需要生成多少条 raw 数据，则由 `configs/data/phase03.yaml` 中的 `counts` 配置：

```yaml
counts:
  追问: 107
  工具调用: 107
  最终 JSON: 107
  确认: 89
  无关: 90
```

以上五类合计 500 条。命令还会读取该配置文件中的数据版本、具体场景、并发数、难例比例、DPO 配额、随机种子和评估集路径，然后调用 GPT-5.6-sol 生成 raw 原始样本。程序只生成一份 raw 业务数据，SFT 和 DPO 都从这份 raw 数据中确定性派生，不再分别调用模型生成两套内容。

完整流水线如下：

```text
                      [五类数据规格卡]  ← 人工审核关口
                             │ 审核通过后，规格卡成为生成 Prompt 模板
                             ▼
  强模型 GPT-5.6-sol ──生成──► raw 原始样本 ──► data/raw/v0.1/samples.jsonl
                                     │
                                     ▼
                   质量校验、难例与语义覆盖审计、训评隔离
                                     │
                                     ▼
                       按业务类别进行 train/val 分层切分
                                     │
                          ┌──────────┴───────────┐
                    （纯规则渲染）           （纯规则扰动）
                          ▼                       ▼
                SFT ShareGPT 样本       DPO ShareGPT 偏好对
                          │             chosen = raw 标准答案
                          │             rejected = 业务错误答案
                          ▼                       ▼
             data/processed/sft/v0.1/  data/processed/dpo/v0.1/
                          └──────────┬────────────┘
                                     ▼
                  dataset_info.json + DATASET_CARD.md
                                     ▼
                       交给 LLaMA-Factory 进行训练
```

命令内部主要完成以下工作：

1. **生成 raw 数据。** 程序根据配置中的五类业务场景和数量组装请求，并发调用强模型生成 500 条统一格式的 raw 样本。生成过程中持续写入 checkpoint，中断后可以从已完成的样本继续运行。
2. **执行质量门禁与数据统计。** 对 raw 样本检查字段和 Schema、上下文与业务自洽性，同时统计各类数据数量、难例占比、业务场景覆盖和 DPO 配额，并检查训练数据与阶段二冻结评估集是否重叠。使用 `--strict-audit` 时，关键指标不达标就会终止构建。
3. **分层切分数据。** 程序使用固定随机种子 `42`，按照五类业务场景分别打乱并进行 9:1 切分，保证训练集和验证集都能覆盖主要业务类别，同时让相同输入能够稳定复现相同结果。
4. **派生 SFT 数据。** SFT 渲染器复用项目真实的 Prompt 组装逻辑，把 raw 上下文转换为 LLaMA-Factory 能读取的 ShareGPT 格式，并把 raw 中的标准答案作为最后一条模型回复。
5. **派生 DPO 数据。** 程序按照配置的五类错误配额选择 raw 样本，将标准答案作为 `chosen`，再通过确定性规则生成 `rejected`。这些错误答案仍然符合输出 Schema，但会故意制造幻觉、动作选择错误、确认状态错误、字段值错误等业务问题。
6. **写出版本化产物。** 最终同时保存 raw、SFT、DPO、LLaMA-Factory 数据集注册文件和数据卡。数据卡记录版本、构建时间、Git 状态、随机种子、各类样本数量、DPO 类型分布、训评隔离结果和语义覆盖结果。

这张图表达的是脚本本身的处理过程：输入是一份阶段配置和强模型生成能力，中间经过 raw 生成、质量门禁、分层切分、SFT 渲染和 DPO 扰动，输出则是可以直接交给 LLaMA-Factory 使用的版本化训练数据集。

---

## 3. 手动实践

如果想自己体验一次完整的数据集生成过程，建议不要直接生成正式的 500 条数据，而是先生成 10 条小样，检查模型输出和最终数据集是否符合预期。

### 3.1 准备一份 10 条小样配置

先复制正式配置，避免修改当前已经用于正式数据集的参数：

```powershell
Copy-Item configs/data/phase03.yaml configs/data/phase03-smoke.yaml
```

打开 `configs/data/phase03-smoke.yaml`，将 `counts` 改为五类各 2 条，总计 10 条：

```yaml
counts:
  追问: 2
  工具调用: 2
  最终 JSON: 2
  确认: 2
  无关: 2
```

同时把正式数据使用的 DPO 固定配额改为空配置：

```yaml
dpo_target_counts: {}
```

10 条数据只能用于体验管线，无法覆盖正式配置要求的全部业务场景和数据比例，因此本次尝试不使用 `--strict-audit`。正式生成时仍应恢复 500 条配置和严格质量门禁。

### 3.2 配置用于生成数据的模型

`phase03-smoke.yaml` 中的以下字段决定使用哪个模型后端：

```yaml
generate_inference_config: configs/inference/openai_responses_gpt_5.6_sol.yaml
```

如果使用自己的云端模型，建议复制一份推理配置：

```powershell
Copy-Item `
  configs/inference/openai_responses_gpt_5.6_sol.yaml `
  configs/inference/openai_responses_my_model.yaml
```

然后修改新文件：

```yaml
backend: openai_responses
model: your-model-name
base_url_env: OPENAI_BASE_URL
api_key_env: OPENAI_API_KEY
max_tokens: 4096
timeout_s: 180
```

这里要求模型服务兼容 OpenAI Responses API。`OPENAI_BASE_URL` 填写 API 根地址，不要在末尾添加 `/responses`，程序会自动拼接该路径。API 地址和密钥通过环境变量配置，不要直接写入仓库：

```powershell
$env:OPENAI_BASE_URL="<你的 API 根地址>"
$env:OPENAI_API_KEY="<你的 API Key>"
```

最后把 `phase03-smoke.yaml` 指向自己的模型配置：

```yaml
generate_inference_config: configs/inference/openai_responses_my_model.yaml
```

如果希望使用本地 `llama-server`，也可以把该字段指向对应的 `llama_server` 配置，例如 `configs/inference/llama_server_qwen3_4b.yaml`。本地后端不提供 Responses API 的严格 JSON Schema 约束，主要依靠提示词、生成后校验和失败重试，因此建议使用指令遵循能力较强的模型。

### 3.3 运行完整管线

执行以下命令：

```powershell
uv run slot-build-dataset --generate `
  --config configs/data/phase03-smoke.yaml `
  --output-root experiments/runs/phase03-manual
```

命令会调用配置好的模型生成 10 条 raw 数据，并继续完成校验、训评隔离、SFT 渲染、DPO 扰动、训练集与验证集切分以及版本登记。完成后，主要产物位于：

```text
experiments/runs/phase03-manual/
├── raw/v0.1/samples.jsonl
└── processed/
    ├── sft/v0.1/train.jsonl
    ├── sft/v0.1/val.jsonl
    ├── dpo/v0.1/train.jsonl
    ├── dpo/v0.1/val.jsonl
    └── v0.1/
        ├── dataset_info.json
        └── DATASET_CARD.md
```

建议先阅读 `samples.jsonl`，再分别检查 SFT 和 DPO 文件，确认 raw 业务内容、SFT 目标答案以及 DPO 的 `chosen/rejected` 是否合理。小样确认没有问题后，再使用正式的 `phase03.yaml` 生成 500 条数据，并恢复 `--strict-audit`。

---

## 4. 本阶段重点学习内容

### 4.1 了解完整的数据集生成方法

本阶段首先需要理解的，是如何把业务需求转化为一条稳定的数据集生成管线。整个过程不是直接让 AI 自由生成 SFT 和 DPO 文件，而是先定义业务规格和数据合同，让强模型只生成统一的 raw 原始样本，再由程序完成校验、切分和格式转换。

这套方法背后包含几个重要思想：

1. **先把业务规则写清楚，再让 AI 生成。** 提示词中不仅要规定 JSON 格式，还要明确业务类别、具体场景、状态继承、工具结果来源、允许行为和禁止行为。AI 擅长根据清晰规则扩写数据，但不能替代作者决定模糊的业务逻辑。
2. **使用规格卡控制场景覆盖。** 将业务拆成不同类别和更细的场景，为每个场景设置硬约束，再按配置中的数量循环生成，避免 AI 只生成最简单、最常见的样本。
3. **让评估阶段发现的问题进入训练数据。** 阶段二暴露出的工具决策、状态继承、幻觉、确认错误和易混边界等问题，会被转化为难例标签和 DPO 错误类型，使训练数据直接对应模型的真实弱项。
4. **提示词负责生成，程序门禁负责兜底。** AI 输出后还要检查字段、Schema、上下文自洽、业务逻辑、难例比例、场景覆盖和训评隔离。校验失败时要求模型重新生成，不能因为 JSON 看起来完整就直接入库。
5. **只生成一份 raw 数据。** raw 保存业务上下文和标准答案，是整套数据的唯一事实来源。SFT 和 DPO 都从它派生，避免分别生成两套数据后出现业务规则不一致。
6. **使用规则扰动构造 DPO。** DPO 的 `chosen` 直接使用 raw 标准答案，`rejected` 则通过确定性规则制造一种典型业务错误，例如编造技师、选错动作、错误确认或修改字段。错误答案仍然符合 Schema，使模型学习的是业务判断差异，而不是简单识别格式错误。

因此，本阶段最值得掌握的不是某一条生成命令，而是“业务规格 → AI 生成 raw → 程序质量门禁 → SFT/DPO 确定性派生”这一整套数据构造方法。

### 4.2 SFT 和 DPO 数据集的组织格式

LLaMA-Factory 常用的数据组织形式主要有两种：Alpaca 和 ShareGPT。

Alpaca 格式通常使用 `instruction`、`input` 和 `output` 表达一条单轮指令数据，结构简单，适合普通问答、文本改写和单轮指令跟随任务。例如：

```json
{
  "instruction": "抽取预约信息",
  "input": "明天下午三点按摩60分钟",
  "output": "{...}"
}
```

ShareGPT 格式则使用消息列表保存完整对话，并通过角色区分用户、助手、工具调用和工具结果。它还可以独立保存 System Prompt 和工具定义，更适合多轮 Agent 训练。

当前项目选择 ShareGPT，主要原因是模型的真实输入并不是简单的“指令 + 问题”，而是包含：

- 完整的 System Prompt 和当前状态。
- 多轮用户与助手消息。
- Assistant 发起的工具调用。
- 工具返回的 Observation。
- 模型本轮需要生成的 Final 或 Tool Call JSON。

如果使用 Alpaca，就需要把这些不同角色的内容全部拼进一个 `input` 字符串，不仅结构不清晰，也容易造成训练格式与实际部署格式不一致。ShareGPT 可以保留真实消息顺序和角色，因此更符合本项目的 Agent 运行方式。

#### SFT 数据格式

一条简化后的 SFT 数据如下：

```json
{
  "system": "预约 Agent 的业务规则、当前时间和 current_state",
  "tools": "[{\"name\":\"find_technicians\",\"parameters\":{...}}]",
  "conversations": [
    {
      "from": "human",
      "value": "明天下午三点做60分钟肩颈按摩"
    },
    {
      "from": "gpt",
      "value": "{\"action\":\"tool_call\",...}"
    }
  ]
}
```

其中：

- `system` 保存业务规则、输出合同、当前时间和当前状态。
- `tools` 保存模型当前可调用的工具 Schema。
- `conversations` 保存按时间排列的消息。
- `human` 表示用户，`gpt` 表示助手目标输出。
- 多轮工具场景还会使用 `function_call` 和 `observation` 两种角色。

SFT 的目标是让模型在给定 System、工具和对话上下文后，学会生成正确的 Final 或 Tool Call。训练配置使用 `train_on_prompt: false`，只对模型目标回复计算损失，不要求模型学习复述输入 Prompt。

#### DPO 数据格式

DPO 与 SFT 共用相同的 System、工具和对话上下文，但最后不再只有一个标准输出，而是同时提供 `chosen` 和 `rejected`：

```json
{
  "system": "预约 Agent 的业务规则、当前时间和 current_state",
  "tools": "[{\"name\":\"find_technicians\",\"parameters\":{...}}]",
  "conversations": [
    {"from": "human", "value": "用户请求"},
    {"from": "function_call", "value": "{...}"},
    {"from": "observation", "value": "{...}"}
  ],
  "chosen": {
    "from": "gpt",
    "value": "正确的业务输出"
  },
  "rejected": {
    "from": "gpt",
    "value": "格式合法但业务错误的输出"
  }
}
```

`chosen` 来自 raw 中经过校验的标准答案，`rejected` 来自规则扰动。DPO 训练通过比较同一上下文下的正确与错误答案，让模型提高对正确业务决策的偏好。

最后，`dataset_info.json` 会把这些字段和角色注册给 LLaMA-Factory，明确数据采用 `sharegpt` 格式，并配置 `from/value`、`human/gpt`、`function_call/observation` 以及 DPO 的 `chosen/rejected` 映射。这样阶段四可以直接按数据集名称加载并训练。

---

## 5. 面试问题与参考回答
我这里都迫不及待组织面试问题了，因为这块基本上是面试官必问的你是如何创建数据集的。我们后期会专门整理面试问题，还会小小包装一下。但是即使现在，我们把我们目前的方法论来总结一下，其实怎么做出来的数据集这个方法也有模有样的。

### 问题：你的项目是如何构建训练数据集的？

我的项目采用的是“评估集先行、失败难例驱动”的数据集构建方式。

项目首先在训练前建立了一套冻结评估集，使用真实基座模型运行完整的 Agent 流程，再根据评测结果定位模型在工具调用、状态继承、结果确认和事实一致性等方面的失败样本。训练数据不是凭感觉生成，而是重点围绕这些真实失败难例进行补充。

在正式生成数据之前，我先梳理完整的业务流程，明确不同场景下的输入、状态变化、工具调用条件、标准输出和禁止行为，再把这些规则写入数据规格和生成提示词。随后使用强模型生成统一的 raw 原始数据，而不是直接分别生成 SFT 和 DPO 数据。

模型生成后，程序会执行质量门禁，检查字段与 Schema、上下文和业务自洽性、各类数据数量、难例占比、业务场景覆盖以及训练集与评估集是否重叠。在自动检查之外，我还会指定具体业务场景，让 AI 从数据集中找出对应样本，再进行人工抽检。

为了降低批量生成后的返工成本，我会先生成 25 条小规模数据，确认数据结构和业务逻辑没有问题后，再扩大到 500 条。通过门禁和抽检的 raw 数据会按规则转换为 ShareGPT 格式的 SFT 数据；DPO 数据则把 raw 标准答案作为 `chosen`，通过确定性规则构造格式合法但业务错误的 `rejected`。

模型训练完成后，我会继续使用原来的冻结评估集比较训练前后的效果，并收集线上或真实测试中的新增失败样本。经过脱敏、校验和人工审核后，这些真实难例会回流到下一版本训练数据中，形成“评估发现问题、数据针对问题、训练解决问题、真实数据继续回流”的持续迭代闭环。
