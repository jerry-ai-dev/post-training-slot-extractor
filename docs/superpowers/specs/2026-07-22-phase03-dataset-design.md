# 阶段三设计文档 · 训练数据集制作（SFT + DPO）

> 对应 `finetune-spec.md` 3.3，方法细节承接 2.7（数据构造）、2.4（痛点与训练分工）。
> 本次交付范围：**全套数据构造管线 + 小样冒烟**（用 mock / 少量真样本跑通全链路，真实 ~1.5k 全量生成留到确认后再执行）。

---

## 0. 决策摘要（已与用户确认）

| 决策项 | 选定 |
|---|---|
| 交付范围 | 全套管线 + 小样冒烟（不在本次大批量烧 API） |
| 合成模型 | 复用仓库已接的 GPT-5.6-sol（`inference/openai_responses` 后端） |
| 难例占比保证 | 信任强模型自标 `tags`，入库时按 tags 统计、不足报警补生成 |
| 训评隔离 | 指纹去重硬隔离（训练样本 vs `data/eval/test.jsonl` 零重叠） |
| 架构 | 方案 A：分层管线（raw 原始样本 → 规则渲染 SFT / 规则扰动 DPO） |
| DPO 扰动器 | 首版**纯规则**（假名池 + 枚举/布尔/动作翻转 + 确定性时间偏移）；预留 LLM 可选后端，不进首版 |
| LLaMA-Factory 合同版本 | 固定 `v0.9.5`；ShareGPT 字段、角色标签和训练参数均以该版本为准 |

---

## 1. 模块划分与数据流

### 1.1 数据流

```
                      [五类数据规格卡]  ← 人工审核关口（本文档第 2 节，非代码）
                             │ 审核通过后卡片=生成 prompt 模板
                             ▼
  强模型(GPT-5.6-sol)  ──生成──►  raw 原始样本   ──► data/raw/vX.Y/*.jsonl
                                     │
                          ┌──────────┴───────────┐
                    (纯规则渲染)             (纯规则扰动)
                          ▼                       ▼
                SFT ShareGPT 样本       DPO ShareGPT preference 对
                          │                       │
                          ▼                       ▼
              data/processed/sft/         data/processed/dpo/
                          └──────────┬────────────┘
                                     ▼
                        质量校验器（复用 schemas/output.py 等）
                                     ▼
                     train/val 分层切分 + 指纹隔离检查 + 版本登记
                                     ▼
                    dataset_info.json（LLaMA-Factory 注册）
```

**关键洞察**：真正要生成、要人工审核的只有一份「原始数据集」（raw）。SFT 与 DPO 都只是拿 raw 做**机械格式变换**得来的——SFT 把 raw 渲染成 ShareGPT 的 `system + tools + conversations`，DPO 使用相同上下文并把标准答案/规则扰动分别放入 `chosen/rejected`。两者都不新增业务内容。因此人工审核的重心 100% 落在第 2 节的五张原始规格卡。

### 1.2 模块表

| # | 模块 | 位置（拟） | 职责 | 复用 |
|---|---|---|---|---|
| 1 | 生成器 `generator` | `src/slot_extractor/data/generator.py` | 按规格卡组装 prompt，调 GPT-5.6-sol，产出 raw 样本 | `inference/factory.py` |
| 2 | raw 校验器 `raw_validator` | `src/slot_extractor/data/raw_validator.py` | 每条 raw 过 JSON/schema/时间/越界/label-input 自洽校验 | `schemas/output.py`、`schemas/sample.py` |
| 3 | SFT 渲染器 `sft_render` | `src/slot_extractor/data/sft_render.py` | raw → ShareGPT `{system,tools,conversations}`，纯规则 | `prompts/template.py`、`prompts/rules.py` |
| 4 | DPO 扰动器 `dpo_perturb` | `src/slot_extractor/data/dpo_perturb.py` | chosen=raw 标准答案；按 P4/P6/P5/P7/P2P3 规则派生 rejected | `schemas/output.py` |
| 5 | 难例统计 `tag_audit` | `src/slot_extractor/data/tag_audit.py` | 按 tags 统计各类难例占比，不足报警 | — |
| 6 | 隔离检查 `isolation` | `src/slot_extractor/data/isolation.py` | 训练样本指纹 vs `test.jsonl` 指纹去重，重叠即拒 | `data/eval/test.jsonl` |
| 7 | 切分+版本 `dataset_build` | `src/slot_extractor/data/dataset_build.py` | 分层切 train/val、写版本卡、生成 `dataset_info.json` | 汇总以上 |
| — | CLI 编排 | `scripts/data/build_dataset.py` | 一条命令串起 1→7，支持 `--mock` 小样冒烟 | 全部 |

### 1.3 三个真正的难点被显式建模成独立模块

架构只解决工程结构，不解决数据质量。以下三个风险各由一个可独立测试的模块承接：

- **难例占比风险**（强模型可能偷懒生成简单样本却标成难例）→ 模块 5 `tag_audit`，入库统计报警。
- **扰动器正确性风险**（rejected 必须"合法但决策错"，不能是垃圾）→ 模块 4 `dpo_perturb`，验收项要求扰动后的 rejected **仍过 schema 校验**。
- **污染/隔离风险** → 模块 6 `isolation`，指纹硬去重。

### 1.4 模块 4（DPO 扰动器）边界说明

首版**纯规则**，不调 LLM。理由：DPO 的 rejected 是"对 chosen 做一个定向、结构化的破坏"，不是"重新理解语义生成新内容"；规则对"精确制造某一类典型错误"比 LLM 更可控、可复现、免费。自然度需求（如编造的技师名要像真名）用**预置假名池**和**确定性偏移**补足，不需要 LLM。预留一个可选 LLM 后端接口，仅当阶段四发现某类 rejected 太假、DPO 训不动时才对那一小类启用，不进首版。

---

## 2. 五类数据规格卡（文档核心 · 人工审核对象）

> 每张卡含：① 业务情况 ② 要素清单 ③ raw 原始示范 ④ 渲染后的 SFT 示范 ⑤ 派生的 DPO 示范。
> 审核方法：逐类核对「要素清单对不对」+「三条示范对不对」，通过后卡片即作为强模型生成 prompt 模板。

> **格式说明（2026-07-26 决策）**：最终训练格式统一改为 ShareGPT。第 2 节五张卡中原有的 `instruction/input/output/chosen/rejected` 外壳仅保留为便于审核业务内容的“逻辑展开视图”，不再代表落盘 schema；实际落盘必须严格使用第 3 节定义的 `system + tools + conversations` 和 ShareGPT preference 格式。raw 七字段合同不变。

### 2.0 上下文组织与 `current_state` 规则（所有卡共用，先读这节）

> 这节把「模型上下文怎么组织」「`current_state` 是什么、每类怎么取值、怎么进输入」定死，五张卡的示范都遵循它。依据阶段二实际代码：`prompts/rules.py` 的 `SYSTEM_RULES` 与 `prompts/template.py` 的 `PromptBuilder`。

**A. 上下文组织（训练分布必须 = 部署分布）**

阶段二喂给模型的上下文由 `PromptBuilder.build_messages` 组装，结构固定为：
```
system = SYSTEM_RULES + FINAL_SCHEMA_HINT + 工具描述 + "当前时间：{current_time}" + "当前状态：{current_state}"
+ history 消息（user / assistant / assistant.tool_calls → tool 事件，按原顺序）
+ 本轮 user（若有 user_input）
```
SFT/DPO 训练样本必须复用同一套组装，否则训练分布与上线分布不一致。**规则前缀（system）里已明文依赖 `current_state`**：`SYSTEM_RULES` 第一句即「输入包含 current_state、完整消息 history 和本轮用户输入」，并规定「未修改字段**继承 current_state**」。

**B. `current_state` 的语义**

- 定义：模型本次决策前，由调用方提供的**可选当前状态**。它可能是完整对象，也可能为 `null`；`null` 表示调用方尚未保存或本次没有可提供的累积状态。它**不是**本轮的 `expected` 结果，也不要求每次调用前一定存在快照。
- 结构：与 final 的槽位同构（gender_preference/technician_gender/start_time/duration_minutes/preferences/technician_name/technician_status/confirmation/info_complete/unrelated/missing_info），外加 `last_reply_type`（上一轮的 reply_type）；**不含** `action`/`reply`。
- `gender_preference` 只保存用户当前有效的性别筛选要求；`technician_gender` 只保存工具已核实的具体技师性别。工具结果触发前 `technician_gender=null`。
- 首轮通常由用户消息触发，`current_state=null`；信息齐全时该次决策输出 Tool Call。工具返回后才发生下一次模型决策，因此“首轮是工具结果轮”不是常规流程。
- 工具结果触发的决策即使 `current_state=null`，也必须从 History 中最近一组匹配的 `assistant.tool_calls → tool` 恢复查询参数和工具事实；若调用方已物化状态，则该对象必须与最新 Tool Call 参数一致。

**C. 每类样本的 `current_state` 取值规则**

| 类别 | current_state |
|---|---|
| 追问（首轮） | `null`（无前序） |
| 追问（多轮改口） | 非 null：上一轮累积槽位 + `last_reply_type` |
| 工具调用（首轮一步到位） | 可 `null`；若由多轮补齐信息后触发则非 null |
| 最终 JSON（工具结果触发） | 可为 `null`；通常由调用方物化为查询参数状态。无状态时从最新完整工具事件恢复 |
| 确认（type5） | **非 null**：上一轮 `confirm_available` 后的状态（`confirmation:false`, `last_reply_type:confirm_available`） |
| 无关（首轮） | `null` |

**D. `current_state` → 输入上下文的渲染规则**

- raw 的 `current_state` 字段经 `PromptBuilder` 渲染进 system 的 **「当前状态：{紧凑JSON}」** 段。
- 因此 **SFT/DPO 的顶层 `system` 都必须显式体现 `current_state`**。第 2 节逻辑示范为了便于审核仍把它单列在 `input` 行首；实际 ShareGPT 落盘时必须移入 `system`，只出现一次。
- `current_state=null` 时也照常渲染成 `current_state: null`（与部署一致，不省略）。

**E. 防混淆红线：`current_state` vs `expected`**

- `current_state` = 决策**前**的输入快照；`expected` = 决策**后**的输出。
- 此处“输入快照”只描述对象存在时的时间语义，不表示 `current_state` 必须非 null。
- 二者槽位可能不同（如确认卡：`current_state.confirmation=false` → `expected.confirmation=true`）。
- 规则「未修改字段继承 current_state」意味着：`expected` 里未被本轮明确改动的槽位，应与 `current_state` 保持一致。

**F. 字段更新采用最小替换原则**

- 用户明确替换哪个条件，只替换该条件及其直接依赖字段；未提及条件保持不变。
- 更换技师：更新 `technician_name`，并将旧 `technician_gender` 清空为 `null`，等待新工具结果；时间、时长、偏好和 `gender_preference` 默认不变。
- 更换时间：只更新 `start_time`，并清空依赖旧查询的 `technician_gender`/`technician_status`；其他查询条件不变。
- 更换偏好或时长同理：只更新明确修改的字段，并使旧工具结果失效。
- 用户同时明确修改多个条件时，逐项应用上述规则；不得顺带改写未提及字段。

**G. raw 样本字段总览（含 DPO 路由字段 `dpo_targets`）**

一条 raw 样本的字段：`id` / `output_kind` / `conversation_kind` / `tags` / `input`（含 history、current_state、user_input、current_time、available_tools）/ `expected` / **`dpo_targets`**。

> **raw ≠ 评估集**：raw 是训练数据源（SFT 拿 `expected` 当 label、DPO 拿它当 chosen），**不走评分器**，因此**不带** `assertions` 与 `reply_expectations`——那两个字段是 `data/eval/test.jsonl` 评测专用（供 `evaluation/assertions.py` 与 reply 语义评分器），对 raw 是纯冗余。
> 由此定死一条实现决策：`raw_validator`（模块 2）**只复用** `schemas/sample.py` 的历史结构校验逻辑 `_validate_history`（+ `_validate_context`），**不复用**顶层 `sample_from_record`（后者强制要求 `assertions`，final 还强制要求 `reply_expectations`，会误伤只有 7 字段的 raw）。为此在 `sample.py` 把 `_validate_history`/`_validate_context` 抽为公开函数供 raw 与 eval 两侧共用。

- `output_kind` 只表示目标输出结构，取值固定为 `final | tool_call`，必须等于 `expected.action`。
- `conversation_kind` 只表示对话轮数，取值固定为 `single_turn | multi_turn`。当前样本的用户输入也计一轮；History 中的 user 消息数加当前非 null `user_input` 达到 2 时为 `multi_turn`，否则为 `single_turn`。工具消息不单独增加用户轮数。

- `dpo_targets`：字符串数组，声明该 chosen 允许派生哪些痛点的 DPO rejected（如 `["P4","P2P3"]`）；不适用 DPO 则为 `[]`。**全体 raw 都带此字段**，SFT/DPO 共用同一套 raw。
- 取值受「类别白名单」约束、由强模型标注、入库规则校验（详见 3.2.1）。
- 该字段**只用于 DPO 构建路由，不进入模型上下文**（不渲染进 SFT/DPO 的 input）。

### 2.1 追问（信息不全 → final + info_complete=false）

**① 业务情况**：用户表达预约意图，但 `start_time` 或 `duration_minutes` 至少缺一个。触发输入如"我想明天做个肩颈"（缺时长）、"想按60分钟"（缺时间）。期望标准答案：`action=final`，`info_complete=false`，`technician_status=not_checked`，`missing_info` 按规范序列出缺失槽位，`reply_type` 为 `ask_*` 系列，`reply` 是追问话术。不得调工具、不得编造技师。

**② 要素清单**

| 要素 | 规则 |
|---|---|
| 必须有值 | `action=final`, `info_complete=false`, `unrelated=false`, `technician_status=not_checked`, `missing_info`, `reply_type`, `reply` |
| 必须为 null/默认 | 缺失的 `start_time`/`duration_minutes`、`technician_name=null`、`confirmation=false` |
| 已知则填 | 已说的 `duration_minutes`/`preferences`/`gender_preference` 要正确抽出；未查询时 `technician_gender=null` |
| 难例定义 | 多义短词（"那算了""换一个"）、多轮改口、相对时间残缺 |
| tags 取值 | `追问` + 难例标签之一：`相对时间`/`多义短词`/`多轮改口` |
| 难例占比目标 | ≥ 50% |

**③ raw 原始示范**
```json
{
  "id": "raw-ask-0001",
  "output_kind": "final",
  "conversation_kind": "single_turn",
  "tags": ["追问", "相对时间"],
  "dpo_targets": ["P7"],
  "input": {
    "history": [],
    "current_state": null,
    "user_input": "我想明天下午做肩颈按摩，时长还没定",
    "current_time": "2026-06-08 10:00",
    "available_tools": ["find_technicians"]
  },
  "expected": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": null,
    "start_time": null,
    "duration_minutes": null,
    "preferences": ["肩颈"],
    "technician_name": null,
    "technician_status": "not_checked",
    "confirmation": false,
    "info_complete": false,
    "unrelated": false,
    "missing_info": ["start_time", "duration_minutes"],
    "reply_type": "ask_start_time_and_duration",
    "reply": "请问您想什么时候过来，按摩多长时间呢？"
  }
}
```

**④ SFT 逻辑示范**（实际按第 3.1 节渲染为 ShareGPT）

以下按 `instruction/input/output` 逻辑展开，只用于同时审核 system、上下文和目标 JSON；它不是物理落盘格式。实际记录使用第 3.1 节的 `system + tools + conversations`，目标 JSON 是最后一条 `gpt.value`。

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文：首轮无 history，仅当前 user_input；current_state=null（首轮无前序）
  "input": "current_state: null\nhistory: []\nuser_input: 我想明天下午做肩颈按摩，时长还没定\ncurrent_time: 2026-06-08 10:00\navailable_tools:\n  - find_technicians(...)",

  // ③ output = raw 标准答案（信息不全→追问），Response-only loss 只对这一段计损
  "output": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": null,
    "start_time": null,
    "duration_minutes": null,
    "preferences": ["肩颈"],
    "technician_name": null,
    "technician_status": "not_checked",
    "confirmation": false,
    "info_complete": false,
    "unrelated": false,
    "missing_info": ["start_time", "duration_minutes"],
    "reply_type": "ask_start_time_and_duration",
    "reply": "请问您想什么时候过来，按摩多长时间呢？"
  }
}
```

**⑤ DPO 示范**（打 **P7 动作摇摆**）

以下是 DPO 的逻辑展开视图，不是物理落盘格式。实际记录使用第 3.2 节的 `system + tools + conversations + chosen/rejected`；`chosen/rejected` 是 `from=gpt` 的消息对象，业务 JSON 紧凑序列化到各自的 `value`。

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文（同 SFT 的 input，含 current_state=null）
  "input": "current_state: null\nhistory: []\nuser_input: 我想明天下午做肩颈按摩，时长还没定\ncurrent_time: 2026-06-08 10:00\navailable_tools:\n  - find_technicians(...)",

  // ③ chosen = raw 标准答案（正确：信息不全应追问，action=final）
  "chosen": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": null,
    "start_time": null,
    "duration_minutes": null,
    "preferences": ["肩颈"],
    "technician_name": null,
    "technician_status": "not_checked",
    "confirmation": false,
    "info_complete": false,
    "unrelated": false,
    "missing_info": ["start_time", "duration_minutes"],
    "reply_type": "ask_start_time_and_duration",
    "reply": "请问您想什么时候过来，按摩多长时间呢？"
  },

  // ④ rejected = 规则扰动（错误：追问态被改写成抢跑 tool_call，凭空补出缺失槽位）
  "rejected": {
    "action": "tool_call",
    "tool_name": "find_technicians",
    "arguments": {
      "technician_name": null,
      "start_time": "2026-06-09 14:00",
      "duration_minutes": 60,
      "gender_preference": null,
      "preferences": ["肩颈"]
    }
  }
}
```
> 扰动说明：P7 规则=「信息不全的追问态被错误改写成直接 tool_call，并凭空补出 chosen 里根本没有的 `start_time`/`duration_minutes`」。rejected 仍是合法 tool_call JSON（过 schema），只是决策错。

### 2.2 工具调用（信息齐全且无工具结果 → tool_call）

**① 业务情况**：`start_time`+`duration_minutes` 都齐、当前条件下还没有工具结果 → 需查技师。指定技师走 `specific`，未指定走 `search`。

**② 要素清单**

| 要素 | 规则 |
|---|---|
| 必须有值 | `action=tool_call`, `tool_name`, `arguments`（五字段齐） |
| arguments 规则 | `technician_name` 指定则填、未指定 null；`start_time` 归一化 `YYYY-MM-DD HH:MM`；`duration_minutes` 正整数；`gender_preference`/`preferences` 按用户提及抽 |
| 禁止 | 输出 final、编造技师结果 |
| 难例定义 | 该调工具 vs 该追问的边界、相对时间参数（"周末下午""下下周三"） |
| tags 取值 | `工具调用` + `相对时间`/`易混边界` |
| 难例占比目标 | ≥ 60%（最易判错，难例多给） |

**③ raw 原始示范**
```json
{
  "id": "raw-tool-0001",
  "output_kind": "tool_call",
  "conversation_kind": "single_turn",
  "tags": ["工具调用", "相对时间"],
  "dpo_targets": ["P6", "P2P3"],
  "input": {
    "history": [],
    "current_state": null,
    "user_input": "明天下午两点找小王做60分钟肩颈",
    "current_time": "2026-06-08 10:00",
    "available_tools": ["find_technicians"]
  },
  "expected": {
    "action": "tool_call",
    "tool_name": "find_technicians",
    "arguments": {
      "technician_name": "小王",
      "start_time": "2026-06-09 14:00",
      "duration_minutes": 60,
      "gender_preference": null,
      "preferences": ["肩颈"]
    }
  }
}
```

**④ SFT 逻辑示范**（实际按第 3.1 节渲染为 ShareGPT）

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文：信息齐全、尚无工具结果；current_state=null（首轮一步到位）
  "input": "current_state: null\nhistory: []\nuser_input: 明天下午两点找小王做60分钟肩颈\ncurrent_time: 2026-06-08 10:00\navailable_tools:\n  - find_technicians(...)",

  // ③ output = raw 标准答案（信息齐全→调工具查技师）
  "output": {
    "action": "tool_call",
    "tool_name": "find_technicians",
    "arguments": {
      "technician_name": "小王",
      "start_time": "2026-06-09 14:00",
      "duration_minutes": 60,
      "gender_preference": null,
      "preferences": ["肩颈"]
    }
  }
}
```

**⑤ DPO 示范**（打 **P6 工具边界**）

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文（同 SFT 的 input，含 current_state=null）
  "input": "current_state: null\nhistory: []\nuser_input: 明天下午两点找小王做60分钟肩颈\ncurrent_time: 2026-06-08 10:00\navailable_tools:\n  - find_technicians(...)",

  // ③ chosen = raw 标准答案（正确：应发起 tool_call）
  "chosen": {
    "action": "tool_call",
    "tool_name": "find_technicians",
    "arguments": {
      "technician_name": "小王",
      "start_time": "2026-06-09 14:00",
      "duration_minutes": 60,
      "gender_preference": null,
      "preferences": ["肩颈"]
    }
  },

  // ④ rejected = 规则扰动（错误：该调工具却直接终答，凭空塞 technician_status=available）
  "rejected": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": null,
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "小王",
    "technician_status": "available",
    "confirmation": false,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "confirm_available",
    "reply": "小王老师明天14:00有空，需要我为您预约吗？"
  }
}
```
> 可选派生 P2P3 变体：`rejected` 保持 `action:tool_call`、把 `arguments.start_time` 做 +1 天偏移（`2026-06-09 14:00` → `2026-06-10 14:00`，相对时间译错）。

### 2.3 最终 JSON（已有工具结果 → final 落槽）

**① 业务情况**：History 末尾带 `assistant.tool_calls→tool` 事件（工具已返回），据结果映射最终状态。available/unavailable/not_found/no_match 各映射对应 `technician_status`+`reply_type`。

**② 要素清单**

| 要素 | 规则 |
|---|---|
| 必须有值 | 14 字段齐；`info_complete=true`（信息与结果均已就绪时） |
| 事实来源 | `technician_status` 取工具真实返回；`technician_name` 只能来自工具结果 |
| 红线 | 工具结果**列表外**的技师一律 `null`；`reply` 与工具事实一致不矛盾 |
| 难例定义 | 列表外信息须填 null、多槽位一次抽全、工具结果落槽 |
| tags 取值 | `最终JSON` + `工具结果`/`幻觉边界` |
| 难例占比目标 | ~50% |

**③ raw 原始示范**（含 tool 事件，末尾无 user_input，由工具结果触发）
```json
{
  "id": "raw-final-0001",
  "output_kind": "final",
  "conversation_kind": "single_turn",
  "tags": ["最终JSON", "工具结果"],
  "dpo_targets": ["P4"],
  "input": {
    "history": [
      {
        "role": "user",
        "content": "明天两点找王芳做60分钟肩颈"
      },
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "c1",
            "type": "function",
            "function": {
              "name": "find_technicians",
              "arguments": "{\"technician_name\":\"王芳\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"gender_preference\":null,\"preferences\":[\"肩颈\"]}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "{\"mode\":\"specific\",\"status\":\"available\",\"technician\":{\"name\":\"王芳\",\"gender\":\"female\"}}"
      }
    ],
    "current_state": {
      "gender_preference": null,
      "technician_gender": null,
      "start_time": "2026-06-09 14:00",
      "duration_minutes": 60,
      "preferences": ["肩颈"],
      "technician_name": "王芳",
      "technician_status": "not_checked",
      "confirmation": false,
      "info_complete": true,
      "unrelated": false,
      "missing_info": [],
      "last_reply_type": null
    },
    "user_input": null,
    "current_time": "2026-06-08 10:00",
    "available_tools": ["find_technicians"]
  },
  "expected": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": false,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "confirm_available",
    "reply": "王芳老师明天14:00有空，需要我为您预约吗？"
  }
}
```

**④ SFT 逻辑示范**（实际按第 3.1 节渲染为 ShareGPT）

> 要点：由工具结果触发的样本，`conversations` **必须保留** `function_call → observation` 事件前缀（来源是 raw 的 `assistant.tool_calls → tool`），不把工具结果拼进 system 文本。

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文：current_state 为查询时快照；保留 assistant 发起的 tool_call 与 tool 返回事件，末尾无 user_input
  "input": "current_state: {\"gender_preference\":null,\"technician_gender\":null,\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"preferences\":[\"肩颈\"],\"technician_name\":\"王芳\",\"technician_status\":\"not_checked\",\"confirmation\":false,\"info_complete\":true,\"unrelated\":false,\"missing_info\":[],\"last_reply_type\":null}\nhistory:\n  user: 明天两点找王芳做60分钟肩颈\n  assistant[tool_call]: find_technicians({\"technician_name\":\"王芳\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"gender_preference\":null,\"preferences\":[\"肩颈\"]})\n  tool: {\"mode\":\"specific\",\"status\":\"available\",\"technician\":{\"name\":\"王芳\",\"gender\":\"female\"}}\ncurrent_time: 2026-06-08 10:00",

  // ③ output = raw 标准答案（读工具结果落槽为 final）
  "output": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": false,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "confirm_available",
    "reply": "王芳老师明天14:00有空，需要我为您预约吗？"
  }
}
```

**⑤ DPO 示范**（打 **P4 幻觉**，DPO 主力）

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文（同 SFT 的 input，含 current_state 查询快照 + tool_call/tool 事件）
  "input": "current_state: {\"gender_preference\":null,\"technician_gender\":null,\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"preferences\":[\"肩颈\"],\"technician_name\":\"王芳\",\"technician_status\":\"not_checked\",\"confirmation\":false,\"info_complete\":true,\"unrelated\":false,\"missing_info\":[],\"last_reply_type\":null}\nhistory:\n  user: 明天两点找王芳做60分钟肩颈\n  assistant[tool_call]: find_technicians({\"technician_name\":\"王芳\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"gender_preference\":null,\"preferences\":[\"肩颈\"]})\n  tool: {\"mode\":\"specific\",\"status\":\"available\",\"technician\":{\"name\":\"王芳\",\"gender\":\"female\"}}\ncurrent_time: 2026-06-08 10:00",

  // ③ chosen = raw 标准答案（正确：只落工具真实返回的王芳）
  "chosen": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": false,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "confirm_available",
    "reply": "王芳老师明天14:00有空，需要我为您预约吗？"
  },

  // ④ rejected = 规则扰动（错误：technician_name 换成工具结果外的假名"李娜"，编造事实）
  "rejected": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "李娜",
    "technician_status": "available",
    "confirmation": false,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "confirm_available",
    "reply": "李娜老师明天14:00有空，需要我为您预约吗？"
  }
}
```
> 另一种 P4 扰动：保持 `technician_name` 不变，把 `technician_status` 从 `available` 改成 `not_found` 却仍报可约（状态与话术自相矛盾）。

### 2.4 确认（type5：用户对已呈现结果做确认/拒绝/暂缓）

**① 业务情况**：上一轮已 `confirm_available`，用户回"好/就他吧"（确认）、"算了/不用了"（拒绝→暂缓）、"换个时间"（改约）。

**② 要素清单**

| 要素 | 规则 |
|---|---|
| 必须有值 | `confirmation` 正确置位；`unrelated=false` |
| 映射 | 确认→`reply_type=booking_authorized`；拒绝/暂缓→`appointment_paused` |
| 难例定义 | confirmation vs unrelated 易混、多义短词 |
| tags 取值 | `确认` + `易混边界`/`多义短词` |
| 难例占比目标 | ≥ 60% |

**③ raw 原始示范**（多轮，前序 history 完整回放 `assistant.tool_calls → tool → confirm_available` 链路，末轮 `user_input="就他吧"`）
```json
{
  "id": "raw-confirm-0001",
  "output_kind": "final",
  "conversation_kind": "multi_turn",
  "tags": ["确认", "多义短词"],
  "dpo_targets": ["P5"],
  "input": {
    "history": [
      {
        "role": "user",
        "content": "明天两点找王芳做60分钟肩颈"
      },
      {
        "role": "assistant",
        "content": null,
        "tool_calls": [
          {
            "id": "c1",
            "type": "function",
            "function": {
              "name": "find_technicians",
              "arguments": "{\"technician_name\":\"王芳\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"gender_preference\":null,\"preferences\":[\"肩颈\"]}"
            }
          }
        ]
      },
      {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "{\"mode\":\"specific\",\"status\":\"available\",\"technician\":{\"name\":\"王芳\",\"gender\":\"female\"}}"
      },
      {
        "role": "assistant",
        "content": "王芳老师明天14:00有空，需要我为您预约吗？"
      }
    ],
    "current_state": {
      "gender_preference": null,
      "technician_gender": "female",
      "start_time": "2026-06-09 14:00",
      "duration_minutes": 60,
      "preferences": ["肩颈"],
      "technician_name": "王芳",
      "technician_status": "available",
      "confirmation": false,
      "info_complete": true,
      "unrelated": false,
      "missing_info": [],
      "last_reply_type": "confirm_available"
    },
    "user_input": "就他吧",
    "current_time": "2026-06-08 10:00",
    "available_tools": ["find_technicians"]
  },
  "expected": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": true,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "booking_authorized",
    "reply": "好的，已为您锁定王芳老师明天14:00的肩颈按摩。"
  }
}
```

**④ SFT 逻辑示范**（实际按第 3.1 节渲染为 ShareGPT）

以下按 `instruction/input/output` 逻辑展开，只用于审核业务内容；实际记录使用第 3.1 节的 `system + tools + conversations`，目标 JSON 是最后一条 `gpt.value`。

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文：current_state 为上一轮 confirm_available 后的决策前快照（confirmation:false）；history 完整保留 tool_call/tool 事件
  "input": "current_state: {\"gender_preference\":null,\"technician_gender\":\"female\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"preferences\":[\"肩颈\"],\"technician_name\":\"王芳\",\"technician_status\":\"available\",\"confirmation\":false,\"info_complete\":true,\"unrelated\":false,\"missing_info\":[],\"last_reply_type\":\"confirm_available\"}\nhistory:\n  user: 明天两点找王芳做60分钟肩颈\n  assistant[tool_call]: find_technicians({\"technician_name\":\"王芳\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"gender_preference\":null,\"preferences\":[\"肩颈\"]})\n  tool: {\"mode\":\"specific\",\"status\":\"available\",\"technician\":{\"name\":\"王芳\",\"gender\":\"female\"}}\n  assistant: 王芳老师明天14:00有空，需要我为您预约吗？\nuser_input: 就他吧\ncurrent_time: 2026-06-08 10:00",

  // ③ output = raw 标准答案（确认成立），Response-only loss 只对这一段计损
  "output": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": true,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "booking_authorized",
    "reply": "好的，已为您锁定王芳老师明天14:00的肩颈按摩。"
  }
}
```

**⑤ DPO 示范**（打 **P5 意图边界**）

以下是 DPO 的逻辑展开视图，便于逐字段审核 chosen/rejected；实际落盘按第 3.2 节使用 `system + tools + conversations`，并将 chosen/rejected 写成 `from=gpt` 的消息对象，其 `value` 是紧凑 JSON 字符串。

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文：current_state 为上一轮 confirm_available 后的决策前快照（confirmation:false）；history 完整保留 tool_call/tool 事件
  "input": "current_state: {\"gender_preference\":null,\"technician_gender\":\"female\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"preferences\":[\"肩颈\"],\"technician_name\":\"王芳\",\"technician_status\":\"available\",\"confirmation\":false,\"info_complete\":true,\"unrelated\":false,\"missing_info\":[],\"last_reply_type\":\"confirm_available\"}\nhistory:\n  user: 明天两点找王芳做60分钟肩颈\n  assistant[tool_call]: find_technicians({\"technician_name\":\"王芳\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"gender_preference\":null,\"preferences\":[\"肩颈\"]})\n  tool: {\"mode\":\"specific\",\"status\":\"available\",\"technician\":{\"name\":\"王芳\",\"gender\":\"female\"}}\n  assistant: 王芳老师明天14:00有空，需要我为您预约吗？\nuser_input: 就他吧\ncurrent_time: 2026-06-08 10:00",

  // ③ chosen = raw 标准答案（正确：识别为确认）
  "chosen": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": true,
    "info_complete": true,
    "unrelated": false,
    "missing_info": [],
    "reply_type": "booking_authorized",
    "reply": "好的，已为您锁定王芳老师明天14:00的肩颈按摩。"
  },

  // ④ rejected = 规则扰动（错误：把"就他吧"误判为闲聊 unrelated=true）
  "rejected": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": "female",
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "preferences": ["肩颈"],
    "technician_name": "王芳",
    "technician_status": "available",
    "confirmation": false,
    "info_complete": true,
    "unrelated": true,
    "missing_info": [],
    "reply_type": "handoff",
    "reply": null
  }
}
```

### 2.5 无关/拒答（unrelated）

**① 业务情况**：非预约相关输入（闲聊、问天气、诱导触发工具的越界问题）。

**② 要素清单**

| 要素 | 规则 |
|---|---|
| 必须有值 | `action=final`, `unrelated=true`, `reply_type=handoff`, `reply=null`（handoff 专属） |
| 其余 | 所有槽位 null/默认；`technician_status=not_checked` |
| 禁止 | 触发工具 |
| 难例定义 | 看似相关实则越界、诱导触发 |
| tags 取值 | `无关` + `越界诱导` |
| 难例占比目标 | ~50% |

**③ raw 原始示范**
```json
{
  "id": "raw-unrelated-0001",
  "output_kind": "final",
  "conversation_kind": "single_turn",
  "tags": ["无关", "越界诱导"],
  "dpo_targets": ["P6"],
  "input": {
    "history": [],
    "current_state": null,
    "user_input": "你们店的按摩椅是什么牌子的",
    "current_time": "2026-06-08 10:00",
    "available_tools": ["find_technicians"]
  },
  "expected": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": null,
    "start_time": null,
    "duration_minutes": null,
    "preferences": [],
    "technician_name": null,
    "technician_status": "not_checked",
    "confirmation": false,
    "info_complete": false,
    "unrelated": true,
    "missing_info": [],
    "reply_type": "handoff",
    "reply": null
  }
}
```

**④ SFT 逻辑示范**（实际按第 3.1 节渲染为 ShareGPT）

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文：与预约无关的越界提问；current_state=null（首轮）
  "input": "current_state: null\nhistory: []\nuser_input: 你们店的按摩椅是什么牌子的\ncurrent_time: 2026-06-08 10:00\navailable_tools:\n  - find_technicians(...)",

  // ③ output = raw 标准答案（判为无关，handoff 且 reply=null）
  "output": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": null,
    "start_time": null,
    "duration_minutes": null,
    "preferences": [],
    "technician_name": null,
    "technician_status": "not_checked",
    "confirmation": false,
    "info_complete": false,
    "unrelated": true,
    "missing_info": [],
    "reply_type": "handoff",
    "reply": null
  }
}
```

**⑤ DPO 示范**（打 **P6**；本例 user_input 无可抽槽位，故不适用 P5，`dpo_targets` 只标 `P6`）

```jsonc
{
  // ① 规则前缀（system），与上线 prompt 同源
  "instruction": "你是预约信息抽取助手，只输出 JSON……",

  // ② 对话上下文（同 SFT 的 input，含 current_state=null）
  "input": "current_state: null\nhistory: []\nuser_input: 你们店的按摩椅是什么牌子的\ncurrent_time: 2026-06-08 10:00\navailable_tools:\n  - find_technicians(...)",

  // ③ chosen = raw 标准答案（正确：判为无关，不触发工具）
  "chosen": {
    "action": "final",
    "gender_preference": null,
    "technician_gender": null,
    "start_time": null,
    "duration_minutes": null,
    "preferences": [],
    "technician_name": null,
    "technician_status": "not_checked",
    "confirmation": false,
    "info_complete": false,
    "unrelated": true,
    "missing_info": [],
    "reply_type": "handoff",
    "reply": null
  },

  // ④ rejected = 规则扰动（错误：被越界提问诱导，unrelated=false 且抢跑 tool_call）
  "rejected": {
    "action": "tool_call",
    "tool_name": "find_technicians",
    "arguments": {
      "technician_name": null,
      "start_time": "2026-06-08 10:00",
      "duration_minutes": 60,
      "gender_preference": null,
      "preferences": []
    }
  }
}
```

---

## 3. 转换方法（写死规则）

### 3.1 raw → SFT（ShareGPT，纯规则渲染）

复用 `prompts/template.py` 的 `PromptBuilder`，把 raw 反序列化成 `Sample` 后生成：

```json
{
  "system": "规则 + output schema + 当前时间 + 当前状态",
  "tools": "[{\"name\":\"find_technicians\",\"description\":\"...\",\"parameters\":{...}}]",
  "conversations": [
    {"from": "human", "value": "历史/当前用户消息"},
    {"from": "gpt", "value": "历史助手自然语言回复"},
    {"from": "function_call", "value": "历史 assistant.tool_calls 的 JSON"},
    {"from": "observation", "value": "历史 tool 结果 JSON"},
    {"from": "gpt", "value": "raw.expected 的紧凑 JSON"}
  ]
}
```

1. **system**：`SYSTEM_RULES` + `FINAL_SCHEMA_HINT` + `TOOL_SCHEMA_HINT` + `当前时间` + `当前状态`。`current_state` 不是第五种消息角色；它由调用方维护，只在 system 中出现一次。
2. **tools**：根据 raw 的 `available_tools` 从固定工具注册表生成 OpenAI-compatible JSON Schema，并将整个工具数组序列化为字符串；无可用工具时固定为字符串 `"[]"`。具体工具描述不再在 system 重复一份，避免同一 schema 出现两次。
3. **conversations**：history 保持原顺序；`user→human`、自然语言 `assistant→gpt`、`assistant.tool_calls→function_call`、`tool→observation`；末尾 `user_input`（若有）追加为 `human`。遵守 LLaMA-Factory 的位置合同：`human/observation` 在奇数位，`gpt/function_call` 在偶数位；不满足时拒绝构建。
4. **目标回复**：raw 的 `expected` 紧凑序列化后追加为最后一条 `from=gpt`。历史工具事件不得压成普通文本，也不得移动到 system。
5. **loss mask**：LLaMA-Factory 配 `train_on_prompt=false` 与 `mask_history=true`，屏蔽 system/user/历史回复，只对最后一个目标决策计算 loss。

> **不要混淆两种 tool call**：raw history 中已经发生过的 OpenAI 工具事件映射为 `function_call → observation`；但本项目当前轮的训练目标始终是业务合同定义的 JSON 对象。即使 `expected.action="tool_call"`，它也必须作为最后一条 `from=gpt` 的 value 训练，不能改成 ShareGPT 的 `function_call` 目标，否则会破坏现有 `parse_model_json`、评估器和部署输出合同。

### 3.2 raw → DPO（ShareGPT preference，纯规则扰动）

DPO 行复用 SFT 的 `system`、`tools` 和不含最后目标回复的 `conversations`；`chosen={"from":"gpt","value":"<expected 紧凑 JSON>"}`，`rejected` 使用相同消息对象结构，value 由扰动器按**痛点**确定性派生。本节把「怎么路由」「每个 P 的完整算法」「标注正确性怎么保证」全部写死到可执行级。

#### 3.2.1 痛点路由：raw 显式带 `dpo_targets` 字段（路线 A）

**程序不猜。每条 raw 自带一个 `dpo_targets` 字段**，值为该 chosen 允许派生的痛点列表（如 `["P4","P2P3"]`；不适用 DPO 则为 `[]`）。扰动器读该字段路由，对列表里的**每一个** P 各派生一条独立偏好对（同 chosen、不同 rejected）。
> 痛点代号在全文统一为 `P4`/`P6`/`P5`/`P7`/`P2P3` 五个 token；`P2P3` 是时间(P2)+字段(P3)难例合并成的单一 token，`dpo_targets` 里只能写 `P2P3`，不得写 `P2` 或 `P3`。

- **谁标**：生成 raw 时由强模型标注，从「该类白名单」中选（下表）。
- **正确性三重保证**：① 强模型只能在该类**合法候选集**里选（下表白名单，超出即非法）；② 入库时规则校验 `dpo_targets ⊆ 白名单[该类]`，越界报错丢弃；③ 派生出的每条 `rejected` 必须过 schema 校验且 `!= chosen`，否则该痛点扰动失败、报警。
- **覆盖范围**：**全体 raw 都标 `dpo_targets`**（不适用则 `[]`），SFT 与 DPO 共用同一套 raw；DPO 构建时从 `dpo_targets` 非空的 raw 里按 4.2 痛点配比抽样，凑够 ~400 对。

**类别 → 合法 `dpo_targets` 白名单**（生成时只能从对应行选）：

| 类别 | 允许的 dpo_targets | 说明 |
|---|---|---|
| 追问 | `P7` | 追问态被改写成抢跑 tool_call |
| 工具调用 | `P6`, `P2P3` | 该调却抢答；或工具参数时间/字段抽错 |
| 最终 JSON | `P4`, `P2P3` | 编造工具结果外技师（主力）；或落槽时间抽错 |
| 确认 | `P5` | 多义短词误判 confirmation/unrelated |
| 无关 | `P5`, `P6` | 被越界提问诱导触发工具 |

> 偏好对总数 = Σ(各 raw 的 `dpo_targets` 长度)，再按 4.2 配比抽样到 ~400。

#### 3.2.2 每个 P 的完整扰动算法（照此实现）

每条算法格式：**前提**（该 P 适用于什么 chosen）→ **改哪些字段** → **连带同步** → **边界/失败处理**。

**P4 幻觉（前提：chosen 是读工具结果的 final，technician_name 来自工具或为 null）**
- 改：`technician_name` ← 从**预置假名池**取一个「不在当前 tool 消息结果中」的名字（如"李娜"）。
- 连带：`reply` 里的技师称呼同步改成该假名（保持话术模板不变，只换名字）。
- 变体（可选，二选一随机）：保持 `technician_name` 不变，把 `technician_status` 从真实值翻转成矛盾值（`available`→`not_found` 但 reply 仍报可约）。
- 边界：**预置假名池**为固定清单（落在 `src/slot_extractor/data/fake_names.py`，与真实技师库零交集）；取名时必须排除本样本工具结果里出现过的所有名字；若该 final 的 technician_name 本就该是 null（`no_match`/`unavailable`/`not_found`），则只用 status 翻转变体。

**P6 工具边界（前提：chosen 是 tool_call；或无关类 chosen 是 final/unrelated=true）★主扰动**
> 依据 baseline 真实失败：4B 最高频错误就是「该 tool_call 却输出 final」（specific/search 的 *-call、*-retry 样本全中），并连带五字段 `wrong_argument`。故 P6 的 tool_call→抢答 final 为**主扰动**，须覆盖 specific 与 search 两种触发。
- tool_call 情形：把整条 `{action:tool_call,...}` 改写成一个 `action:final` 骨架，`technician_status` 置 `available` 并凭空补出"确认话术"（抢答）。
- 无关情形：把 `unrelated:true` 改成 `false`，`action:final` 改写成 `action:tool_call`，`arguments` 用 user_input 里能凑的字段；user_input 凑不出时用确定性兜底（`start_time` 取 `current_time`、`duration_minutes` 取默认 `60`、`technician_name`/`gender_preference` 为 `null`、`preferences` 为 `[]`），保证过 tool_call schema。
- 连带：final↔tool_call 切换时，字段集合必须整体换成目标 action 的 schema（final 14 字段 / tool_call 3 字段），不能混。
- tool_call→抢答 final 时补全：`reply_type` 置 `confirm_available`、`reply` 补一句确认话术，其余槽位从原 tool_call 的 `arguments` 平移。
- 边界：改写后必须过对应 action 的 schema 校验。

**P5 意图边界（前提：chosen 是确认类或无关类 final）**
- 子规则 P5-a（布尔翻转）：确认类把 `confirmation` 判反，或把 `unrelated:false`→`true`（把"就他吧"当闲聊）；连带 `reply_type`/`reply` 跟随（如误判 unrelated 后 `reply_type:handoff, reply:null`）。
- 子规则 P5-b（**确认态回退追问态**，依据 baseline 真实失败新增）：把 chosen 的确认态 `reply_type`（`booking_authorized`/`appointment_paused`/`acknowledge_result`）改成 `ask_start_time`（或其他 ask_*），并把 `confirmation` 改回 `false`——复刻 1.7B「把『先不了』『知道了』误判成追问」的真实高频错误。
  > 真实证据：confirm-reject-01「先不了」→ 模型输出 `ask_start_time`（应为 `appointment_paused`）；specific-unavailable-01-confirm「知道了」→ 输出 `ask_start_time`（应为 `acknowledge_result`）。
- 子规则 P5-c（**无关类误判为预约**）：无关类 chosen（`unrelated:true`）把 `unrelated` 改成 `false` 且**保持 `action:final`**（不改成 tool_call，以与 P6 区分），按 user_input 勉强能抽的槽位填入，`reply_type` 改成 `ask_*`（信息不全）或 `confirm_available`、`reply` 给对应话术；若 user_input 完全无可抽槽位（如"你们店按摩椅什么牌子"），则该 raw 不适用 P5，生成时不给它标 `P5`（该越界诱导交由 P6 覆盖）。
- 边界：翻转/回退后整条仍须是合法 final。

**P7 动作摇摆（前提：chosen 是追问态 final，info_complete=false）**
- 改：`action:final` → `action:tool_call`，凭空补出 chosen 里缺失的 `start_time`/`duration_minutes`（抢跑）。
- 连带：换成 tool_call 的 3 字段 schema，丢弃 final 专属字段。
- 边界：补出的槽位用确定性默认值——`start_time`：user_input 含相对日期（"明天""周末"）则解析该日 14:00，否则取 `current_time` 当天 14:00；`duration_minutes`：缺失则补默认 `60`。保证过 tool_call schema。

**P2P3 难例（前提：chosen 含具体 start_time 或关键槽位）**
- 改：对 `start_time`（tool_call 的 arguments 或 final）做**确定性偏移**（`+1 天` 或 `整点 ±60 分钟`），或把某个关键槽位抽错成确定性邻近值（`duration_minutes` 做 `±30` 分钟、`gender_preference`/`technician_gender` 在 `female`/`male` 间翻转）；每条 rejected 只扰动一个目标字段。
- 连带：仅改目标字段，其余不动。
- 边界：偏移后仍须是合法 `YYYY-MM-DD HH:MM` / 合法枚举。
- **null 字段边界**：枚举翻转（`gender_preference`/`technician_gender`）仅当该字段在 chosen 中**非 null** 时才可执行；若目标字段为 `null`（示范多为 null），翻转无定义，扰动器必须**回退到 `start_time` 偏移**。若连 `start_time` 也不可扰动（如追问态缺 start_time），则该痛点扰动失败、报警并丢弃该对（与 3.2 验收红线 `rejected != chosen` 一致，禁止产出与 chosen 相同的 rejected）。

**验收红线**：每条 `rejected` 必须**仍过 `schemas/output.py` 校验**（是"合法但决策错误的 JSON"），且 `rejected != chosen`；任一不满足则该痛点扰动失败、报警并丢弃该对。P1（JSON 格式）不做 DPO，SFT 已能收敛。

### 3.3 LLaMA-Factory 注册合同

阶段三生成的 `dataset_info.json` 必须使用以下映射；不允许在实现时退回 Alpaca columns。合同基线固定为 LLaMA-Factory `v0.9.5`（官方 2026-05-30 发布），升级版本必须先重跑格式加载和阶段四空跑，不得静默漂移：

```json
{
  "phase03_sft_v0_1": {
    "file_name": "sft/v0.1/train.jsonl",
    "formatting": "sharegpt",
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "function_tag": "function_call",
      "observation_tag": "observation"
    }
  },
  "phase03_dpo_v0_1": {
    "file_name": "dpo/v0.1/train.jsonl",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools",
      "chosen": "chosen",
      "rejected": "rejected"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "function_tag": "function_call",
      "observation_tag": "observation"
    }
  },
  "phase03_dpo_val_v0_1": {
    "file_name": "dpo/v0.1/val.jsonl",
    "formatting": "sharegpt",
    "ranking": true,
    "columns": {
      "messages": "conversations",
      "system": "system",
      "tools": "tools",
      "chosen": "chosen",
      "rejected": "rejected"
    },
    "tags": {
      "role_tag": "from",
      "content_tag": "value",
      "user_tag": "human",
      "assistant_tag": "gpt",
      "function_tag": "function_call",
      "observation_tag": "observation"
    }
  }
}
```

训练时 `template` 取目标基座模型对应的 LLaMA-Factory chat template；数据文件不手写 Qwen/InternLM 特殊 token。SFT 配置固定 `train_on_prompt: false`、`mask_history: true`，使每个决策点只监督最后一条 `gpt` 目标消息。

**部署一致性硬约束**：上述 `tools` 不是只为通过 LLaMA-Factory 校验而增加的冗余字段。阶段四对微调模型评估、阶段五 llama.cpp 部署时，必须把同一份 OpenAI-compatible tools schema 交给对应 chat template；不得训练时用顶层 `tools`、推理时又退回 system 内的手写工具清单。阶段三只负责产出训练数据和工具注册表，运行时 Backend 的结构化 tools 适配在阶段四空跑前完成并做 token/template 对齐测试。

---

## 4. 配比、规模、划分与版本

### 4.1 SFT 类别配比（按任务分，低频类过采样）—— spec 表一

| 任务类别 | 首版条数 | 占比 | 类内难例占比目标 |
|---|---:|---:|---:|
| 追问（信息不全） | 300 | ~21% | ≥ 50% |
| 工具调用（查技师） | 300 | ~21% | ≥ 60% |
| 最终预约 JSON | 300 | ~21% | ~50% |
| 确认（type5） | 250 | ~18% | ≥ 60% |
| 无关/拒答 | 250 | ~18% | ~50% |
| **合计** | **~1,400** | **100%** | — |

### 4.2 DPO 偏好对配比（按痛点分，向幻觉/边界倾斜）—— spec 表三

| 痛点 | 占比 | 目标对数（按 ~400） |
|---|---:|---:|
| P4 幻觉（主力） | ~40% | ~160 |
| P6 工具调用边界 | ~20% | ~80 |
| P5 意图理解边界 | ~20% | ~80 |
| P7 动作决策摇摆 | ~15% | ~60 |
| P2P3 难例（时间/字段） | ~5% | ~20 |
| P1 JSON 格式 | 0% | 0（SFT 收敛） |
| **合计** | **100%** | **~400（300~500 区间）** |

### 4.3 划分与版本

- **train/val 分层切分 9:1**：按 4.1 的五类比例**分层切**，保证 val 每类都有足够样本单独算分。val 仅用于训练期监控 loss/分类别指标/early stop，**不等于** `data/eval/test.jsonl`。
- **版本管理**：数据集打版本号 `v0.1`，落盘 `data/raw/v0.1/`、`data/processed/{sft,dpo}/v0.1/`，写版本卡记录来源构成、各类实际配比、生成脚本 commit 哈希。
- **数据增强**（只动输入不动 label）：工具描述措辞泛化、候选列表数量/顺序/姓名随机化、口语噪声注入、历史长度泛化。红线：任何会改变正确答案的"增强"都是脏数据。

### 4.4 小样冒烟范围（本次交付）

- 每类生成 **5 条** raw（合计 25 条）走通 生成→校验→渲染→扰动→切分→注册 全链路；
- MockBackend 离线可跑（CI 零依赖）；GPT-5.6-sol 真机跑一次 25 条验证真连真出；
- 真实 ~1,400 全量生成留到本管线验收通过、用户确认后再执行。

---

## 5. 校验、隔离、测试与 DoD

### 5.1 三级质量校验

1. **raw 校验**（模块 2）：JSON 可解析、14 字段/tool schema 合法（复用 `output.py`）、时间格式 `YYYY-MM-DD HH:MM`、history 结构合法（复用 `sample.py`）、label 与 input 自洽。**注意 `output.py` 只做逐字段校验、不校验跨字段联动**，因此 chosen 的以下一致性必须在本步单独校验：`missing_info` 与实际缺失槽位一致；`technician_name` 只能来自 tool 结果或为 `null`（列表外技师不得填入）；`unrelated=true` ⇒ 所有预约槽位为 null/默认且 `reply_type=handoff,reply=null`；`confirmation=true` ⇒ `info_complete=true,missing_info=[]`；`technician_status` 与 `technician_name`/`reply_type` 对应（`not_found`/`no_match` ⇒ `technician_name=null`）；`info_complete` 仅由 `start_time`+`duration_minutes` 是否齐全决定。
2. **SFT 渲染校验**：`system` 非空；`tools` 可解析为 JSON 数组且与 `available_tools` 一致；`conversations` 角色序列及奇偶位合法；最后一条必须为 `gpt` 且其 value 过 `output.py`；工具事件必须保持 `function_call → observation` 配对。
3. **DPO 扰动校验**：`system/tools/conversations` 与对应 SFT 共同上下文完全相同；`conversations` 不含当前目标；`chosen/rejected` 均为 `from=gpt` 的消息对象，二者 value 均过 `output.py` 且不相等；rejected 命中声明的痛点类型。

### 5.2 训评隔离（指纹硬去重）

- 对每条训练样本与 `data/eval/test.jsonl` 的每条，计算**规范化文本指纹**（history + user_input + current_time 归一后 hash）。
- 训练样本指纹若命中评测集指纹 → **拒绝入库并报错**，保证 DoD「零重叠」。

### 5.3 难例占比审计（模块 5）

- 入库时按 `tags` 统计每类实际难例占比，低于 4.1 目标 → 报警并列出缺口，提示补生成。

### 5.4 测试

- 单元测试覆盖：SFT 渲染器（含 tool 事件前缀保留）、DPO 五类扰动器（含 rejected 必过 schema、chosen≠rejected）、隔离指纹去重、tag_audit 报警。
- 集成测试：`build_dataset.py --mock` 端到端跑通 25 条小样，产出 train/val/dpo + dataset_info.json。
- `ruff check .` + `pytest`（`not local_backend` 全绿）。

### 5.5 交付线（DoD）

- [ ] `data/processed/{sft,dpo}/v0.1/{train,val}.jsonl`（小样规模）产出；
- [ ] 全部样本通过三级质量校验；
- [ ] 与 `data/eval/test.jsonl` 指纹零重叠（隔离检查通过）；
- [ ] `dataset_info.json` 按 ShareGPT role/content tags 注册 SFT/DPO 两个数据集；训练配置为 `train_on_prompt=false`、`mask_history=true`；
- [ ] 版本卡 `v0.1` 登记（来源构成/配比/脚本 commit）；
- [ ] `--mock` 冒烟绿、真机 25 条真连真出验证；
- [ ] 阶段三 `project-log/phase-03-dataset/log.md` 更新。

---

## 6. 不在本阶段范围

- 真实 ~1,400 SFT / ~400 DPO 全量生成（管线验收后单独执行）；
- LLaMA-Factory 实际训练（阶段四）；
- 失败 case 回流（阶段六数据闭环）；
- DPO 扰动器的 LLM 可选后端（仅预留接口）。
