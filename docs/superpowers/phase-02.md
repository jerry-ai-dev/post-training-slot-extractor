# 阶段二：冻结评估集与本地 Baseline

## 1. 阶段目标

阶段二把阶段一的通用 Harness 落到按摩预约 Agent 的真实业务合同上，形成一把稳定的评估尺子，并使用本地 Qwen3 模型建立训练前 Baseline。

本阶段最终交付包括：

- 一份当前为 51 条的协议回归评估集；
- 一份机器可读的数据合同；
- 严格的 Tool Call 与 Final 输出协议；
- 确定性的协议和任务正确性评分；
- 场景切片、错误标签和性能统计；
- 三个本地模型的正式 Baseline 报告。

## 2. 运行时任务

模型每轮接收：

```text
完整消息历史 + 当前结构化状态
+ 当前用户输入 + 当前时间 + 可用工具
```

模型必须输出以下二者之一：

```text
Tool Call：信息完整且需要查询技师
Final：继续追问、展示工具结果、确认、拒绝、暂缓或转人工
```

评估对象不是单句槽位抽取器，而是能够维护状态、选择动作并生成一致回复的预约 Agent。

## 3. 数据集

### 3.1 文件

- 样本：`data/eval/test.jsonl`
- 数据合同：`data/eval/dataset_contract.json`
- 数据说明：`data/eval/DATASET_CARD.md`
- 校验和：`data/eval/test.sha256`

评估集与训练数据严格隔离。`test.sha256` 对当前正式样本做 SHA-256 冻结，仓库不保留历史校验文件或迁移备份。

### 3.2 输入结构

每条样本的 `input` 包含：

- `history`：用户、助手、工具调用和工具结果组成的完整事件历史；
- `current_state`：上一轮已确认的结构化状态；
- `user_input`：当前用户输入；
- `current_time`：相对时间解析基准；
- `available_tools`：当前允许调用的工具。

History 使用 `assistant.tool_calls` 与对应的 `tool` 消息承载工具事件；结构化状态仍通过 `current_state` 独立传入。

### 3.3 场景覆盖

51 条样本覆盖：

- 信息缺失与追问；
- 完整信息下的技师查询；
- 指定技师可用、不可用和未找到；
- 条件搜索匹配与无匹配；
- 工具结果展示；
- 用户确认、拒绝和暂缓；
- 多轮槽位继承、覆盖和工具结果失效；
- 无关输入转人工。

其中 36 条期望 `final`，15 条期望 `tool_call`。下方 Baseline 已按当前 51 条数据集重新运行。

## 4. 输出合同

### 4.1 Tool Call

Tool Call 必须严格包含：

```json
{
  "action": "tool_call",
  "tool_name": "find_technicians",
  "arguments": {
    "technician_name": null,
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "gender": null,
    "preferences": ["肩颈"]
  }
}
```

动作、工具名和参数分别计入任务正确性。信息不完整时不得调用工具；信息完整且没有仍然有效的工具结果时应调用工具。

### 4.2 Final

Final 必须严格包含 13 个字段：`action`、`gender`、`start_time`、`duration_minutes`、`preferences`、`technician_name`、`technician_status`、`confirmation`、`info_complete`、`unrelated`、`missing_info`、`reply_type` 和 `reply`。

`reply_type` 表示回复承担的业务动作，`reply` 必须与结构化状态和工具事实一致。无关输入使用完整的 Final 合同转人工，不省略字段。

## 5. 状态与工具规则

状态更新遵循以下原则：

1. 当前用户明确提供的值覆盖旧值；
2. 未被修改的有效字段继续继承；
3. 不得从无证据文本中猜测姓名、性别、时间、时长或偏好；
4. 姓名、时间、时长、性别或偏好变化后，旧工具结果失效；
5. 工具结果只能映射到其明确返回的技师和可用状态；
6. 用户确认只在已有可确认方案时成立；
7. 相对时间以 `current_time` 为基准归一化。

## 6. 评分设计

### 6.1 协议遵循

`protocol` 是二元评分：输出必须能解析为 JSON，并严格满足对应动作的 Schema。缺字段、多字段、类型错误、非法枚举或混入解释文本均判定失败。

### 6.2 任务正确性

`task_correctness` 是唯一业务质量总分。

Final 使用：

```text
70% × structured_score + 30% × reply_score
```

`structured_score` 检查动作、槽位、流程状态、缺失字段和回复类型；`reply_score` 检查要求的语义动作、必要事实、禁止动作和事实一致性。Tool Call 使用动作、工具名和参数正确性的平均分。

偏好语义、回复语义、回复事实一致性和断言结果作为任务分内部诊断，不再重复形成多个总分维度。

偏好语义先做文本归一化和高置信别名快速匹配，再执行否定含义门控；其余表达使用本地中文 Embedding `BAAI/bge-small-zh-v1.5` 计算余弦相似度，阈值为 `0.70`。例如“肩颈”和“肩颈按摩”应视为等价，“肩颈”和“不要按摩肩颈”不得匹配。多项偏好按一对一最佳匹配计算 precision、recall 和 F1，不调用 LLM judge 或外部评分 API。

首次运行需要下载约 92 MB 的 Embedding 权重。执行 `uv sync` 安装依赖后，可用以下命令提前缓存模型：

```powershell
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-small-zh-v1.5', device='cpu')"
```

Windows 默认缓存位于 `%USERPROFILE%\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5`；缓存完成后可设置 `$env:HF_HUB_OFFLINE="1"` 进行离线评测。

### 6.3 资源与时间

资源指标在当前阶段显示 `N/A`。性能报告保留平均总时延、P95 总时延、平均首字时延和平均 tokens/s。

### 6.4 场景切片与错误标签

报告按 `missing_information`、`tool_call`、`tool_result`、`multi_turn`、`confirmation` 和 `unrelated` 统计任务正确性。

逐样本结果保留 `wrong_action`、`wrong_tool`、`wrong_argument`、`wrong_field`、`wrong_reply_type`、`reply_semantic` 和 `reply_faithfulness` 等错误标签，用于指导后续训练数据构造。

## 7. Baseline 结果

正式报告位于 `reports/baseline-m0/`。三个本地模型和一个远端对照模型使用同一份评估集、同一 Prompt 合同和同一评分代码。

| 模型 | 协议遵循 | 任务正确性 | 有效通过率 | 平均时延 | P95 | 平均首字时延 | 吞吐 |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen3-0.6B | 39.2% | 37.7% | 2/51 | 3.68s | **4.81s** | **1.72s** | 22.0 tok/s |
| Qwen3-1.7B | 72.5% | 64.7% | 6/51 | 6.57s | 9.77s | 3.12s | 12.9 tok/s |
| Qwen3-4B-Instruct | 82.4% | 67.6% | 25/51 | 10.01s | 14.10s | 5.82s | 9.7 tok/s |
| GPT-5.6-sol（远端） | **100.0%** | **98.8%** | **51/51** | **3.45s** | 5.10s | N/A | **36.1 tok/s** |

有效通过定义为协议检查通过且任务正确性不低于 0.95；任务分严格等于 1 仅作为内部诊断。

### 7.1 结论

- 0.6B 速度最快，但协议稳定性和任务质量不足；
- 1.7B 在质量与速度之间最均衡，Tool Call 场景表现最好；
- 4B 整体质量、结构化状态、Reply、工具结果和多轮能力最高，但时延最大；
- 4B 的主要短板是完整信息下的工具动作决策；
- 当前模型均不满足最终 CPU 流式响应目标，后续需要训练、Schema 约束、确定性工具门控和运行时优化共同解决。

## 8. 阶段交付

阶段二最终交付：

- 可校验的 51 条协议回归评估集；
- 严格的数据与输出合同；
- Reply-aware 的状态和工具上下文；
- 单一任务正确性评分与确定性诊断；
- 场景切片、错误标签和性能统计；
- 可供后续 SFT、DPO、量化和部署阶段复用的 Baseline。

