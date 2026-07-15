# 按摩预约 Agent 冻结评估集

## 基本信息

- 机器合同：v2.1
- 样本文件：`data/eval/test.jsonl`
- 样本数量：51
- 校验和：`data/eval/test.sha256`
- 当前范围：预约信息收集、技师查询、结果展示、用户确认、拒绝或暂缓、无关输入流转

当前评估任务定义为：

```text
上一状态 + 完整消息历史 + 当前输入
→ 新状态 + 工具调用或最终回复
```

## 运行时输入

每条样本的 `input` 固定包含：

```json
{
  "history": [],
  "current_state": null,
  "user_input": "明天下午两点做60分钟按摩",
  "current_time": "2026-06-08 10:00",
  "available_tools": ["find_technicians"]
}
```

### History

`history` 保存按时间顺序发生的用户、助手和工具事件：

```json
[
  {"role": "user", "content": "明天下午两点找王芳做60分钟按摩"},
  {
    "role": "assistant",
    "content": null,
    "tool_calls": [{
      "id": "call_specific-available-01-call",
      "type": "function",
      "function": {
        "name": "find_technicians",
        "arguments": "{\"technician_name\":\"王芳\",\"start_time\":\"2026-06-09 14:00\",\"duration_minutes\":60,\"gender\":null,\"preferences\":[]}"
      }
    }]
  },
  {
    "role": "tool",
    "tool_call_id": "call_specific-available-01-call",
    "content": "{\"mode\":\"specific\",\"status\":\"available\",\"requested_technician\":\"王芳\",\"technician\":{\"name\":\"王芳\",\"gender\":\"female\"}}"
  }
]
```

自然语言助手回复使用普通 `assistant.content`；工具调用使用
`assistant.tool_calls`；工具结果使用与 `tool_call_id` 对应的 `tool` 消息。
如果一条对话调用工具两次，History 必须保留两组完整工具事件。

### Current State

`current_state` 是本轮开始前系统已保存的结构化状态。第一轮为 `null`；后续轮次为完整对象：

```json
{
  "gender": null,
  "start_time": "2026-06-09 14:00",
  "duration_minutes": 60,
  "preferences": [],
  "technician_name": "王芳",
  "technician_status": "available",
  "confirmation": false,
  "info_complete": true,
  "unrelated": false,
  "missing_info": [],
  "last_reply_type": "confirm_available"
}
```

它只能来自此前已发生的用户输入、模型输出和工具结果，不能包含本轮 expected 才知道的信息。

工具结果触发的即时推理使用 `user_input=null`，并以对应的 `tool` 消息作为
History 最后一条事件。后续用户轮继续保留已经发生的工具消息，同时通过
`current_state` 提供最新规范化状态；旧工具结果不得证明修改后查询条件的可用性。

工具结果轮的 `current_state` 必须先由最新 `assistant.tool_calls` 参数物化：姓名、
时间、时长、性别和偏好与最新调用完全一致，`technician_status=not_checked`。
已经是 available/unavailable/not_found/no_match 的状态必须能在 History 中找到对应
工具证据。用户修改任一查询条件后，expected 必须重新输出 Tool Call。

## 输出协议

### Tool Call

Tool Call 严格只有三个顶层字段：

```json
{
  "action": "tool_call",
  "tool_name": "find_technicians",
  "arguments": {
    "technician_name": null,
    "start_time": "2026-06-09 14:00",
    "duration_minutes": 60,
    "gender": null,
    "preferences": []
  }
}
```

Tool Call 不包含 `reply_type` 或 `reply`，系统直接执行工具，不向用户发送回复。

### 无关输入

无关输入仍必须输出完整 Final 13 字段，固定为：

```json
{"action":"final","gender":null,"start_time":null,"duration_minutes":null,"preferences":[],"technician_name":null,"technician_status":"not_checked","confirmation":false,"info_complete":false,"unrelated":true,"missing_info":[],"reply_type":"handoff","reply":null}
```

`confirmation=false` 表示用户没有确认任何预约方案；`info_complete=false` 表示这不是可执行预约；`missing_info=[]` 表示无关请求不进入预约信息追问。

### Final

Final 严格包含十三个字段：

```json
{
  "action": "final",
  "gender": "female",
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
  "reply": "王芳技师明天下午2点有空，可以为您安排60分钟肩颈按摩，您确认吗？"
}
```

除 `handoff` 外，Final 的 `reply` 必须是非空自然语言。`handoff` 使用 `reply=null`，由其他 Agent 接管。

## Reply Type

| 类型 | 场景 |
|---|---|
| `handoff` | 无关输入，交给其他 Agent |
| `ask_start_time` | 缺预约时间 |
| `ask_duration` | 缺按摩时长 |
| `ask_start_time_and_duration` | 时间和时长都缺 |
| `confirm_available` | 已核实技师可用，请用户确认 |
| `inform_unavailable` | 指定技师该时段没空 |
| `inform_not_found` | 指定技师不存在 |
| `inform_no_match` | 条件搜索无匹配 |
| `booking_authorized` | 用户接受可用方案，授权上层创建预约 |
| `acknowledge_result` | 用户确认已知悉失败查询结果 |
| `appointment_paused` | 用户拒绝或暂缓待确认方案 |

`booking_authorized` 不表示预约已经写入。回复不得使用“预约成功”“已经预约好了”等完成态表述。

## 状态规则

- 当前明确修改覆盖旧值。
- 未修改字段继承 `current_state`。
- `start_time` 或 `duration_minutes` 缺失时禁止 Tool Call。
- 修改姓名、时间、时长、性别或偏好后，旧工具结果失效。
- `available/unavailable/not_found/no_match` 只能来自当前工具结果或已经保存的已核实状态。
- `confirmation=true` 只允许用户明确接受或知悉同一待确认结果且没有修改条件。
- “先不了”等表达使用 `appointment_paused`，不再与刚展示方案的 `confirm_available` 混为同一状态。

## Reply Expectations

Final 样本额外提供确定性语义要求：

```json
{
  "required_acts": [
    "inform_technician_available",
    "request_confirmation"
  ],
  "forbidden_acts": [
    "claim_booking_success"
  ],
  "required_fields": [
    "technician_name",
    "start_time",
    "duration_minutes"
  ],
  "references": [
    "王芳技师明天下午2点有空，可以安排60分钟，您确认吗？",
    "明天下午2点王芳技师可以服务60分钟，这个安排您看可以吗？"
  ]
}
```

标准回复不是唯一合法表达。`references` 提供多种可接受说法，`required_acts` 描述必须表达的行为，`forbidden_acts` 描述不能出现的语义，`required_fields` 决定回复中必须明确提到的槽位事实。

## 评分维度

最终分数卡只包含：

1. `protocol`：原始 JSON、精确字段集合、字段类型和 Final/Tool Call 协议；
2. `task_correctness`：Tool Call 业务正确性，或 Final 的结构化结果与 Reply 综合正确性；
3. `resource`：资源占位，目前为 `n/a`；
4. timing：原始时延与吞吐统计，不转换成质量分。

Final 的任务正确性使用：

```text
task_correctness = 0.70 * structured_score + 0.30 * reply_score
```

偏好语义、Reply Type、必需/禁止语义动作和 Reply 事实一致性仍被检查，但只作为统一任务分的内部子检查与错误标签，不再单独计分。多轮、无关、缺失信息、Tool Call、工具结果和确认轮只作为场景切片。

偏好语义不再依赖字符 n-gram，也不把穷举别名作为主要方案。评分器先做文本归一化；完全相同或命中少量高置信别名时直接匹配；肯定/否定极性不一致时直接拒绝；其余表达使用本地中文 Embedding `BAAI/bge-small-zh-v1.5` 计算余弦相似度，当前阈值为 `0.70`。多项偏好按一对一最佳匹配计算 precision、recall 和 F1。因此“肩颈/肩颈按摩”“足部/足部按摩”可以匹配，而“肩颈/不要按摩肩颈”会被否定门控拒绝。首次评分需下载约 92 MB 权重，缓存后可离线使用；评分过程不调用 LLM judge 或外部评分 API。

## 场景覆盖

当前 51 条样本覆盖：

- 三种必填信息缺失组合；
- 五类无关输入；
- 指定技师 available/unavailable/not_found；
- 条件搜索 matched/no_match；
- 用户重选技师、修改时间、修改偏好和重新查询；
- 可用方案确认；
- 失败结果知悉；
- 拒绝或暂缓待确认方案；
- 多轮槽位继承和覆盖。

当前范围不覆盖已落库预约的取消/改约、工具超时重试和新的业务字段。

## 校验

运行：

```powershell
uv run python scripts/eval/validate_dataset.py `
  --cases data/eval/test.jsonl `
  --contract data/eval/dataset_contract.json
```

校验器检查输入键、状态形状、完整消息 History、工具调用配对、严格输出 Schema、Reply Type 映射、语义要求、工具链后继关系和 assertions。
