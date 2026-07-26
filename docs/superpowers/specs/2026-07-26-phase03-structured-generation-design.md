# Phase 03 严格结构化生成设计

## 目标

让 GPT-5.6-sol 真机连续生成 25 条 raw 样本，并保证每条在进入数据集前同时通过 JSON Schema、raw 业务合同、训评隔离、SFT 渲染和 DPO 扰动校验。任何一条失败时，构建整体失败且不写入半成品。

## 根因

当前实现仅通过自然语言提示词描述 raw 合同。模型曾输出错误字段集合、非法时间、非法 history 消息结构和非法 `technician_status` 枚举。提示词与最多三次错误反馈无法稳定约束语法层结构。

## 架构

### 1. Raw JSON Schema

新增唯一的 raw 响应 schema 源，覆盖：

- 七个顶层字段，`additionalProperties: false`；
- `final` 与 `tool_call` 两种 expected 分支；
- final 的 14 个精确字段、类型和枚举；
- tool_call 及五个 arguments 字段；
- history 的 user、assistant 自然消息、assistant 工具调用、tool 结果四种精确对象；
- 类别、conversation kind、DPO target token 和基础时间格式。

Schema 只约束结构，不承担跨字段业务一致性。

### 2. Backend 传参

扩展 `GenerationParams`，允许调用方携带可选 JSON Schema。`OpenAIResponsesBackend` 将其映射为 Responses API 的 `text.format` 严格 JSON Schema；MockBackend 和其他 Backend 可忽略该可选参数，保持现有推理路径兼容。

### 3. 生成与重试

`RawGenerator` 每次调用都传递 raw schema。响应先经过 JSON 解析和 schema 保证，再由现有 `raw_sample_from_record`、`validate_raw_sample` 执行业务校验。业务失败时把精确错误反馈给模型，最多重试三次；耗尽后抛错，不接受或自动篡改标签。

### 4. 原子构建

25 条 raw 全部生成和验证成功后才进入 `build_dataset`。构建器继续执行训评隔离、审计、SFT/DPO 转换和原子文件写入。失败请求不得留下 `experiments/runs/phase03-gpt-smoke` 半成品。

## 错误处理

- Responses API 返回 `incomplete` 时，错误消息包含 `incomplete_details.reason`；
- Schema 不被服务支持时明确失败，不静默退回普通文本生成；
- 业务校验重试耗尽时报告 sample id、最后校验错误和尝试次数；
- 不在生成后通过本地规则偷偷修正模型标签。

## 测试

- 单测确认 Responses 请求包含严格 `text.format` schema；
- 单测确认未传 schema 的既有调用 payload 不变；
- 单测确认 RawGenerator 每次尝试都传 schema；
- 单测覆盖 final/tool_call/history/enums 的 schema 关键节点；
- 现有 mock 25 条、全量非本地测试和 Ruff 必须继续通过；
- 真机验收以 `raw=25, sft_train=20, sft_val=5`、DPO 实际计数、零重叠和 exit 0 为唯一成功标准。

## 非目标

- 不生成正式约 1,400/400 条训练数据；
- 不放宽 raw 或输出业务合同；
- 不引入独立修复模型；
- 不修改在线评估与部署的默认生成格式。
