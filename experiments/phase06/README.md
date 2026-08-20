# Phase 06 迭代实验台账

这里是阶段六的事实源。每次优化都创建一个新轮次，旧轮次只读保留，不覆盖历史结果。

## 开始一个新轮次

1. 复制 `_template/` 为 `round-NNN/`。
2. 在 `registry.yaml` 登记轮次、父轮次和负责人。
3. 填写 `round.yaml`、`problems.yaml`、`strategies.yaml` 和 `variants.yaml`。
4. 完成 `reports/problem-analysis.md` 与 `reports/strategy-plan.md`，人工审批后才能训练。
5. 将实际配置、命令和输入哈希冻结到 `package/`。
6. 远端执行后，把原始结果放入 `imported/runs/<run-id>/`，先校验再分析。
7. 填写 `result-analysis.md`、`conclusion.md` 和 `artifacts.yaml`，人工确认后关闭轮次。
8. 更新 `summary/` 中的跨轮索引。

轮次可以从任意已关闭轮次分支，`parent_round` 不必是编号最大的轮次。失败、跳过、参数偏差和不完整实验也必须保留。

## 命名约定

- 轮次：`round-001`、`round-002`。
- 问题：`P001`；假设：`H001`；策略：`S001`；变体：`V001`。
- Run：`r001-v001-qwen3-0.6b-sft-s01`，末尾序号用于同配置重复运行。
- 数据、评估、Prompt、schema 和 scorer 均使用不可变版本号，不使用 `latest` 作为事实记录。

大型 checkpoint、merged model 和 GGUF 可放在 Git 外，但必须在 `artifacts.yaml` 登记 URI、SHA-256、来源 run 和重建方法。

