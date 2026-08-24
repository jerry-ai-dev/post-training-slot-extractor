# Round 004 远端训练

这是最后一轮定向 SFT，同时评测冻结的 51 条主评测集和训练前创建的 24 条独立盲测集。

```bash
bash experiments/phase06/round-004/package/commands/01-prepare.sh
bash experiments/phase06/round-004/package/commands/02-train.sh
bash experiments/phase06/round-004/package/commands/03-evaluate.sh
bash experiments/phase06/round-004/package/commands/04-package-results.sh
```

下载 `experiments/phase06/round-004/round-004-cloud-results.tar.gz`。默认包包含训练日志、指标、loss 图和两套评测结果，不包含 Adapter、checkpoint、optimizer。

先下载轻量包并保持实例运行；只有分析确认达标后，才运行 `05-package-selected-adapter.sh <run-id>` 单独取回最终 Adapter。

## 第四轮补充：统一盲测第三轮候选模型

第四轮两个模型完成后，还需用同一套 24 条盲测集评估第三轮表现最好的 1.7B，才能与第四轮 0.6B 公平比较。这一步只做推理，不会重新训练，也不会修改 Adapter：

```bash
git pull
bash experiments/phase06/round-004/package/commands/06-evaluate-prior-finalist.sh
```

前提是远端仍存在 `models/adapters/r003-qwen3-1.7b-sft/adapter_config.json`。完成后下载：

```text
experiments/phase06/round-004/round-004-prior-blind-results.tar.gz
```

补充包只包含 predictions、scorecard、运行日志、环境元数据和 SHA-256 校验文件，不包含 Adapter。
