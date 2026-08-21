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
