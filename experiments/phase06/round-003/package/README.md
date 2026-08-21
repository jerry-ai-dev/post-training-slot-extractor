# Round 003 远端 SFT 训练

本轮只训练两个 SFT 模型，不运行 DPO。结果包不包含 Adapter、checkpoint、optimizer
或其他大权重；只有确认模型效果达到标准后，才单独取回对应 Adapter。

在仓库根目录依次执行：

```bash
bash experiments/phase06/round-003/package/commands/01-prepare.sh
bash experiments/phase06/round-003/package/commands/02-train.sh
bash experiments/phase06/round-003/package/commands/03-evaluate.sh
bash experiments/phase06/round-003/package/commands/04-package-results.sh
```

最后只需把以下结果包下载到本地：

```text
experiments/phase06/round-003/round-003-cloud-results.tar.gz
```

轻量结果包包含：

- 完整控制台训练日志；
- `trainer_state.json`、`trainer_log.jsonl`；
- `train_results.json`、`eval_results.json`；
- 训练与验证 loss 图；
- 逐样本预测、评分卡、评测日志；
- 实际训练配置、依赖版本、GPU 环境和所有文件的 SHA-256。

结果包明确不包含 `adapter_model.safetensors`、checkpoint 和 optimizer。下载并验证轻量包后即可释放实例；如果分析结果达到最终候选标准，需要在释放实例前另外下载对应 Adapter。

确认某个模型达标后，才执行可选脚本：

```bash
bash experiments/phase06/round-003/package/commands/05-package-selected-adapter.sh r003-qwen3-0.6b-sft
```

参数也可以换成 `r003-qwen3-1.7b-sft`。该脚本只打包选中的最终 Adapter，不包含训练 checkpoint 和 optimizer。
