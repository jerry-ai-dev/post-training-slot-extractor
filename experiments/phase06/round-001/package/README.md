# Round 001 远端 SFT 训练

本轮仅训练两个 SFT 模型，不运行 DPO。

## AutoDL 一条龙流程

选择带 CUDA 的镜像并激活 Python 3.12 环境，在仓库根目录依次执行：

```bash
bash experiments/phase06/round-001/package/commands/01-prepare.sh
bash experiments/phase06/round-001/package/commands/02-train.sh
bash experiments/phase06/round-001/package/commands/03-evaluate.sh
bash experiments/phase06/round-001/package/commands/04-package-results.sh
```

最后下载：

- `experiments/phase06/round-001/round-001-cloud-results.tar.gz`

评测使用 Transformers + PEFT 直接加载 Base + LoRA，在 GPU 上对
`data/eval/test.jsonl`（eval-v0.2）逐条推理；不需要先合并模型或转 GGUF。
两个模型会顺序加载，避免同时占用显存。每个模型保存：

- `predictions.jsonl`：逐样本输出、各维度分数和失败原因；
- `scorecard.json`：通过率、维度与场景切片汇总；
- `server.log`：模型加载和推理日志；
- `evaluation.json`：基础模型、adapter、数据集和评测时间。

结果包还包含两个 LoRA adapter、实际训练配置、训练日志、Git commit、依赖版本、
GPU 环境及所有结果文件的 SHA-256，因此下载一个压缩包即可完成本地归档和复核。

输出目录：

- `models/adapters/r001-qwen3-0.6b-sft`
- `models/adapters/r001-qwen3-1.7b-sft`

训练结束后保留 adapter、实际渲染配置、trainer state/log、训练及验证 loss、依赖版本、GPU 环境和开始结束时间。不要覆盖 Phase 04 的旧 adapter。

若只需重跑一个模型，可直接调用：

```bash
python -m scripts.eval.run_phase06_cloud --run-id r001-qwen3-0.6b-sft
```
