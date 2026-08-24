# Round 002 AutoDL 训练包

本轮仍只训练 SFT。0.6B 使用 `small` 数据视图，1.7B 使用 `large` 数据视图；两者共享
完整 v0.2 回放和 210 条残余错误核心样本，各自再加入 80 条专项样本。

在仓库根目录依次执行：

```bash
bash experiments/phase06/round-002/package/commands/01-prepare.sh
bash experiments/phase06/round-002/package/commands/02-train.sh
bash experiments/phase06/round-002/package/commands/03-evaluate.sh
bash experiments/phase06/round-002/package/commands/04-package-results.sh
```

最终下载：

```text
experiments/phase06/round-002/round-002-cloud-results.tar.gz
```

建议沿用：

```bash
export HF_ENDPOINT=https://hf-mirror.com
export HF_HOME=/root/autodl-tmp/huggingface
export HF_HUB_DISABLE_XET=1
export HF_HUB_DOWNLOAD_TIMEOUT=600
```
