# Round 005：评分器校准与本地终验

本轮不训练模型。它冻结第三轮 1.7B 与第四轮 0.6B 两个候选 Adapter，修正已由人工复核确认的语义评分误判，并在同一台 Windows CPU 上重新运行 51 条主评测和 24 条 holdout。

运行：

```powershell
powershell -ExecutionPolicy Bypass -File experiments/phase06/round-005/run-local.ps1
```

结果写入 `experiments/phase06/round-005/results/`。本轮结果用于识别模型真实错误和确定工程候选，不把小样本上的一条差距解释为参数规模的显著能力差异。

若 Hugging Face 全精度 CPU 推理过慢，可先使用 llama.cpp 的 LoRA 转换工具生成本地 GGUF Adapter，再运行：

```powershell
.venv\Scripts\python.exe -m scripts.eval.run_phase06_llamacpp
```

该路径使用本地 Q8 基座加 F16 LoRA，结果单独写入 `results/llamacpp`，不得与云端全精度延迟直接比较。
