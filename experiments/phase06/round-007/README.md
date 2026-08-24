# Phase 06 Round 007：Q8/Q4终选与真实TTFT

本轮建立在云端重复复评结果一致的前提上，只比较 `Q8_0` 与 `Q4_K_M`。质量结论来自云端完整评测；速度结论来自最终Round 004合并模型在本地CPU上的真实流式请求。

运行：

```powershell
.venv\Scripts\python.exe -m scripts.eval.run_phase06_round007_final_local
```

默认关闭 llama.cpp Prompt Cache，选择8条由短到长的真实评测请求，每条重复3次，使用8线程。输出写入 `local-final/benchmark.json`，模型权重不进入Git。
