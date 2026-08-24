# Phase 06 Round 008：Q4本地推理优化

本阶段固定最终 `r004-qwen3-0.6b-sft-Q4_K_M`，不训练、不重新量化。依次测试CPU线程与batch参数、跨请求Prompt Cache、实验性精简Prompt，并在24条holdout上比较原Prompt和精简Prompt质量。

```powershell
.venv\Scripts\python.exe -m scripts.eval.run_phase06_round008_local
```

原始结果写入 `results/round008-results.json`，结论写入 `reports/conclusion.md`。

已验证配置只适用于本轮单slot实验。生产启用并发后必须重新检查Prompt Cache命中率，不能直接把单slot缓存收益外推到多用户并发。
