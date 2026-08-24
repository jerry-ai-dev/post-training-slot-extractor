# Phase 06 Round 006：量化质量矩阵

本轮固定最终 `r004-qwen3-0.6b-sft` Adapter，云端合并后构建八档 GGUF，并在同一 GPU offload 环境运行完整质量评测。Adapter、合并权重和 GGUF 都是外部资产，不进入 Git。

## 1. 准备 Adapter

如果云端仍保留第四轮训练目录，直接把 `--adapter` 指向该目录。否则在本地仓库外打包约 40 MB 的 Adapter：

```powershell
tar -czf r004-qwen3-0.6b-sft-adapter.tar.gz -C models/adapters r004-qwen3-0.6b-sft
```

通过 AutoDL 文件上传、SCP 或对象存储传到云端并解压；不要提交到 Git。

## 2. 云端构建

先准备 llama.cpp 源码及 Linux 可执行文件，并确认配置中的三个工具路径正确。然后：

```bash
python -m scripts.quantize.build_phase06_round006_matrix \
  --adapter /实际路径/r004-qwen3-0.6b-sft
```

脚本可断点续跑：已经存在的合并模型、F16 和量化文件会复用。IQ2 使用同一份校准数据生成的 imatrix；其余档位为标准量化。

## 3. 云端完整质量评测

```bash
python -m scripts.eval.run_phase06_round006_quality
```

结果包含硬件、服务日志、预测和分数卡，写入 `experiments/phase06/round-006/cloud-results/`。打包时只取结果，不取模型：

```bash
tar -czf experiments/phase06/round-006/phase06-round006-results.tar.gz \
  experiments/phase06/round-006/cloud-results
```

云端结果只用于完整质量比较；本地 CPU 的 Prefill、Decode、真实 TTFT 和内存基准将在筛选候选后单独运行。

## 4. 本地 CPU 速度矩阵

本地可与云端同步运行。该测试使用相同架构的 0.6B F16 权重生成速度专用量化文件，结果只用于比较位宽的 Prefill/Decode 性能，不参与质量结论：

```powershell
.venv\Scripts\python.exe -m scripts.quantize.run_phase06_round006_local_speed
```

输出为 `experiments/phase06/round-006/local-speed/benchmark.json`。每档覆盖 128/512/1024 token Prefill、64/128 token Decode并重复三次；每项完成后立即写检查点，可断点续跑。
