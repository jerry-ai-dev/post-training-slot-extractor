# llama.cpp 本地冒烟验证

阶段一用这条路径验证评估 harness 能连真实的本地 OpenAI 兼容后端。冒烟模型必须跟随 `finetune-spec.md` 2.3.6 的主实验系列，默认使用 Qwen3 低配档纯文本 GGUF。官方 `Qwen/Qwen3-0.6B-GGUF` 仓库当前仅提供 `Q8_0` 一档（约 610MB），故本阶段实际采用 `Qwen3-0.6B-Q8_0.gguf`。它只用于验证链路，但不能跨用非主实验系列（尤其不用带视觉编码器的 Qwen3.5/3.6）；若阶段四实验矩阵最终锁定其他 Qwen3 低配档，本文件和后端配置同步调整。

## 获取 llama.cpp（Windows CPU）

从 llama.cpp releases 下载预编译 CPU 包（避免本地编译），解压后把 `llama-server.exe` 及其全部 `*.dll` 依赖放到 `deployment/llama_cpp/bin/`：

```powershell
# 例：b9912 版 CPU x64 包 llama-<ver>-bin-win-cpu-x64.zip（约 16MB）
# 解压后 exe 与 ggml-*.dll 同层，需一并拷入 bin/
```

## 获取模型

从 `https://huggingface.co/Qwen/Qwen3-0.6B-GGUF` 下载 `Qwen3-0.6B-Q8_0.gguf` 到 `models/gguf/`。

阶段二 M0 baseline 另用两档（均放 `models/gguf/`）：
- `Qwen3-1.7B-Q8_0.gguf`：官方 `Qwen/Qwen3-1.7B-GGUF`（仅提供 Q8_0 一档），非门控，可直接 `curl` 下载。
- `Qwen3-4B-Instruct-2507-Q4_K_M.gguf`：官方 4B 仓库门控，改用 `unsloth/Qwen3-4B-Instruct-2507-GGUF` 镜像。

## 期望目录

- `deployment/llama_cpp/bin/llama-server.exe`（及同目录下的 `ggml-*.dll` 等依赖）
- `models/gguf/Qwen3-0.6B-Q8_0.gguf`

## 启动服务

```powershell
.\scripts\serve\start_llama_server.ps1
```

服务应监听 `http://127.0.0.1:8080/v1`。

## 运行评估

```powershell
uv run slot-eval --backend-config configs/inference/llama_server.yaml --cases tests/fixtures/phase01_eval.jsonl --report-dir reports/generated
```

这个小模型的 JSON 质量可能很低（Qwen3 为思考型模型，`<think>` 推理易在 `max_tokens` 内耗尽而无正文输出）。阶段一验收条件不是模型分数高，而是命令能完整返回分数卡，并且速度维度有真实计时值。

## 已验收结果（2026-07-08）

按上述步骤真机跑通：`slot-eval` 用 `configs/inference/llama_server.yaml` 真连本地 `llama-server` 出分数卡；速度维度 `first_token_ms ≈ 4.6s`、`tokens_per_s ≈ 31–47`，为真实埋点；质量维度因空输出为 0 分，符合预期。详见 `project-log/phase-01-scaffold/log.md`。
