#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
ROOT="experiments/phase06/round-004/cloud-results"
mkdir -p "$ROOT"
date --iso-8601=seconds > "$ROOT/training-started-at.txt"
python -m scripts.train.render_config --run-id r004-qwen3-0.6b-sft
llamafactory-cli train configs/training/llamafactory/_rendered/r004-qwen3-0.6b-sft.yaml 2>&1 | tee "$ROOT/r004-qwen3-0.6b-sft-train.log"
python -m scripts.train.render_config --run-id r004-qwen3-1.7b-sft
llamafactory-cli train configs/training/llamafactory/_rendered/r004-qwen3-1.7b-sft.yaml 2>&1 | tee "$ROOT/r004-qwen3-1.7b-sft-train.log"
date --iso-8601=seconds > "$ROOT/training-finished-at.txt"
