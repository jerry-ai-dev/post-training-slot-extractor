#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
RESULTS="experiments/phase06/round-003/cloud-results"
mkdir -p "$RESULTS"
date --iso-8601=seconds > "$RESULTS/training-started-at.txt"
python -m scripts.train.render_config --run-id r003-qwen3-0.6b-sft
llamafactory-cli train configs/training/llamafactory/_rendered/r003-qwen3-0.6b-sft.yaml 2>&1 | tee "$RESULTS/r003-qwen3-0.6b-sft-train.log"
python -m scripts.train.render_config --run-id r003-qwen3-1.7b-sft
llamafactory-cli train configs/training/llamafactory/_rendered/r003-qwen3-1.7b-sft.yaml 2>&1 | tee "$RESULTS/r003-qwen3-1.7b-sft-train.log"
date --iso-8601=seconds > "$RESULTS/training-finished-at.txt"
