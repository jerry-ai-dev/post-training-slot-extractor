#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
ROOT="experiments/phase06/round-004/cloud-results"
python -m scripts.eval.run_phase06_cloud \
  --plan experiments/phase06/round-004/package/run-plan.yaml \
  --cases data/eval/test.jsonl \
  --results-root "$ROOT"
python -m scripts.eval.run_phase06_cloud \
  --plan experiments/phase06/round-004/package/run-plan.yaml \
  --cases data/eval/phase06_holdout_v0.3.jsonl \
  --results-root "$ROOT/blind"
