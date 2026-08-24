#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ROOT="experiments/phase06/round-002/cloud-results"
for ID in r002-qwen3-0.6b-sft r002-qwen3-1.7b-sft; do
  test -f "$ROOT/$ID/scorecard.json"
  test -f "models/adapters/$ID/adapter_config.json"
done
python -m pip freeze > "$ROOT/pip-freeze.txt"
nvidia-smi -q > "$ROOT/nvidia-smi-final.txt"
git status --short > "$ROOT/git-status.txt"
find "$ROOT" models/adapters/r002-qwen3-0.6b-sft models/adapters/r002-qwen3-1.7b-sft \
  -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$ROOT/SHA256SUMS"
tar -czf experiments/phase06/round-002/round-002-cloud-results.tar.gz \
  "$ROOT" \
  models/adapters/r002-qwen3-0.6b-sft \
  models/adapters/r002-qwen3-1.7b-sft \
  configs/training/llamafactory/_rendered/r002-qwen3-0.6b-sft.yaml \
  configs/training/llamafactory/_rendered/r002-qwen3-1.7b-sft.yaml
echo "Result package: experiments/phase06/round-002/round-002-cloud-results.tar.gz"
