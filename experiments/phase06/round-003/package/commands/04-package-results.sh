#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
ROOT="experiments/phase06/round-003/cloud-results"
test -f "$ROOT/r003-qwen3-0.6b-sft/scorecard.json"
test -f "$ROOT/r003-qwen3-1.7b-sft/scorecard.json"
for RUN_ID in r003-qwen3-0.6b-sft r003-qwen3-1.7b-sft; do
  SOURCE="models/adapters/$RUN_ID"
  DESTINATION="$ROOT/$RUN_ID/training"
  test -f "$SOURCE/train_results.json"
  test -f "$SOURCE/eval_results.json"
  test -f "$SOURCE/trainer_state.json"
  mkdir -p "$DESTINATION"
  for NAME in \
    train_results.json \
    eval_results.json \
    trainer_state.json \
    trainer_log.jsonl \
    training_loss.png \
    training_eval_loss.png; do
    if [[ -f "$SOURCE/$NAME" ]]; then
      cp "$SOURCE/$NAME" "$DESTINATION/$NAME"
    fi
  done
done
python -m pip freeze > "$ROOT/pip-freeze.txt"
nvidia-smi -q > "$ROOT/nvidia-smi-final.txt"
git status --short > "$ROOT/git-status.txt"
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | \
  xargs -0 sha256sum > "$ROOT/SHA256SUMS"
tar -czf experiments/phase06/round-003/round-003-cloud-results.tar.gz \
  "$ROOT" \
  configs/training/llamafactory/_rendered/r003-qwen3-0.6b-sft.yaml \
  configs/training/llamafactory/_rendered/r003-qwen3-1.7b-sft.yaml
echo "Result package: experiments/phase06/round-003/round-003-cloud-results.tar.gz"
