#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"

RUN_ID="r003-qwen3-1.7b-sft"
ADAPTER="models/adapters/$RUN_ID"
ROOT="experiments/phase06/round-004/cloud-results/prior-blind"
RUN_DIR="$ROOT/$RUN_ID"
ARCHIVE="experiments/phase06/round-004/round-004-prior-blind-results.tar.gz"

if [[ ! -f "$ADAPTER/adapter_config.json" ]]; then
  echo "Missing prior-round adapter: $ADAPTER" >&2
  echo "This is evaluation only; do not retrain. Restore the existing Round 003 adapter first." >&2
  exit 1
fi

python -m scripts.eval.run_phase06_cloud \
  --plan experiments/phase06/round-003/package/run-plan.yaml \
  --run-id "$RUN_ID" \
  --cases data/eval/phase06_holdout_v0.3.jsonl \
  --results-root "$ROOT"

test -f "$RUN_DIR/predictions.jsonl"
test -f "$RUN_DIR/scorecard.json"
test -f "$RUN_DIR/evaluation.json"

git rev-parse HEAD > "$ROOT/git-commit.txt"
git status --short > "$ROOT/git-status.txt"
find "$ROOT" -type f ! -name SHA256SUMS -print0 | sort -z | xargs -0 sha256sum > "$ROOT/SHA256SUMS"

# Only evaluation outputs are archived. The Adapter is intentionally excluded.
tar -czf "$ARCHIVE" "$ROOT"
echo "Prior finalist evaluation package: $ARCHIVE"
