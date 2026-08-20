#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
python -m scripts.eval.run_phase06_cloud \
  --plan experiments/phase06/round-001/package/run-plan.yaml \
  --cases data/eval/test.jsonl \
  --results-root experiments/phase06/round-001/cloud-results
