#!/usr/bin/env bash
set -euo pipefail

RUNS=(
  qwen3-0.6b-sft
  qwen3-1.7b-sft
  qwen3-0.6b-dpo-b01
  qwen3-0.6b-dpo-b03
  qwen3-1.7b-dpo-b01
  qwen3-1.7b-dpo-b03
)

from_run=""
skip_complete=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --from-run) from_run="$2"; shift 2 ;;
    --skip-complete) skip_complete=true; shift ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

started=false
[[ -z "$from_run" ]] && started=true
for run_id in "${RUNS[@]}"; do
  if [[ "$run_id" == "$from_run" ]]; then started=true; fi
  if [[ "$started" != true ]]; then continue; fi
  run_dir="experiments/runs/phase04-${run_id}"
  if [[ "$skip_complete" == true && -f "${run_dir}/manifest.json" ]]; then
    continue
  fi
  python -m scripts.train.render_config --run-id "$run_id"
  config="configs/training/llamafactory/_rendered/${run_id}.yaml"
  llamafactory-cli train "$config"
  python -m scripts.train.collect_artifacts \
    --run-id "$run_id" \
    --training-output "models/adapters/${run_id}" \
    --rendered-config "$config"
done
