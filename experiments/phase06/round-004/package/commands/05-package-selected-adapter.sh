#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
RUN_ID="${1:-}"
case "$RUN_ID" in
  r004-qwen3-0.6b-sft|r004-qwen3-1.7b-sft) ;;
  *) echo "Usage: $0 r004-qwen3-0.6b-sft|r004-qwen3-1.7b-sft" >&2; exit 2 ;;
esac
SOURCE="models/adapters/$RUN_ID"
test -f "$SOURCE/adapter_config.json"
test -f "$SOURCE/adapter_model.safetensors"
sha256sum "$SOURCE/adapter_config.json" "$SOURCE/adapter_model.safetensors" > "$SOURCE/ADAPTER_SHA256SUMS"
tar -czf "experiments/phase06/round-004/${RUN_ID}-adapter.tar.gz" --exclude='checkpoint-*' --exclude='optimizer.pt' "$SOURCE"
echo "Selected adapter package: experiments/phase06/round-004/${RUN_ID}-adapter.tar.gz"
