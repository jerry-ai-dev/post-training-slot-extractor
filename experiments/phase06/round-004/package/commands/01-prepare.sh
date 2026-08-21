#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HOME="${HF_HOME:-/root/autodl-tmp/huggingface}"
python - <<'PY'
import sys
if sys.version_info < (3, 12): raise SystemExit("Python 3.12+ is required")
print("Python:", sys.version)
PY
command -v nvidia-smi >/dev/null
nvidia-smi
sha256sum -c experiments/phase06/round-004/package/checksums.sha256
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m pip install -e . pytest ruff
python -m pytest tests/unit/test_phase06_round4_sft.py tests/unit/test_eval_dataset.py -q
mkdir -p experiments/phase06/round-004/cloud-results
python -m pip freeze > experiments/phase06/round-004/cloud-results/pip-freeze.txt
nvidia-smi -q > experiments/phase06/round-004/cloud-results/nvidia-smi.txt
git rev-parse HEAD > experiments/phase06/round-004/cloud-results/git-commit.txt
