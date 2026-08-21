#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"

python - <<'PY'
import sys
if sys.version_info < (3, 12):
    raise SystemExit("Python 3.12+ is required")
print("Python:", sys.version)
PY

command -v nvidia-smi >/dev/null || { echo "nvidia-smi not found" >&2; exit 1; }
nvidia-smi
sha256sum -c experiments/phase06/round-002/package/checksums.sha256
python -m pip install --upgrade pip
python -m pip install -r requirements-train.txt
python -m pip install -e .
python -m pip install pytest ruff
python -m pytest tests/unit/test_phase06_round2_sft.py tests/unit/test_eval_dataset.py -q
mkdir -p experiments/phase06/round-002/cloud-results
python -m pip freeze > experiments/phase06/round-002/cloud-results/pip-freeze.txt
nvidia-smi -q > experiments/phase06/round-002/cloud-results/nvidia-smi.txt
git rev-parse HEAD > experiments/phase06/round-002/cloud-results/git-commit.txt
