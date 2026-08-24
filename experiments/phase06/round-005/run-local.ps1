$ErrorActionPreference = "Stop"

$root = (git rev-parse --show-toplevel).Trim()
Set-Location $root

$env:HF_HUB_OFFLINE = "1"
$env:TRANSFORMERS_OFFLINE = "1"
$env:PYTHONPATH = "$root\src;$root"
$python = Join-Path $root ".venv-train\Scripts\python.exe"
$evaluatorPython = Join-Path $root ".venv\Scripts\python.exe"
$cli = Join-Path $root ".venv-train\Scripts\llamafactory-cli.exe"
$plan = "experiments/phase06/round-005/package/run-plan.yaml"
$results = "experiments/phase06/round-005/results"

& $python -m scripts.eval.run_phase06_cloud `
  --plan $plan `
  --cases data/eval/test.jsonl `
  --results-root "$results/main" `
  --port 8010 `
  --device cpu `
  --evaluator-python $evaluatorPython `
  --llamafactory-cli $cli
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }

& $python -m scripts.eval.run_phase06_cloud `
  --plan $plan `
  --cases data/eval/phase06_holdout_v0.3.jsonl `
  --results-root "$results/holdout" `
  --port 8010 `
  --device cpu `
  --evaluator-python $evaluatorPython `
  --llamafactory-cli $cli
exit $LASTEXITCODE
