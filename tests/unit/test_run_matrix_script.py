from pathlib import Path


def test_run_matrix_is_fail_fast_and_dependency_ordered() -> None:
    script = Path("scripts/train/run_matrix.sh").read_text(encoding="utf-8")
    assert "set -euo pipefail" in script
    runs = [
        "qwen3-0.6b-sft",
        "qwen3-1.7b-sft",
        "qwen3-0.6b-dpo-b01",
        "qwen3-0.6b-dpo-b03",
        "qwen3-1.7b-dpo-b01",
        "qwen3-1.7b-dpo-b03",
    ]
    positions = [script.index(run_id) for run_id in runs]
    assert positions == sorted(positions)
    loop = script[script.index('for run_id in "${RUNS[@]}"') :]
    assert loop.index("render_config") < loop.index("llamafactory-cli train")
    assert loop.index("llamafactory-cli train") < loop.index("collect_artifacts")
    assert "--from-run" in script and "--skip-complete" in script
    assert "uv run" not in script
    assert "python -m scripts.train.render_config" in script
