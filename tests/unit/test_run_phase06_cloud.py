from pathlib import Path

import pytest

from scripts.eval.run_phase06_cloud import CloudEvaluationError, api_config, load_runs


def test_load_round_001_runs() -> None:
    runs = load_runs(Path("experiments/phase06/round-001/package/run-plan.yaml"))
    assert [run["run_id"] for run in runs] == [
        "r001-qwen3-0.6b-sft",
        "r001-qwen3-1.7b-sft",
    ]


def test_api_config_requires_trained_adapter(tmp_path: Path) -> None:
    run = {
        "run_id": "demo",
        "model": {"name": "Qwen/Qwen3-0.6B"},
        "output_dir": str(tmp_path / "adapter"),
    }
    with pytest.raises(CloudEvaluationError, match="adapter is missing"):
        api_config(run)


def test_api_config_uses_base_plus_lora(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    config = api_config(
        {
            "run_id": "demo",
            "model": {"name": "Qwen/Qwen3-0.6B"},
            "output_dir": str(adapter),
        }
    )
    assert config["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert config["adapter_name_or_path"] == str(adapter.resolve())
    assert config["enable_thinking"] is False
