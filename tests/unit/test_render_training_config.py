from pathlib import Path

import pytest
import yaml

from scripts.train.render_config import parse_set_values, render_run


def test_render_sft_is_complete_and_deterministic(tmp_path: Path) -> None:
    first = render_run("qwen3-0.6b-sft", output_root=tmp_path)
    first_bytes = first.read_bytes()
    second = render_run("qwen3-0.6b-sft", output_root=tmp_path)
    assert first_bytes == second.read_bytes()
    config = yaml.safe_load(first.read_text(encoding="utf-8"))
    assert config["stage"] == "sft"
    assert config["model_name_or_path"] == "Qwen/Qwen3-0.6B"
    assert "run_id" not in config


def test_render_dpo_keeps_parent_adapter_and_beta(tmp_path: Path) -> None:
    path = render_run("qwen3-1.7b-dpo-b03", output_root=tmp_path)
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["adapter_name_or_path"] == "models/adapters/qwen3-1.7b-sft"
    assert config["pref_beta"] == 0.3


def test_unknown_run_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown run_id"):
        render_run("missing", output_root=tmp_path)


def test_cli_overrides_are_typed_and_limited() -> None:
    assert parse_set_values(["max_steps=2", "use_cpu=true", "bf16=false"]) == {
        "max_steps": 2,
        "use_cpu": True,
        "bf16": False,
    }
    with pytest.raises(ValueError, match="not allowed"):
        parse_set_values(["learning_rate=0.9"])
