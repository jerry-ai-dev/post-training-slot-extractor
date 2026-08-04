from pathlib import Path

import pytest
import yaml

from scripts.train.dryrun import DryRunError, build_dryrun_jobs, validate_training_data


def test_build_dryrun_jobs_uses_06b_and_sft_before_dpo(tmp_path: Path) -> None:
    jobs = build_dryrun_jobs(tmp_path)
    assert [job.run_id for job in jobs] == ["qwen3-0.6b-sft", "qwen3-0.6b-dpo-b01"]
    for job in jobs:
        config = yaml.safe_load(job.config.read_text(encoding="utf-8"))
        assert config["max_steps"] == 2 and config["use_cpu"] is True
        assert config["bf16"] is False and config["fp16"] is False
        assert config["overwrite_output_dir"] is True
        assert config["cutoff_len"] == 256
        assert config["per_device_train_batch_size"] == 1
        assert config["gradient_accumulation_steps"] == 1
        assert config["do_eval"] is False
        assert config["disable_gradient_checkpointing"] is True
        assert config["lora_target"] == "q_proj,v_proj"
        assert Path(config["dataset_dir"], "dataset_info.json").is_file()
        assert config["dataset"] in {"phase04_dryrun_sft", "phase04_dryrun_dpo"}
        assert "eval_dataset" not in config
    dpo = yaml.safe_load(jobs[1].config.read_text(encoding="utf-8"))
    assert dpo["adapter_name_or_path"] == str(jobs[0].output_dir)


def test_validate_training_data_accepts_current_v01() -> None:
    summary = validate_training_data(Path("data/processed"), "v0.1")
    assert summary == {"sft_rows_checked": 1, "dpo_rows_checked": 1, "no_think": True}


def test_validate_training_data_rejects_thinking_tokens(tmp_path: Path) -> None:
    sft = tmp_path / "sft" / "v0.1"
    dpo = tmp_path / "dpo" / "v0.1"
    sft.mkdir(parents=True)
    dpo.mkdir(parents=True)
    (sft / "train.jsonl").write_text(
        '{"conversations":[{"from":"gpt","value":"<think>hidden</think>"}]}\n',
        encoding="utf-8",
    )
    (dpo / "train.jsonl").write_text(
        '{"chosen":{"value":"{}"},"rejected":{"value":"{\\"x\\":1}"}}\n',
        encoding="utf-8",
    )
    with pytest.raises(DryRunError, match="thinking token"):
        validate_training_data(tmp_path, "v0.1")
