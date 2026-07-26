import subprocess
import sys
from pathlib import Path

import yaml

from slot_extractor.utils.jsonl import read_jsonl


def test_mock_pipeline_builds_25_samples(tmp_path) -> None:
    config = yaml.safe_load(Path("configs/data/phase03.yaml").read_text(encoding="utf-8"))
    config["counts"] = {category: 5 for category in config["counts"]}
    config.pop("dpo_target_counts", None)
    test_config = tmp_path / "phase03_test.yaml"
    test_config.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.data.build_dataset",
            "--mock",
            "--config",
            str(test_config),
            "--output-root",
            str(tmp_path),
        ],
        text=True,
        capture_output=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "raw=25, sft_train=20, sft_val=5, dpo_train=28, dpo_val=6" in completed.stdout
    assert len(list(read_jsonl(tmp_path / "raw/v0.1/samples.jsonl"))) == 25
    assert len(list(read_jsonl(tmp_path / "processed/sft/v0.1/train.jsonl"))) == 20
    assert len(list(read_jsonl(tmp_path / "processed/sft/v0.1/val.jsonl"))) == 5
    assert len(list(read_jsonl(tmp_path / "processed/dpo/v0.1/train.jsonl"))) == 28
    assert len(list(read_jsonl(tmp_path / "processed/dpo/v0.1/val.jsonl"))) == 6
    assert (tmp_path / "processed/v0.1/dataset_info.json").exists()
    assert (tmp_path / "processed/v0.1/DATASET_CARD.md").exists()
