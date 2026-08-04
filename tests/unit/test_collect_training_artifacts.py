import json
from pathlib import Path

import pytest
import yaml

from scripts.train.collect_artifacts import ArtifactError, collect_artifacts, verify_artifacts


def _fake_training_output(root: Path) -> tuple[Path, Path]:
    output = root / "training"
    output.mkdir()
    (output / "adapter_config.json").write_text('{"r":16}\n', encoding="utf-8")
    (output / "adapter_model.safetensors").write_bytes(b"adapter")
    (output / "trainer_state.json").write_text(
        json.dumps({"log_history": [{"loss": 1.2, "step": 1}]}) + "\n",
        encoding="utf-8",
    )
    rendered = root / "rendered.yaml"
    rendered.write_text(
        yaml.safe_dump(
            {
                "stage": "sft",
                "model_name_or_path": "Qwen/Qwen3-0.6B",
                "model_revision": "main",
            }
        ),
        encoding="utf-8",
    )
    return output, rendered


def test_collect_artifacts_creates_auditable_run(tmp_path: Path) -> None:
    output, rendered = _fake_training_output(tmp_path)
    result = collect_artifacts(
        "qwen3-0.6b-sft",
        output,
        rendered,
        Path("requirements-train.txt"),
        tmp_path / "runs",
    )
    assert (result / "adapter/adapter_config.json").is_file()
    assert (result / "config.rendered.yaml").is_file()
    assert (result / "requirements-train.txt").is_file()
    assert (result / "trainer_log.jsonl").read_text().count("\n") == 1
    manifest = json.loads((result / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["run_id"] == "qwen3-0.6b-sft"
    assert manifest["status"] == "trained"
    assert verify_artifacts(result, training_only=True)["run_id"] == "qwen3-0.6b-sft"


def test_collect_artifacts_rejects_incomplete_training_output(tmp_path: Path) -> None:
    output, rendered = _fake_training_output(tmp_path)
    (output / "trainer_state.json").unlink()
    with pytest.raises(ArtifactError, match="trainer_state.json"):
        collect_artifacts(
            "qwen3-0.6b-sft",
            output,
            rendered,
            Path("requirements-train.txt"),
            tmp_path / "runs",
        )

