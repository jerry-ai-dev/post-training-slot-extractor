import hashlib
import json
import zipfile
from pathlib import Path

import pytest

from scripts.train.package_cloud_artifacts import PHASE04_RUN_IDS, PackageError, package_runs


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _training_run(root: Path, run_id: str) -> None:
    run = root / f"phase04-{run_id}"
    (run / "adapter").mkdir(parents=True)
    (run / "adapter" / "adapter_config.json").write_text("{}", encoding="utf-8")
    (run / "adapter" / "adapter_model.safetensors").write_bytes(b"adapter")
    (run / "config.rendered.yaml").write_text("stage: sft\n", encoding="utf-8")
    (run / "requirements-train.txt").write_text("torch==2.6.0\n", encoding="utf-8")
    (run / "trainer_log.jsonl").write_text("{}\n", encoding="utf-8")
    manifest = {
        "run_id": run_id,
        "config_sha256": _sha256(run / "config.rendered.yaml"),
        "requirements_sha256": _sha256(run / "requirements-train.txt"),
    }
    (run / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def test_package_runs_requires_all_six_training_runs(tmp_path: Path) -> None:
    with pytest.raises(PackageError, match="missing run"):
        package_runs(tmp_path / "runs", tmp_path / "phase04.zip")


def test_package_runs_preserves_run_directories(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    for run_id in PHASE04_RUN_IDS:
        _training_run(runs, run_id)

    output = package_runs(runs, tmp_path / "phase04.zip")

    with zipfile.ZipFile(output) as archive:
        names = set(archive.namelist())
    for run_id in PHASE04_RUN_IDS:
        assert f"phase04-{run_id}/manifest.json" in names
