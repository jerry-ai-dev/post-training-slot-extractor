from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


class ArtifactError(RuntimeError):
    """Raised when a training run cannot be collected or verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_revision() -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"], text=True, capture_output=True, check=False
    )
    return completed.stdout.strip() if completed.returncode == 0 else "unknown"


def _require_files(paths: list[Path]) -> None:
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ArtifactError(f"missing required artifact(s): {', '.join(missing)}")


def collect_artifacts(
    run_id: str,
    training_output: Path,
    rendered_config: Path,
    requirements: Path,
    runs_root: Path = Path("experiments/runs"),
) -> Path:
    adapter_config = training_output / "adapter_config.json"
    trainer_state = training_output / "trainer_state.json"
    _require_files([adapter_config, trainer_state, rendered_config, requirements])
    if not any(training_output.glob("adapter_model.*")):
        raise ArtifactError(f"missing adapter_model file in {training_output}")
    destination = runs_root / f"phase04-{run_id}"
    if destination.exists():
        raise ArtifactError(f"run directory already exists: {destination}")
    staging = runs_root / f".phase04-{run_id}.staging"
    if staging.exists():
        shutil.rmtree(staging)
    staging.mkdir(parents=True)
    shutil.copytree(training_output, staging / "adapter")
    shutil.copy2(rendered_config, staging / "config.rendered.yaml")
    shutil.copy2(requirements, staging / "requirements-train.txt")
    state = json.loads(trainer_state.read_text(encoding="utf-8"))
    log_history = state.get("log_history")
    if not isinstance(log_history, list):
        raise ArtifactError("trainer_state.json has no log_history list")
    (staging / "trainer_log.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in log_history),
        encoding="utf-8",
        newline="\n",
    )
    config = yaml.safe_load(rendered_config.read_text(encoding="utf-8"))
    manifest: dict[str, Any] = {
        "run_id": run_id,
        "status": "trained",
        "collected_at": datetime.now(UTC).isoformat(),
        "git_commit": _git_revision(),
        "stage": config["stage"],
        "base_model": config["model_name_or_path"],
        "base_revision": config.get("model_revision", "main"),
        "adapter_path": "adapter",
        "config_sha256": _sha256(staging / "config.rendered.yaml"),
        "requirements_sha256": _sha256(staging / "requirements-train.txt"),
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    verify_artifacts(staging, training_only=True)
    staging.replace(destination)
    return destination


def verify_artifacts(run_dir: Path, *, training_only: bool = False) -> dict[str, Any]:
    required = [
        run_dir / "manifest.json",
        run_dir / "config.rendered.yaml",
        run_dir / "requirements-train.txt",
        run_dir / "trainer_log.jsonl",
        run_dir / "adapter" / "adapter_config.json",
    ]
    if not training_only:
        required.extend(
            [run_dir / "predictions.jsonl", run_dir / "scorecard.json", run_dir / "server.log"]
        )
    _require_files(required)
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    if manifest["config_sha256"] != _sha256(run_dir / "config.rendered.yaml"):
        raise ArtifactError("config.rendered.yaml hash mismatch")
    if manifest["requirements_sha256"] != _sha256(run_dir / "requirements-train.txt"):
        raise ArtifactError("requirements-train.txt hash mismatch")
    return manifest


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect or verify phase04 training artifacts.")
    parser.add_argument("--run-id")
    parser.add_argument("--training-output", type=Path)
    parser.add_argument("--rendered-config", type=Path)
    parser.add_argument("--requirements", type=Path, default=Path("requirements-train.txt"))
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--verify-only", type=Path)
    parser.add_argument("--training-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.verify_only:
            print(json.dumps(verify_artifacts(args.verify_only, training_only=args.training_only)))
        else:
            if not args.run_id or not args.training_output or not args.rendered_config:
                raise ArtifactError(
                    "collection requires --run-id, --training-output and --rendered-config"
                )
            print(
                collect_artifacts(
                    args.run_id,
                    args.training_output,
                    args.rendered_config,
                    args.requirements,
                    args.runs_root,
                )
            )
    except ArtifactError as error:
        print(f"artifact error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
