from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

from scripts.train.collect_artifacts import ArtifactError, verify_artifacts

PHASE04_RUN_IDS = (
    "qwen3-0.6b-sft",
    "qwen3-1.7b-sft",
    "qwen3-0.6b-dpo-b01",
    "qwen3-0.6b-dpo-b03",
    "qwen3-1.7b-dpo-b01",
    "qwen3-1.7b-dpo-b03",
)


class PackageError(RuntimeError):
    """Raised when the cloud training package is incomplete."""


def package_runs(runs_root: Path, output: Path) -> Path:
    run_dirs = [runs_root / f"phase04-{run_id}" for run_id in PHASE04_RUN_IDS]
    missing = [path.name for path in run_dirs if not path.is_dir()]
    if missing:
        raise PackageError(f"missing run directories: {', '.join(missing)}")
    try:
        for run_dir in run_dirs:
            verify_artifacts(run_dir, training_only=True)
            if not any((run_dir / "adapter").glob("adapter_model.*")):
                raise PackageError(f"missing adapter weights: {run_dir.name}")
    except ArtifactError as error:
        raise PackageError(str(error)) from error

    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_suffix(output.suffix + ".tmp")
    staging.unlink(missing_ok=True)
    try:
        with zipfile.ZipFile(staging, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for run_dir in run_dirs:
                for path in sorted(run_dir.rglob("*")):
                    if path.is_file():
                        archive.write(path, path.relative_to(runs_root).as_posix())
        staging.replace(output)
    finally:
        staging.unlink(missing_ok=True)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Package six phase04 cloud training runs.")
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--out", type=Path, default=Path("phase04-training-artifacts.zip"))
    args = parser.parse_args(argv)
    try:
        print(package_runs(args.runs_root, args.out))
    except PackageError as error:
        print(f"package error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
