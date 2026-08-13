from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from slot_extractor.quantization.pipeline import (
    PipelineError,
    PipelinePaths,
    QuantizationPipeline,
)
from slot_extractor.quantization.registry import ModelRegistry, RegistryError
from slot_extractor.quantization.runner import CommandRunner, Toolchain


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the exact Phase 05 Q4_K_M matrix.")
    parser.add_argument("--config", type=Path, default=Path("configs/quantization/phase05.yaml"))
    parser.add_argument("--model-id", action="append")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true")
    parser.add_argument("--anchors", action="store_true")
    parser.add_argument("--summary", type=Path)
    return parser


def _pipeline(config_path: Path) -> QuantizationPipeline:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    tools = payload["toolchain"]
    toolchain = Toolchain(
        resolve=Path(tools["resolve"]),
        merge=Path(tools["merge"]),
        convert_f16=Path(tools["convert_f16"]),
        imatrix=Path(tools["imatrix"]),
        quantize=Path(tools["quantize"]),
        server=Path(tools["server"]),
    )
    paths = PipelinePaths(
        Path(payload["work_root"]), Path(payload["calibration_data"])
    )
    return QuantizationPipeline(
        ModelRegistry.from_config(config_path),
        toolchain,
        CommandRunner(threads=int(payload.get("threads", 8))),
        paths,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    completed: list[str] = []
    failed: list[str] = []
    try:
        pipeline = _pipeline(args.config)
        if args.anchors:
            anchors = pipeline.run_anchors()
            completed.extend(manifest.model_id for manifest in anchors)
        if args.model_id or args.force:
            requested = args.model_id or [
                model.model_id for model in pipeline.registry.quantization_targets()
            ]
            for model_id in requested:
                try:
                    pipeline.run(model_id, force=args.force)
                    completed.append(model_id)
                except (PipelineError, RegistryError):
                    failed.append(model_id)
                    if not args.continue_on_error:
                        break
        else:
            result = pipeline.run_matrix(continue_on_error=args.continue_on_error)
            completed.extend(result.completed)
            failed.extend(result.failed)
    except (OSError, KeyError, TypeError, yaml.YAMLError, RegistryError) as exc:
        failed.append(f"configuration: {exc}")
    summary = {"completed": completed, "failed": failed}
    rendered = json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    if args.summary:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(rendered, encoding="utf-8", newline="\n")
    print(rendered, end="")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
