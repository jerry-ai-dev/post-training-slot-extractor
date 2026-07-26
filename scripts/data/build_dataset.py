from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any

import yaml

from slot_extractor.data.dataset_build import build_dataset
from slot_extractor.data.generator import (
    GenerationRequest,
    RawGenerator,
    generation_sample_id,
)
from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record
from slot_extractor.inference.factory import build_backend_from_config
from slot_extractor.utils.jsonl import read_jsonl, write_jsonl


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build the phase03 SFT/DPO datasets")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-root")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--mock", action="store_true")
    mode.add_argument("--generate", action="store_true")
    mode.add_argument("--raw-input")
    parser.add_argument("--strict-audit", action="store_true")
    return parser


def _load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as file:
        value = yaml.safe_load(file)
    if not isinstance(value, dict):
        raise ValueError("phase03 config must be an object")
    return value


def _generation_requests(config: dict[str, Any]) -> list[GenerationRequest]:
    if "scenarios" not in config:
        return [
            GenerationRequest(category, index)
            for category, count in config["counts"].items()
            for index in range(1, int(count) + 1)
        ]
    requests: list[GenerationRequest] = []
    for category, count in config["counts"].items():
        scenarios = config["scenarios"][category]
        if not scenarios:
            raise ValueError(f"{category} must define at least one scenario")
        requests.extend(
            GenerationRequest(
                category,
                index,
                str(scenarios[(index - 1) % len(scenarios)]),
            )
            for index in range(1, int(count) + 1)
        )
    return requests


def _generate(
    config: dict[str, Any],
    backend_key: str,
    checkpoint_path: Path | None = None,
) -> list[RawSample]:
    backend = build_backend_from_config(config[backend_key])
    requests = _generation_requests(config)
    order = [generation_sample_id(request) for request in requests]
    completed: dict[str, RawSample] = {}
    if checkpoint_path and checkpoint_path.exists():
        completed = {
            sample.id: sample
            for sample in (raw_sample_from_record(record) for record in read_jsonl(checkpoint_path))
            if sample.id in order
        }
    pending = [request for request in requests if generation_sample_id(request) not in completed]
    total = len(requests)
    if completed:
        print(f"resumed={len(completed)}/{total}", flush=True)
    concurrency = int(config.get("generation_concurrency", 1))
    generator = RawGenerator(backend)
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = {executor.submit(generator.generate_one, request): request for request in pending}
        for future in as_completed(futures):
            sample = future.result()
            completed[sample.id] = sample
            if checkpoint_path:
                write_jsonl(
                    checkpoint_path,
                    [asdict(completed[sample_id]) for sample_id in order if sample_id in completed],
                )
            print(f"generated={len(completed)}/{total} id={sample.id}", flush=True)
    return [completed[sample_id] for sample_id in order]


def run(args: argparse.Namespace) -> int:
    config = _load_config(args.config)
    output_root = Path(args.output_root or config["output_root"])
    checkpoint = (
        output_root
        / "raw"
        / str(config["version"])
        / str(config.get("checkpoint_name", ".generation_checkpoint.jsonl"))
    )
    if args.raw_input:
        samples = [raw_sample_from_record(row) for row in read_jsonl(args.raw_input)]
    elif args.generate:
        samples = _generate(config, "generate_inference_config", checkpoint)
    else:
        samples = _generate(config, "mock_inference_config")
    eval_records = list(read_jsonl(config["eval_path"]))
    result = build_dataset(
        samples,
        eval_records,
        output_root,
        str(config["version"]),
        int(config["seed"]),
        {key: float(value) for key, value in config["hard_case_thresholds"].items()},
        {key: int(value) for key, value in config.get("dpo_target_counts", {}).items()} or None,
        strict_audit=args.strict_audit,
    )
    if args.generate and checkpoint.exists():
        checkpoint.unlink()
    print(
        f"raw={result.raw_count}, sft_train={result.train_count}, "
        f"sft_val={result.val_count}, dpo_train={result.dpo_train_count}, "
        f"dpo_val={result.dpo_val_count}"
    )
    print("eval_overlap=0")
    for path in (
        result.raw_samples,
        result.sft_train,
        result.sft_val,
        result.dpo_train,
        result.dpo_val,
        result.dataset_info,
        result.version_card,
    ):
        print(path)
    if result.audit.deficits:
        print(f"audit_deficits={sorted(result.audit.deficits)}")
    return 0


def main() -> int:
    try:
        return run(_parser().parse_args())
    except Exception as exc:
        print(f"phase03 build failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
