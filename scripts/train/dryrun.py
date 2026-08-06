from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from scripts.train.render_config import atomic_dump_yaml, load_yaml, render_run


@dataclass(frozen=True)
class DryRunJob:
    run_id: str
    config: Path
    output_dir: Path
    log_path: Path


class DryRunError(RuntimeError):
    """Raised when the local training gate does not complete."""


def _first_jsonl(path: Path) -> dict[str, Any]:
    try:
        first = next(line for line in path.read_text(encoding="utf-8").splitlines() if line)
    except (FileNotFoundError, StopIteration) as error:
        raise DryRunError(f"training data is missing or empty: {path}") from error
    payload = json.loads(first)
    if not isinstance(payload, dict):
        raise DryRunError(f"training row must be an object: {path}")
    return payload


def validate_training_data(processed_root: Path, version: str) -> dict[str, int | bool]:
    sft = _first_jsonl(processed_root / "sft" / version / "train.jsonl")
    dpo = _first_jsonl(processed_root / "dpo" / version / "train.jsonl")
    assistant_values = [
        message.get("value", "")
        for message in sft.get("conversations", [])
        if message.get("from") == "gpt"
    ]
    if not assistant_values:
        raise DryRunError("SFT row has no assistant response")
    if any(token in value for value in assistant_values for token in ("<think>", "</think>")):
        raise DryRunError("SFT assistant response contains a thinking token")
    chosen = dpo.get("chosen")
    rejected = dpo.get("rejected")
    if not isinstance(chosen, dict) or not isinstance(rejected, dict):
        raise DryRunError("DPO row must contain chosen and rejected message objects")
    if chosen == rejected:
        raise DryRunError("DPO chosen and rejected messages must differ")
    if any(token in json.dumps(dpo, ensure_ascii=False) for token in ("<think>", "</think>")):
        raise DryRunError("DPO row contains a thinking token")
    return {"sft_rows_checked": 1, "dpo_rows_checked": 1, "no_think": True}


def prepare_smoke_dataset(root: Path) -> Path:
    dataset_root = root / "dataset"
    dataset_root.mkdir(parents=True, exist_ok=True)
    sft = _first_jsonl(Path("data/processed/sft/v0.1/train.jsonl"))
    dpo = _first_jsonl(Path("data/processed/dpo/v0.1/train.jsonl"))
    sft["system"] = "只输出一个 JSON 对象。"
    dpo["system"] = "只输出一个 JSON 对象。"
    (dataset_root / "sft.jsonl").write_text(
        json.dumps(sft, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (dataset_root / "dpo.jsonl").write_text(
        json.dumps(dpo, ensure_ascii=False, separators=(",", ":")) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    tags = {
        "role_tag": "from",
        "content_tag": "value",
        "user_tag": "human",
        "assistant_tag": "gpt",
        "function_tag": "function_call",
        "observation_tag": "observation",
    }
    info = {
        "phase04_dryrun_sft": {
            "file_name": "sft.jsonl",
            "formatting": "sharegpt",
            "columns": {"messages": "conversations", "system": "system", "tools": "tools"},
            "tags": tags,
        },
        "phase04_dryrun_dpo": {
            "file_name": "dpo.jsonl",
            "formatting": "sharegpt",
            "ranking": True,
            "columns": {
                "messages": "conversations",
                "system": "system",
                "tools": "tools",
                "chosen": "chosen",
                "rejected": "rejected",
            },
            "tags": tags,
        },
    }
    (dataset_root / "dataset_info.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return dataset_root


def build_dryrun_jobs(root: Path) -> list[DryRunJob]:
    config_root = root / "configs"
    output_root = root / "outputs"
    log_root = root / "logs"
    dataset_root = prepare_smoke_dataset(root)
    jobs: list[DryRunJob] = []
    for run_id in ("qwen3-0.6b-sft", "qwen3-0.6b-dpo-b01"):
        output_dir = output_root / run_id
        config = render_run(
            run_id,
            output_root=config_root,
            overrides={
                "max_steps": 2,
                "use_cpu": True,
                "bf16": False,
                "fp16": False,
                "output_dir": str(output_dir),
                "overwrite_output_dir": True,
            },
        )
        payload = load_yaml(config)
        payload.update(
            {
                "cutoff_len": 256,
                "per_device_train_batch_size": 1,
                "per_device_eval_batch_size": 1,
                "gradient_accumulation_steps": 1,
                "do_eval": False,
                "eval_strategy": "no",
                "load_best_model_at_end": False,
                "disable_gradient_checkpointing": True,
                "lora_target": "q_proj,v_proj",
                "dataset_dir": str(dataset_root),
                "dataset": (
                    "phase04_dryrun_dpo" if run_id.endswith("dpo-b01") else "phase04_dryrun_sft"
                ),
            }
        )
        payload.pop("eval_dataset", None)
        if run_id.endswith("dpo-b01"):
            payload["adapter_name_or_path"] = str(jobs[0].output_dir)
        atomic_dump_yaml(config, payload)
        jobs.append(DryRunJob(run_id, config, output_dir, log_root / f"{run_id}.log"))
    return jobs


def execute_jobs(jobs: list[DryRunJob]) -> None:
    for job in jobs:
        job.log_path.parent.mkdir(parents=True, exist_ok=True)
        with job.log_path.open("w", encoding="utf-8", newline="\n") as log:
            completed = subprocess.run(
                [sys.executable, "-m", "llamafactory.cli", "train", str(job.config)],
                text=True,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if completed.returncode != 0:
            raise DryRunError(f"{job.run_id} failed; see {job.log_path}")
        if not (job.output_dir / "adapter_config.json").is_file():
            raise DryRunError(f"{job.run_id} did not write adapter_config.json")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the phase04 local CPU training gate.")
    parser.add_argument("--root", type=Path, default=Path(".phase04-dryrun"))
    parser.add_argument("--prepare-only", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        print(json.dumps(validate_training_data(Path("data/processed"), "v0.1")))
        jobs = build_dryrun_jobs(args.root)
        for job in jobs:
            print(f"prepared {job.run_id}: {job.config}")
        if not args.prepare_only:
            execute_jobs(jobs)
    except DryRunError as error:
        print(f"dry-run failed: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
