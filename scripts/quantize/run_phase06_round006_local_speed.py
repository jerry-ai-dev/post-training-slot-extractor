"""Build speed-only 0.6B quants and benchmark CPU prefill/decode with llama-bench."""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml


def run(command: list[str]) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(command, text=True, capture_output=True)
    if completed.returncode:
        raise RuntimeError(f"command failed: {' '.join(command)}\n{completed.stderr}")
    return completed


def quantize_matrix(cfg: dict[str, Any]) -> list[tuple[str, Path]]:
    local = cfg["local_speed"]
    source = Path(local["source_f16"])
    imatrix = Path(local["source_imatrix"])
    output = Path(local["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    if not source.is_file():
        raise FileNotFoundError(source)
    artifacts = [("F16", source)]
    for quant_value in cfg["quantizations"]:
        quant = str(quant_value)
        if quant == "F16":
            continue
        target = output / f"qwen3-0.6b-{quant}-speed-only.gguf"
        if not target.is_file():
            temporary = target.with_suffix(target.suffix + ".partial")
            temporary.unlink(missing_ok=True)
            command = [local["quantize_exe"]]
            if quant.startswith("IQ"):
                if not imatrix.is_file():
                    raise FileNotFoundError(imatrix)
                command += ["--imatrix", str(imatrix)]
            command += [str(source), str(temporary), quant]
            run(command)
            temporary.replace(target)
        artifacts.append((quant, target))
    return artifacts


def benchmark(cfg: dict[str, Any], artifacts: list[tuple[str, Path]]) -> Path:
    local = cfg["local_speed"]
    results = Path(local["results_dir"])
    results.mkdir(parents=True, exist_ok=True)
    target = results / "benchmark.json"
    rows: list[dict[str, Any]] = []
    if target.is_file():
        rows = json.loads(target.read_text(encoding="utf-8")).get("rows", [])
    completed = {(row["quantization"], row["phase"], row["tokens"]) for row in rows}

    def save() -> None:
        document = {
            "purpose": "CPU speed only; quality is measured with the final r004 adapter on cloud",
            "created_at": datetime.now(UTC).isoformat(),
            "platform": platform.platform(),
            "processor": platform.processor(),
            "rows": rows,
        }
        target.write_text(
            json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    for quant, model in artifacts:
        tests = [("prefill", value) for value in local["prompt_tokens"]]
        tests += [("decode", value) for value in local["generation_tokens"]]
        for phase, tokens in tests:
            if (quant, phase, tokens) in completed:
                continue
            command = [
                local["bench_exe"],
                "-m",
                str(model),
                "-p",
                str(tokens if phase == "prefill" else 0),
                "-n",
                str(tokens if phase == "decode" else 0),
                "-t",
                str(local["threads"]),
                "-r",
                str(local["repetitions"]),
                "-o",
                "json",
            ]
            payload = json.loads(run(command).stdout)
            for item in payload:
                item_phase = "prefill" if item["n_prompt"] else "decode"
                if item_phase != phase:
                    continue
                rows.append(
                    {
                        "quantization": quant,
                        "model_bytes": model.stat().st_size,
                        "phase": phase,
                        "tokens": tokens,
                        "tokens_per_second": item["avg_ts"],
                        "stddev_tokens_per_second": item["stddev_ts"],
                        "threads": item["n_threads"],
                        "raw": item,
                    }
                )
            save()
            print(f"benchmarked: {quant} {phase}{tokens}", flush=True)
    return target


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/quantization/phase06-round006.yaml")
    )
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    artifacts = quantize_matrix(cfg)
    if not args.prepare_only:
        print(benchmark(cfg, artifacts))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
