"""Evaluate every Phase 06 Round 006 GGUF on one frozen cloud GPU setup."""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from scripts.eval.run_phase06_llamacpp import stop, wait_ready


def command_output(command: list[str]) -> str | None:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.STDOUT).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def hardware_metadata(server: str, gpu_layers: int) -> dict[str, Any]:
    return {
        "captured_at": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "memory": command_output(["bash", "-lc", "free -b | head -2"]),
        "gpu": command_output(
            ["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"]
        ),
        "llama_server_version": command_output([server, "--version"]),
        "gpu_layers": gpu_layers,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", type=Path, default=Path("configs/quantization/phase06-round006.yaml")
    )
    parser.add_argument("--quantization", action="append")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    manifest_path = Path(cfg["paths"]["output_dir"]) / "matrix-manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    evaluation = cfg["evaluation"]
    result_root = Path(cfg["paths"]["results_dir"])
    result_root.mkdir(parents=True, exist_ok=True)
    server = cfg["tools"]["server"]
    (result_root / "hardware.json").write_text(
        json.dumps(
            hardware_metadata(server, evaluation["gpu_layers"]), ensure_ascii=False, indent=2
        )
        + "\n",
        encoding="utf-8",
    )
    selected = set(args.quantization or [])
    for artifact in manifest["artifacts"]:
        quant = artifact["quantization"]
        if selected and quant not in selected:
            continue
        model = Path(artifact["path"])
        run_id = f"r004-qwen3-0.6b-sft-{quant}"
        run_root = result_root / run_id
        run_root.mkdir(parents=True, exist_ok=True)
        backend = run_root / "backend.yaml"
        base_url = f"http://127.0.0.1:{evaluation['port']}/v1"
        backend.write_text(
            yaml.safe_dump(
                {
                    "backend": "llama_server",
                    "model": run_id,
                    "base_url": base_url,
                    "api_key": "local-no-key",
                    "temperature": evaluation["temperature"],
                    "max_tokens": evaluation["max_tokens"],
                    "timeout_s": 600,
                },
                sort_keys=False,
            ),
            encoding="utf-8",
        )
        command = [
            server,
            "-m",
            str(model),
            "--host",
            "127.0.0.1",
            "--port",
            str(evaluation["port"]),
            "--n-gpu-layers",
            str(evaluation["gpu_layers"]),
            "--ctx-size",
            str(evaluation["context_size"]),
            "--jinja",
        ]
        with (run_root / "server.log").open("w", encoding="utf-8") as log:
            process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT, text=True)
            try:
                wait_ready(process, base_url, timeout_s=300)
                for cases_value in evaluation["cases"]:
                    cases = Path(cases_value)
                    split = "holdout" if "holdout" in cases.name else "main"
                    subprocess.run(
                        [
                            sys.executable,
                            "-m",
                            "scripts.eval.collect_analysis",
                            "--backend-config",
                            str(backend),
                            "--cases",
                            str(cases),
                            "--run-id",
                            run_id,
                            "--run-dir",
                            str(run_root / split),
                            "--evaluation-backend",
                            "llama_cpp_gguf_gpu_offload",
                            "--evaluation-device",
                            "cuda",
                        ],
                        check=True,
                    )
            finally:
                stop(process)
        print(f"evaluated: {quant}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
