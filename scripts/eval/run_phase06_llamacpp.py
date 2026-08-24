"""Run the frozen Phase 06 finalists locally with llama.cpp and GGUF LoRA adapters."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml


@dataclass(frozen=True)
class Candidate:
    run_id: str
    base: Path
    lora: Path


CANDIDATES = (
    Candidate(
        "r004-qwen3-0.6b-sft",
        Path("models/gguf/Qwen3-0.6B-Q8_0.gguf"),
        Path("models/gguf/r004-qwen3-0.6b-sft-lora-f16.gguf"),
    ),
    Candidate(
        "r003-qwen3-1.7b-sft",
        Path("models/gguf/Qwen3-1.7B-Q8_0.gguf"),
        Path("models/gguf/r003-qwen3-1.7b-sft-lora-f16.gguf"),
    ),
)


def wait_ready(process: subprocess.Popen[str], url: str, timeout_s: float = 120) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"llama-server exited with code {process.returncode}")
        try:
            if httpx.get(f"{url}/models", timeout=2).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    raise RuntimeError("llama-server startup timed out")


def stop(process: subprocess.Popen[str]) -> None:
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=15)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def evaluate(
    candidate: Candidate, *, server: Path, results: Path, port: int, threads: int
) -> None:
    for required in (server, candidate.base, candidate.lora):
        if not required.is_file():
            raise FileNotFoundError(required)
    model_root = results / candidate.run_id
    model_root.mkdir(parents=True, exist_ok=True)
    backend = model_root / "backend.yaml"
    base_url = f"http://127.0.0.1:{port}/v1"
    backend.write_text(
        yaml.safe_dump(
            {
                "backend": "llama_server",
                "model": candidate.run_id,
                "base_url": base_url,
                "api_key": "local-no-key",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout_s": 300,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
    with (model_root / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                str(server),
                "-m",
                str(candidate.base),
                "--lora",
                str(candidate.lora),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--threads",
                str(threads),
                "--jinja",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
            text=True,
        )
        try:
            wait_ready(process, base_url)
            for split, cases in (
                ("main", Path("data/eval/test.jsonl")),
                ("holdout", Path("data/eval/phase06_holdout_v0.3.jsonl")),
            ):
                run_dir = model_root / split
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
                        candidate.run_id,
                        "--run-dir",
                        str(run_dir),
                        "--evaluation-backend",
                        "llama_cpp_q8_base_plus_f16_lora",
                        "--evaluation-device",
                        "cpu",
                    ],
                    check=True,
                    env=env,
                )
        finally:
            stop(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", type=Path, default=Path("deployment/llama_cpp/bin/llama-server.exe")
    )
    parser.add_argument(
        "--results", type=Path, default=Path("experiments/phase06/round-005/results/llamacpp")
    )
    parser.add_argument("--port", type=int, default=8010)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    try:
        for candidate in CANDIDATES:
            print(f"evaluate: {candidate.run_id}", flush=True)
            evaluate(
                candidate,
                server=args.server,
                results=args.results,
                port=args.port,
                threads=args.threads,
            )
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        print(f"local llama.cpp evaluation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
