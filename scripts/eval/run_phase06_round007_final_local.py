"""Benchmark final Q8/Q4 models with real streamed task prompts on local CPU."""

from __future__ import annotations

import argparse
import ctypes
import json
import os
import statistics
import subprocess
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from scripts.eval.run_phase06_llamacpp import stop, wait_ready
from slot_extractor.inference.llama_server import LlamaServerBackend, LlamaServerConfig
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.sample import load_samples

MODELS = {
    "Q8_0": Path("models/gguf/phase06-round006-local/r004-qwen3-0.6b-sft-Q8_0.gguf"),
    "Q4_K_M": Path("models/gguf/phase06-round006-local/r004-qwen3-0.6b-sft-Q4_K_M.gguf"),
}


class PeakMemoryMonitor:
    def __init__(self, pid: int) -> None:
        self.pid = pid
        self.peak = 0
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def _run(self) -> None:
        while not self._stop.wait(0.02):
            try:
                self.peak = max(self.peak, process_rss(self.pid))
            except OSError:
                return

    def __enter__(self) -> PeakMemoryMonitor:
        self.thread.start()
        return self

    def __exit__(self, *_: object) -> None:
        self._stop.set()
        self.thread.join(timeout=1)


def process_rss(pid: int) -> int:
    if os.name != "nt":
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
        line = next(line for line in status.splitlines() if line.startswith("VmRSS:"))
        return int(line.split()[1]) * 1024

    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    kernel32 = ctypes.windll.kernel32
    handle = kernel32.OpenProcess(0x0410, False, pid)
    if not handle:
        raise OSError(f"cannot open process {pid}")
    try:
        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        if not ctypes.windll.psapi.GetProcessMemoryInfo(
            handle, ctypes.byref(counters), counters.cb
        ):
            raise OSError(f"cannot read process {pid} memory")
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)


def representative_messages(count: int) -> list[tuple[str, list[dict[str, Any]]]]:
    builder = PromptBuilder()
    candidates = []
    for path in (Path("data/eval/test.jsonl"), Path("data/eval/phase06_holdout_v0.3.jsonl")):
        for sample in load_samples(path):
            messages = builder.build_messages(sample)
            length = sum(len(str(message.get("content", ""))) for message in messages)
            candidates.append((length, sample.id, messages))
    candidates.sort(key=lambda item: item[0])
    if count == 1:
        middle = candidates[len(candidates) // 2]
        return [(middle[1], middle[2])]
    indices = [round(index * (len(candidates) - 1) / (count - 1)) for index in range(count)]
    return [(candidates[index][1], candidates[index][2]) for index in indices]


def median(values: list[float | None]) -> float | None:
    present = [value for value in values if value is not None]
    return statistics.median(present) if present else None


def benchmark_model(
    quant: str,
    model: Path,
    messages: list[tuple[str, list[dict[str, Any]]]],
    *,
    server: Path,
    port: int,
    threads: int,
    repetitions: int,
    output: Path,
) -> dict[str, Any]:
    if not model.is_file():
        raise FileNotFoundError(model)
    base_url = f"http://127.0.0.1:{port}/v1"
    log = output / f"{quant}-server.log"
    with log.open("w", encoding="utf-8") as stream:
        process = subprocess.Popen(
            [
                str(server),
                "-m",
                str(model),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--threads",
                str(threads),
                "--ctx-size",
                "4096",
                "--n-gpu-layers",
                "0",
                "--no-cache-prompt",
                "--jinja",
            ],
            stdout=stream,
            stderr=subprocess.STDOUT,
            text=True,
        )
        try:
            wait_ready(process, base_url, timeout_s=180)
            backend = LlamaServerBackend(
                LlamaServerConfig(quant, base_url, temperature=0.0, max_tokens=512, timeout_s=300)
            )
            backend.generate(messages[0][1])  # warm model and kernels; excluded from results
            rows = []
            with PeakMemoryMonitor(process.pid) as memory:
                for sample_id, prompt in messages:
                    for repetition in range(repetitions):
                        result = backend.generate(prompt)
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "repetition": repetition + 1,
                                "input_tokens": result.input_tokens,
                                "output_tokens": result.output_tokens,
                                "prefill_ms": result.prefill_ms,
                                "prefill_tokens_per_s": result.prefill_tokens_per_s,
                                "ttft_ms": result.first_token_ms,
                                "decode_ms": result.decode_ms,
                                "decode_tokens_per_s": result.decode_tokens_per_s,
                                "total_ms": result.total_ms,
                            }
                        )
            return {
                "quantization": quant,
                "model": str(model),
                "model_bytes": model.stat().st_size,
                "peak_server_rss_bytes": memory.peak,
                "rows": rows,
                "median": {
                    key: median([row[key] for row in rows])
                    for key in (
                        "prefill_ms",
                        "prefill_tokens_per_s",
                        "ttft_ms",
                        "decode_ms",
                        "decode_tokens_per_s",
                        "total_ms",
                    )
                },
            }
        finally:
            stop(process)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server", type=Path, default=Path("deployment/llama_cpp/bin/llama-server.exe")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("experiments/phase06/round-007/local-final")
    )
    parser.add_argument("--port", type=int, default=8027)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    prompts = representative_messages(args.samples)
    results = [
        benchmark_model(
            quant,
            model,
            prompts,
            server=args.server,
            port=args.port,
            threads=args.threads,
            repetitions=args.repetitions,
            output=args.output,
        )
        for quant, model in MODELS.items()
    ]
    document = {
        "created_at": datetime.now(UTC).isoformat(),
        "method": {
            "samples": args.samples,
            "repetitions": args.repetitions,
            "threads": args.threads,
            "prompt_cache": False,
            "timing": "cold-prompt streamed HTTP; one excluded warm-up request",
        },
        "results": results,
    }
    target = args.output / "benchmark.json"
    target.write_text(json.dumps(document, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(target)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
