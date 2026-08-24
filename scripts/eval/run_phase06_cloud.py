from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import yaml


class CloudEvaluationError(RuntimeError):
    """Raised when a Phase 06 cloud evaluation cannot complete safely."""


def load_runs(plan: Path) -> list[dict[str, Any]]:
    payload = yaml.safe_load(plan.read_text(encoding="utf-8"))
    runs = payload.get("runs") if isinstance(payload, dict) else None
    if not isinstance(runs, list) or not runs:
        raise CloudEvaluationError(f"run plan has no runs: {plan}")
    return runs


def api_config(run: dict[str, Any]) -> dict[str, Any]:
    adapter = Path(run["output_dir"])
    if not (adapter / "adapter_config.json").is_file():
        raise CloudEvaluationError(f"trained adapter is missing: {adapter}")
    return {
        "model_name_or_path": run["model"]["name"],
        "adapter_name_or_path": str(adapter.resolve()),
        "finetuning_type": "lora",
        "template": "qwen3",
        "enable_thinking": False,
        "infer_backend": "huggingface",
    }


def _wait(url: str, process: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise CloudEvaluationError(f"API exited with code {process.returncode}")
        try:
            if httpx.get(url, timeout=3).is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(2)
    raise CloudEvaluationError(f"API startup timed out after {timeout_s}s")


def _stop(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def evaluate(
    run: dict[str, Any],
    *,
    cases: Path,
    results_root: Path,
    port: int,
    timeout_s: float,
    cli: str = "llamafactory-cli",
    device: str = "cuda",
    evaluator_python: str = sys.executable,
) -> None:
    run_id = run["run_id"]
    run_dir = results_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    runtime = run_dir / "runtime"
    runtime.mkdir(exist_ok=True)
    api_path = runtime / "llamafactory-api.yaml"
    api_path.write_text(
        yaml.safe_dump(api_config(run), sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    backend_path = runtime / "backend.yaml"
    backend_path.write_text(
        yaml.safe_dump(
            {
                "backend": "llama_server",
                "model": run_id,
                "base_url": f"http://127.0.0.1:{port}/v1",
                "api_key": "local-no-key",
                "temperature": 0.0,
                "max_tokens": 512,
                "timeout_s": 600,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    env = {**os.environ, "API_PORT": str(port)}
    if device == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = ""
    with (run_dir / "server.log").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [cli, "api", str(api_path)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            _wait(f"http://127.0.0.1:{port}/v1/models", process, timeout_s)
            subprocess.run(
                [
                    evaluator_python,
                    "-m",
                    "scripts.eval.collect_analysis",
                    "--backend-config",
                    str(backend_path),
                    "--cases",
                    str(cases),
                    "--run-id",
                    run_id,
                    "--run-dir",
                    str(run_dir),
                    "--evaluation-backend",
                    "llamafactory_huggingface",
                    "--evaluation-device",
                    device,
                ],
                check=True,
                env=env,
            )
        finally:
            _stop(process)
    metadata = {
        "run_id": run_id,
        "evaluated_at": datetime.now(UTC).isoformat(),
        "dataset": str(cases),
        "adapter": run["output_dir"],
        "base_model": run["model"]["name"],
    }
    (run_dir / "evaluation.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate Phase 06 LoRA adapters on cloud GPU.")
    parser.add_argument(
        "--plan", type=Path, default=Path("experiments/phase06/round-001/package/run-plan.yaml")
    )
    parser.add_argument("--cases", type=Path, default=Path("data/eval/test.jsonl"))
    parser.add_argument(
        "--results-root",
        type=Path,
        default=Path("experiments/phase06/round-001/cloud-results"),
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-s", type=float, default=900)
    parser.add_argument("--run-id", action="append", default=[])
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--llamafactory-cli", default="llamafactory-cli")
    parser.add_argument("--evaluator-python", default=sys.executable)
    args = parser.parse_args(argv)
    try:
        selected = set(args.run_id)
        runs = [r for r in load_runs(args.plan) if not selected or r["run_id"] in selected]
        if selected - {r["run_id"] for r in runs}:
            raise CloudEvaluationError("unknown --run-id in run plan")
        for run in runs:
            print(f"evaluate: {run['run_id']}", flush=True)
            evaluate(
                run,
                cases=args.cases,
                results_root=args.results_root,
                port=args.port,
                timeout_s=args.startup_timeout_s,
                cli=args.llamafactory_cli,
                device=args.device,
                evaluator_python=args.evaluator_python,
            )
    except (CloudEvaluationError, OSError, subprocess.SubprocessError) as error:
        print(f"cloud evaluation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
