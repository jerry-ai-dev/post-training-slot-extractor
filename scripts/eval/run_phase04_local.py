from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import httpx
import yaml

from scripts.eval.diff_runs import diff_predictions
from scripts.eval.render_phase04_reports import render_reports
from scripts.eval.select_phase04 import select_winner
from scripts.train.collect_artifacts import verify_artifacts
from scripts.train.package_cloud_artifacts import PHASE04_RUN_IDS

PARENT_PAIRS = (
    ("qwen3-0.6b-sft", "qwen3-0.6b-dpo-b01"),
    ("qwen3-0.6b-sft", "qwen3-0.6b-dpo-b03"),
    ("qwen3-1.7b-sft", "qwen3-1.7b-dpo-b01"),
    ("qwen3-1.7b-sft", "qwen3-1.7b-dpo-b03"),
)


class LocalEvaluationError(RuntimeError):
    """Raised when a local phase04 run cannot be evaluated safely."""


def build_api_config(run_dir: Path) -> dict[str, Any]:
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    adapter = (run_dir / manifest["adapter_path"]).resolve()
    return {
        "model_name_or_path": manifest["base_model"],
        "adapter_name_or_path": str(adapter),
        "finetuning_type": "lora",
        "template": "qwen3",
        "enable_thinking": False,
        "infer_backend": "huggingface",
    }


def is_complete(run_dir: Path) -> bool:
    return all((run_dir / name).is_file() for name in (
        "predictions.jsonl", "scorecard.json", "server.log"
    ))


def _evaluator_python() -> str:
    root = Path(__file__).resolve().parents[2]
    candidate = root / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    return str(candidate) if candidate.is_file() else sys.executable


def _wait_until_ready(url: str, process: subprocess.Popen[Any], timeout_s: float) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise LocalEvaluationError(f"LLaMA-Factory API exited with {process.returncode}")
        try:
            response = httpx.get(url, timeout=2.0)
            if response.is_success:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise LocalEvaluationError(f"LLaMA-Factory API health check timed out: {url}")


def _stop(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=15)


def evaluate_run(
    run_id: str,
    run_dir: Path,
    *,
    cases: Path,
    port: int,
    startup_timeout_s: float,
) -> None:
    verify_artifacts(run_dir, training_only=True)
    runtime_dir = run_dir / ".local-eval"
    runtime_dir.mkdir(exist_ok=True)
    api_config = runtime_dir / "llamafactory-api.yaml"
    api_config.write_text(
        yaml.safe_dump(build_api_config(run_dir), sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    backend_config = runtime_dir / "backend.yaml"
    backend_config.write_text(
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
    log_path = run_dir / "server.log"
    src_dir = str((Path(__file__).resolve().parents[2] / "src").resolve())
    pythonpath = os.environ.get("PYTHONPATH")
    env = {
        **os.environ,
        "API_PORT": str(port),
        "CUDA_VISIBLE_DEVICES": "",
        "PYTHONPATH": os.pathsep.join(filter(None, (src_dir, pythonpath))),
    }
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            ["llamafactory-cli", "api", str(api_config)],
            stdout=log,
            stderr=subprocess.STDOUT,
            env=env,
        )
        try:
            _wait_until_ready(f"http://127.0.0.1:{port}/v1/models", process, startup_timeout_s)
            subprocess.run(
                [
                    _evaluator_python(),
                    "-m",
                    "scripts.eval.collect_analysis",
                    "--backend-config",
                    str(backend_config),
                    "--cases",
                    str(cases),
                    "--run-id",
                    run_id,
                    "--run-dir",
                    str(run_dir),
                    "--evaluation-backend",
                    "llamafactory_huggingface",
                    "--evaluation-device",
                    "cpu",
                ],
                check=True,
                env=env,
            )
        finally:
            _stop(process)


def run_matrix(
    runs_root: Path,
    *,
    cases: Path,
    port: int = 8000,
    startup_timeout_s: float = 900,
    skip_complete: bool = False,
) -> None:
    for run_id in PHASE04_RUN_IDS:
        run_dir = runs_root / f"phase04-{run_id}"
        if skip_complete and is_complete(run_dir):
            print(f"skip complete: {run_id}")
            continue
        print(f"evaluate: {run_id}")
        evaluate_run(
            run_id, run_dir, cases=cases, port=port, startup_timeout_s=startup_timeout_s
        )


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def postprocess(runs_root: Path, reports_root: Path = Path("reports")) -> dict[str, Any]:
    incomplete = [
        run_id
        for run_id in PHASE04_RUN_IDS
        if not is_complete(runs_root / f"phase04-{run_id}")
    ]
    if incomplete:
        names = ", ".join(incomplete)
        raise LocalEvaluationError(f"cannot select winner; incomplete runs: {names}")

    diff_dir = reports_root / "phase04-diffs"
    diff_dir.mkdir(parents=True, exist_ok=True)
    diffs = []
    for parent_id, child_id in PARENT_PAIRS:
        left = _jsonl(runs_root / f"phase04-{parent_id}" / "predictions.jsonl")
        right = _jsonl(runs_root / f"phase04-{child_id}" / "predictions.jsonl")
        result = diff_predictions(
            left, right, left_run_id=parent_id, right_run_id=child_id
        )
        path = diff_dir / f"{parent_id}__to__{child_id}.json"
        path.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        diffs.append(result)

    cards = []
    for run_id in PHASE04_RUN_IDS:
        card = json.loads(
            (runs_root / f"phase04-{run_id}" / "scorecard.json").read_text(encoding="utf-8")
        )
        card.setdefault("stage", "dpo" if "-dpo-" in run_id else "sft")
        card["parameter_billions"] = 0.6 if "0.6b" in run_id else 1.7
        if "-dpo-" in run_id:
            card["parent_run_id"] = run_id.split("-dpo-")[0] + "-sft"
        cards.append(card)
    selection = select_winner(cards)
    selection_path = reports_root / "phase04-selection.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(
        json.dumps(selection, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    render_reports(cards, selection, reports_root, diffs=diffs)
    return selection


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate six phase04 LoRA runs on Windows CPU.")
    parser.add_argument("--runs-root", type=Path, default=Path("experiments/runs"))
    parser.add_argument("--cases", type=Path, default=Path("data/eval/test.jsonl"))
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--startup-timeout-s", type=float, default=900)
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--reports-root", type=Path, default=Path("reports"))
    args = parser.parse_args(argv)
    try:
        run_matrix(
            args.runs_root,
            cases=args.cases,
            port=args.port,
            startup_timeout_s=args.startup_timeout_s,
            skip_complete=args.skip_complete,
        )
        selection = postprocess(args.runs_root, args.reports_root)
        print(f"winner: {selection['winner']}")
    except (
        OSError,
        KeyError,
        ValueError,
        LocalEvaluationError,
        subprocess.SubprocessError,
    ) as error:
        print(f"local evaluation error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
