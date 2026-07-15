# scripts/eval/run_eval.py
from __future__ import annotations

import argparse
from pathlib import Path

from slot_extractor.evaluation.runner import run_evaluation
from slot_extractor.evaluation.scorecard import render_scorecard, write_scorecard_json
from slot_extractor.inference.factory import build_backend_from_config
from slot_extractor.schemas.sample import load_samples


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Slot-Extractor evaluation.")
    parser.add_argument("--backend-config", required=True, help="Path to inference backend YAML.")
    parser.add_argument("--cases", required=True, help="Path to evaluation jsonl cases.")
    parser.add_argument(
        "--report-dir", default="reports/generated", help="Directory for JSON report."
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    samples = load_samples(Path(args.cases))
    backend = build_backend_from_config(Path(args.backend_config))
    scorecard = run_evaluation(samples=samples, backend=backend)
    print(render_scorecard(scorecard))
    report_path = write_scorecard_json(scorecard, args.report_dir)
    print(f"Report JSON: {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
