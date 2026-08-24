"""Rescore stored predictions without invoking a model again."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from scripts.eval.phase04_artifacts import EFFECTIVE_TASK_THRESHOLD
from slot_extractor.evaluation.runner import default_scorers
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.schemas.sample import load_samples


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--cases", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args(argv)

    stored = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in args.predictions.read_text(encoding="utf-8").splitlines()
        )
    }
    records = []
    for sample in load_samples(args.cases):
        old = stored[sample.id]
        generation = GenerationResult(
            text=old["model_output"],
            model=args.run_id,
            prefill_ms=None,
            first_token_ms=old["timing"].get("first_token_ms"),
            total_ms=old["timing"].get("total_ms"),
            output_tokens=None,
            tokens_per_s=old["timing"].get("tokens_per_s"),
            raw={},
        )
        dimensions = {}
        for scorer in default_scorers():
            if scorer.applies_to(sample):
                score = scorer.score(sample, generation)
                payload = {"score": score.score, "passed": score.passed, "detail": score.detail}
                if scorer.dimension == "task_correctness":
                    payload.update(json.loads(score.detail))
                dimensions[scorer.dimension] = payload
        records.append({**old, "dimensions": dimensions})

    for record in records:
        protocol = record["dimensions"]["protocol"]["score"] == 1.0
        task = record["dimensions"]["task_correctness"]["score"]
        record["effective_pass"] = protocol and task >= EFFECTIVE_TASK_THRESHOLD
        record["failure_reasons"] = [
            reason
            for reason, failed in (
                ("protocol", not protocol),
                ("task_correctness", task < EFFECTIVE_TASK_THRESHOLD),
            )
            if failed
        ]
    args.run_dir.mkdir(parents=True, exist_ok=True)
    args.predictions.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
        newline="\n",
    )
    scorecard_path = args.run_dir / "scorecard.json"
    scorecard = json.loads(scorecard_path.read_text(encoding="utf-8"))
    for dimension in ("protocol", "task_correctness"):
        values = [record["dimensions"][dimension]["score"] for record in records]
        scorecard["aggregate_dimensions"][dimension]["score"] = sum(values) / len(values)
    numerator = sum(record["effective_pass"] for record in records)
    scorecard["effective_pass"] = {
        "numerator": numerator,
        "denominator": len(records),
        "rate": numerator / len(records),
    }
    scorecard["evaluation_environment"]["rescored_with"] = "phase06-semantic-calibrated-v2"
    scorecard_path.write_text(
        json.dumps(scorecard, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
