from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any

from scripts.eval.phase04_artifacts import write_phase04_artifacts
from slot_extractor.evaluation.runner import default_scorers
from slot_extractor.evaluation.scenarios import aggregate_scenario_slices
from slot_extractor.evaluation.scorecard import aggregate_scorecard, summarize_timing
from slot_extractor.schemas.results import CaseResult, GenerationResult
from slot_extractor.schemas.sample import load_samples

DEFAULT_ROUND_ID = "round-001"


def _run_ids(round_id: str) -> tuple[str, str]:
    number = int(round_id.removeprefix("round-"))
    prefix = f"r{number:03d}"
    return (f"{prefix}-qwen3-0.6b-sft", f"{prefix}-qwen3-1.7b-sft")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_cloud_package(source: Path, round_id: str = DEFAULT_ROUND_ID) -> int:
    checksum_file = source / f"experiments/phase06/{round_id}/cloud-results/SHA256SUMS"
    if not checksum_file.is_file():
        raise ValueError(f"missing checksum manifest: {checksum_file}")
    checked = 0
    for line in checksum_file.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split(maxsplit=1)
        path = source / relative.removeprefix("./")
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"cloud artifact checksum mismatch: {relative}")
        checked += 1
    return checked


def _rescore(run_id: str, source: Path, destination: Path, cases: Path) -> None:
    samples = load_samples(cases)
    sample_by_id = {sample.id: sample for sample in samples}
    raw_rows = [
        json.loads(line)
        for line in (source / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    scorers = default_scorers()
    records: list[dict[str, Any]] = []
    scored_cases: list[CaseResult] = []
    for row in raw_rows:
        sample = sample_by_id[row["id"]]
        timing = row.get("timing", {})
        generation = GenerationResult(
            text=row["model_output"],
            model=run_id,
            prefill_ms=None,
            first_token_ms=timing.get("first_token_ms"),
            total_ms=timing.get("total_ms") or 0.0,
            tokens_per_s=timing.get("tokens_per_s"),
        )
        dimensions = {
            scorer.dimension: scorer.score(sample, generation)
            for scorer in scorers
            if scorer.applies_to(sample)
        }
        dimension_payload: dict[str, dict[str, Any]] = {}
        for name, score in dimensions.items():
            payload: dict[str, Any] = {
                "score": score.score,
                "passed": score.passed,
                "detail": score.detail,
            }
            if name == "task_correctness":
                payload.update(json.loads(score.detail))
            dimension_payload[name] = payload
        records.append({**row, "dimensions": dimension_payload})
        scored_cases.append(
            CaseResult(
                sample_id=sample.id,
                output_kind=sample.output_kind,
                conversation_kind=sample.conversation_kind,
                model_output=generation.text,
                dimensions=dimensions,
                total_ms=timing.get("total_ms"),
                first_token_ms=timing.get("first_token_ms"),
                tokens_per_s=timing.get("tokens_per_s"),
            )
        )
    scorecard = aggregate_scorecard(model=run_id, cases=scored_cases)
    timing = summarize_timing(scored_cases)
    task_scores = {
        record["id"]: record["dimensions"]["task_correctness"]["score"]
        for record in records
    }
    analysis = {
        "model": run_id,
        "backend_config": str(source / "runtime/backend.yaml"),
        "cases_path": str(cases),
        "aggregate_dimensions": {
            name: {"score": score.score, "passed": score.passed, "detail": score.detail}
            for name, score in scorecard.dimensions.items()
        },
        "aggregate_timing": {
            key: getattr(timing, key)
            for key in (
                "count",
                "total_ms_mean",
                "total_ms_p50",
                "total_ms_p95",
                "total_ms_max",
                "total_ms_min",
                "first_token_ms_mean",
                "tokens_per_s_mean",
            )
        },
        "scenario_slices": aggregate_scenario_slices(samples, task_scores),
        "records": records,
    }
    write_phase04_artifacts(
        analysis,
        destination,
        run_id=run_id,
        evaluation_environment={
            "backend": "llamafactory_huggingface",
            "device": "cuda",
            "latency_comparable_to_m0": False,
            "rescored_offline": True,
        },
    )


def import_results(
    source: Path,
    destination: Path,
    cases: Path,
    *,
    round_id: str = DEFAULT_ROUND_ID,
) -> dict[str, Any]:
    checked = verify_cloud_package(source, round_id)
    cloud = source / f"experiments/phase06/{round_id}/cloud-results"
    models = source / "models/adapters"
    destination.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for run_id in _run_ids(round_id):
        run_destination = destination / run_id
        run_destination.mkdir(exist_ok=True)
        _rescore(run_id, cloud / run_id, run_destination, cases)
        packaged_training = cloud / run_id / "training"
        training_source = packaged_training if packaged_training.is_dir() else models / run_id
        for name in (
            "train_results.json",
            "eval_results.json",
            "trainer_state.json",
            "trainer_log.jsonl",
            "training_loss.png",
            "training_eval_loss.png",
        ):
            path = training_source / name
            if path.is_file():
                shutil.copy2(path, run_destination / name)
        adapter = models / run_id / "adapter_model.safetensors"
        adapter_in_package = adapter.is_file()
        runs.append(
            {
                "run_id": run_id,
                "adapter_in_package": adapter_in_package,
                "adapter_sha256": _sha256(adapter) if adapter_in_package else None,
                "adapter_bytes": adapter.stat().st_size if adapter_in_package else None,
                "adapter_committed": False,
            }
        )
    for name in (
        "git-commit.txt",
        "git-status.txt",
        "pip-freeze.txt",
        "nvidia-smi.txt",
        "nvidia-smi-final.txt",
        "training-started-at.txt",
        "training-finished-at.txt",
    ):
        shutil.copy2(cloud / name, destination / name)
    manifest = {
        "schema_version": 1,
        "round_id": round_id,
        "verified_cloud_files": checked,
        "source_package": source.name,
        "policy": "Metrics and provenance are committed; adapters and checkpoints stay external.",
        "runs": runs,
    }
    (destination / "import-manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify and import compact Phase 06 cloud results."
    )
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--round-id", default=DEFAULT_ROUND_ID)
    parser.add_argument(
        "--destination",
        type=Path,
        default=None,
    )
    parser.add_argument("--cases", type=Path, default=Path("data/eval/test.jsonl"))
    args = parser.parse_args()
    destination = args.destination or Path(f"experiments/phase06/{args.round_id}/imported")
    print(
        json.dumps(
            import_results(
                args.source,
                destination,
                args.cases,
                round_id=args.round_id,
            ),
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
