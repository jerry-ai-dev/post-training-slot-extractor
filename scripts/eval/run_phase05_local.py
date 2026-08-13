"""Sequential Phase 05 local evaluation entrypoint."""

import argparse
import json
import platform
import sys
from dataclasses import asdict, dataclass, is_dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any

import httpx
import yaml

from scripts.eval.phase05_artifacts import (
    models_to_run,
    write_failure,
    write_local_marker,
    write_matrix_summary,
    write_model_result,
)
from scripts.eval.phase05_metrics import WorkloadSample, aggregate_workload
from scripts.eval.phase05_reports import render_phase05_reports
from slot_extractor.evaluation.runner import default_scorers
from slot_extractor.evaluation.scenarios import aggregate_scenario_slices
from slot_extractor.evaluation.scorecard import aggregate_scorecard
from slot_extractor.inference.llama_server import LlamaServerBackend, LlamaServerConfig
from slot_extractor.inference.llama_server_manager import LlamaServerManager
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.quantization.manifest import read_and_verify_manifest
from slot_extractor.quantization.registry import ModelRegistry
from slot_extractor.schemas.results import CaseResult, DimensionScore
from slot_extractor.schemas.sample import load_samples


@dataclass(frozen=True)
class Phase05Config:
    registry: Path
    manifests_root: Path
    reports_root: Path
    quality_cases: Path
    q4_model_ids: tuple[str, ...]
    f16_anchor_ids: tuple[str, ...]
    workloads: tuple[str, ...]
    include_8k: bool
    warmup_requests: int
    repetitions: int
    execution: str


def load_phase05_config(path: Path) -> Phase05Config:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    config = Phase05Config(
        registry=Path(payload["registry"]),
        manifests_root=Path(payload["manifests_root"]),
        reports_root=Path(payload["reports_root"]),
        quality_cases=Path(payload["quality_cases"]),
        q4_model_ids=tuple(payload["q4_model_ids"]),
        f16_anchor_ids=tuple(payload["f16_anchor_ids"]),
        workloads=tuple(payload["workloads"]),
        include_8k=bool(payload["include_8k"]),
        warmup_requests=int(payload["warmup_requests"]),
        repetitions=int(payload["repetitions"]),
        execution=payload["execution"],
    )
    registry = ModelRegistry.from_config(config.registry)
    if config.q4_model_ids != tuple(model.model_id for model in registry.quantization_targets()):
        raise ValueError("q4_model_ids must exactly match the canonical registry")
    if config.f16_anchor_ids != tuple(model.model_id for model in registry.anchors()):
        raise ValueError("f16_anchor_ids must exactly match the canonical registry")
    if config.workloads != ("short", "medium", "2k", "4k"):
        raise ValueError("required workloads are short, medium, 2k and 4k")
    if config.execution != "windows_cpu_sequential":
        raise ValueError("Phase 05 evaluation must be windows_cpu_sequential")
    return config


@dataclass(frozen=True)
class MatrixSummary:
    completed_model_ids: tuple[str, ...]
    failed_model_ids: tuple[str, ...]


def manifest_payload(manifest: Any) -> dict[str, Any]:
    return asdict(manifest) if is_dataclass(manifest) else dict(manifest)


def build_quality_backend(base_url: str, model_id: str) -> LlamaServerBackend:
    return LlamaServerBackend(
        LlamaServerConfig(model=model_id, base_url=base_url, timeout_s=180)
    )


def evaluate_quality(base_url: str, config: Phase05Config, model_id: str) -> dict[str, Any]:
    samples = load_samples(config.quality_cases)
    backend = build_quality_backend(base_url, model_id)
    prompt_builder = PromptBuilder()
    scorers = default_scorers()
    records: list[dict[str, Any]] = []
    cases: list[CaseResult] = []
    for sample in samples:
        messages = prompt_builder.build_messages(sample)
        generation = backend.generate(messages)
        dimensions: dict[str, DimensionScore] = {}
        dimension_payload: dict[str, dict[str, Any]] = {}
        for scorer in scorers:
            if not scorer.applies_to(sample):
                continue
            score = scorer.score(sample, generation)
            dimensions[scorer.dimension] = score
            payload = asdict(score)
            if scorer.dimension == "task_correctness":
                payload.update(json.loads(score.detail))
            dimension_payload[scorer.dimension] = payload
        cases.append(
            CaseResult(
                sample.id,
                sample.output_kind,
                sample.conversation_kind,
                generation.text,
                dimensions,
                generation.total_ms,
                generation.first_token_ms,
                generation.tokens_per_s,
            )
        )
        records.append(
            {
                "id": sample.id,
                "output_kind": sample.output_kind,
                "conversation_kind": sample.conversation_kind,
                "tags": sample.tags,
                "input": sample.input,
                "messages_sent": messages,
                "expected": sample.expected,
                "model_output": generation.text,
                "dimensions": dimension_payload,
                "timing": {
                    "total_ms": generation.total_ms,
                    "first_token_ms": generation.first_token_ms,
                    "tokens_per_s": generation.tokens_per_s,
                },
            }
        )
    scorecard = aggregate_scorecard(model_id, cases)
    task_scores = {
        record["id"]: record["dimensions"]["task_correctness"]["score"]
        for record in records
    }
    return {
        "model": model_id,
        "dataset": str(config.quality_cases),
        "n": len(records),
        "aggregate_dimensions": {
            name: asdict(score) for name, score in scorecard.dimensions.items()
        },
        "aggregate_timing": asdict(scorecard.timing) if scorecard.timing else None,
        "scenario_slices": aggregate_scenario_slices(samples, task_scores),
        "records": records,
    }


def measure_workloads(
    base_url: str,
    process: Any,
    workloads: tuple[str, ...],
    *,
    warmup_requests: int,
    repetitions: int,
    file_size_bytes: int = 0,
) -> list[WorkloadSample]:
    prompts = {
        "short": "预约明天下午三点按摩。",
        "medium": "我想预约明天下午三点做六十分钟肩颈按摩，女性技师优先。" * 4,
        "2k": "请根据以下预约信息提取字段并判断是否需要查询技师。" * 128,
        "4k": "请根据以下完整对话历史提取预约字段并判断下一步动作。" * 256,
        "8k": "请审阅以下超长预约对话并输出规范结果。" * 512,
    }
    if any(workload not in prompts for workload in workloads):
        raise ValueError("unknown workload")

    def request(workload: str, phase: str) -> WorkloadSample:
        started = perf_counter()
        response = httpx.post(
            f"{base_url}/chat/completions",
            json={
                "model": "phase05",
                "messages": [{"role": "user", "content": prompts[workload]}],
                "temperature": 0,
                "max_tokens": 128,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=120,
        )
        response.raise_for_status()
        elapsed = (perf_counter() - started) * 1000
        payload = response.json()
        timings = payload.get("timings", {})
        return WorkloadSample(
            workload,
            phase,
            elapsed if phase == "cold" else None,
            timings.get("prompt_ms"),
            timings.get("ttft_ms"),
            timings.get("predicted_ms"),
            elapsed,
            payload.get("usage", {}).get("completion_tokens"),
            None,
            file_size_bytes,
        )

    rows: list[WorkloadSample] = []
    for workload in workloads:
        rows.append(request(workload, "cold"))
        for _ in range(warmup_requests):
            request(workload, "hot")
        rows.extend(request(workload, "hot") for _ in range(repetitions))
    return rows


def run_matrix(
    config_path: Path,
    *,
    skip_complete: bool = False,
    include_8k: bool = False,
    reports_root: Path | None = None,
    server_manager: LlamaServerManager,
) -> MatrixSummary:
    config = load_phase05_config(config_path)
    if include_8k:
        config = replace(config, workloads=(*config.workloads, "8k"), include_8k=True)
    root = reports_root or config.reports_root
    registry = ModelRegistry.from_config(config.registry)
    ids = tuple(
        models_to_run(
            root, (*config.q4_model_ids, *config.f16_anchor_ids), skip_complete=skip_complete
        )
    )
    completed: list[str] = []
    failed: list[str] = []
    for model_id in ids:
        spec = registry.get(model_id)
        process = None
        try:
            manifest = read_and_verify_manifest(spec.manifest_path)
            process = server_manager.start(model_id, root / model_id / "server.log")
            server_manager.wait_ready(process, timeout_s=60)
            quality = evaluate_quality(server_manager.base_url, config, model_id)
            rows = measure_workloads(
                server_manager.base_url,
                process,
                config.workloads,
                warmup_requests=config.warmup_requests,
                repetitions=config.repetitions,
                file_size_bytes=(
                    spec.artifact_path.stat().st_size if spec.artifact_path.is_file() else 0
                ),
            )
            aggregates = {
                workload: aggregate_workload([row for row in rows if row.workload == workload])
                for workload in config.workloads
                if any(row.workload == workload for row in rows)
            }
            write_model_result(
                root,
                model_id,
                {
                    "status": "complete",
                    "quality": quality,
                    "workloads": {
                        key: {phase: value.to_dict() for phase, value in aggregate.phases.items()}
                        for key, aggregate in aggregates.items()
                    },
                    "manifest": manifest_payload(manifest),
                },
            )
            completed.append(model_id)
        except Exception as error:
            write_failure(root, model_id, error)
            failed.append(model_id)
        finally:
            if process is not None:
                server_manager.stop(process)
    write_matrix_summary(root, {"completed_model_ids": completed, "failed_model_ids": failed})
    return MatrixSummary(tuple(completed), tuple(failed))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run Phase 05 local evaluation sequentially.")
    parser.add_argument("--config", type=Path, default=Path("configs/evaluation/phase05.yaml"))
    parser.add_argument("--skip-complete", action="store_true")
    parser.add_argument("--include-8k", action="store_true")
    parser.add_argument("--reports-root", type=Path)
    args = parser.parse_args(argv)
    config = load_phase05_config(args.config)
    quantization = yaml.safe_load(config.registry.read_text(encoding="utf-8"))
    manager = LlamaServerManager(
        ModelRegistry.from_config(config.registry), Path(quantization["toolchain"]["server"])
    )
    root = args.reports_root or config.reports_root
    started = datetime.now(UTC).isoformat()
    summary = run_matrix(
        args.config,
        skip_complete=args.skip_complete,
        include_8k=args.include_8k,
        reports_root=root,
        server_manager=manager,
    )
    render_phase05_reports(root)
    write_local_marker(
        root,
        {
            "marker": "phase05-local",
            "started_at": started,
            "finished_at": datetime.now(UTC).isoformat(),
            "host": platform.node(),
            "platform": platform.platform(),
            "python": sys.version,
            "models_attempted": [*summary.completed_model_ids, *summary.failed_model_ids],
            "models_completed": list(summary.completed_model_ids),
            "models_failed": list(summary.failed_model_ids),
            "real_run": True,
        },
    )
    return 1 if summary.failed_model_ids else 0


if __name__ == "__main__":
    raise SystemExit(main())
