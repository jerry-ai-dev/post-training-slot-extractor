from dataclasses import replace
from pathlib import Path

import pytest

from slot_extractor.quantization.pipeline import (
    PipelineError,
    PipelinePaths,
    QuantizationPipeline,
)
from slot_extractor.quantization.registry import ModelRegistry
from slot_extractor.quantization.runner import Toolchain, ToolRunnerError


class FakeRunner:
    def __init__(self, fail_model: str | None = None):
        self.commands: list[list[str]] = []
        self.fail_model = fail_model

    def run(self, argv, cwd, log_path):
        command = [str(part) for part in argv]
        self.commands.append(command)
        if self.fail_model and self.fail_model in command:
            raise ToolRunnerError("injected failure")
        output = Path(command[command.index("--output") + 1])
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes((command[0] + " output").encode())

    def version(self, executable, *, cwd=None):
        return f"{executable}-v1"


def registry_in(tmp_path: Path) -> ModelRegistry:
    original = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))
    return ModelRegistry(
        tuple(
            replace(
                model,
                artifact_path=tmp_path / model.artifact_path,
                manifest_path=tmp_path / model.manifest_path,
            )
            for model in original.models
        )
    )


def pipeline_in(tmp_path: Path, runner: FakeRunner) -> QuantizationPipeline:
    calibration = tmp_path / "data/calibration/phase05.txt"
    calibration.parent.mkdir(parents=True)
    calibration.write_text("预约王芳", encoding="utf-8")
    tools = Toolchain(
        *(
            Path(name)
            for name in ("resolve", "merge", "convert-f16", "build-imatrix", "quantize", "verify")
        )
    )
    return QuantizationPipeline(
        registry_in(tmp_path), tools, runner, PipelinePaths(tmp_path / "work", calibration)
    )


def test_pipeline_runs_exact_stage_order(tmp_path: Path):
    runner = FakeRunner()
    result = pipeline_in(tmp_path, runner).run("qwen3-0.6b-base-q4-k-m")

    assert [Path(command[0]).name for command in runner.commands] == [
        "resolve",
        "merge",
        "convert-f16",
        "build-imatrix",
        "quantize",
        "verify",
    ]
    assert result.status == "complete"


def test_pipeline_reuses_only_verified_cache(tmp_path: Path):
    runner = FakeRunner()
    pipeline = pipeline_in(tmp_path, runner)
    pipeline.run("qwen3-0.6b-base-q4-k-m")
    before = len(runner.commands)

    pipeline.run("qwen3-0.6b-base-q4-k-m")

    assert len(runner.commands) == before


def test_failed_target_does_not_stop_matrix(tmp_path: Path):
    failed_id = "qwen3-0.6b-base-q4-k-m"
    result = pipeline_in(tmp_path, FakeRunner(failed_id)).run_matrix()

    assert result.failed == (failed_id,)
    assert len(result.completed) == 7


def test_frozen_evaluation_data_is_rejected_before_tools_run(tmp_path: Path):
    runner = FakeRunner()
    frozen = Path("data/eval/test.jsonl")
    pipeline = QuantizationPipeline(
        registry_in(tmp_path),
        Toolchain(*(Path(str(index)) for index in range(6))),
        runner,
        PipelinePaths(tmp_path / "work", frozen),
    )

    with pytest.raises(PipelineError, match="calibration data must not use frozen"):
        pipeline.run("qwen3-0.6b-base-q4-k-m")
    assert runner.commands == []


def test_run_anchors_keeps_only_two_sft_f16_outputs(tmp_path: Path):
    runner = FakeRunner()

    manifests = pipeline_in(tmp_path, runner).run_anchors()

    assert [manifest.model_id for manifest in manifests] == [
        "qwen3-0.6b-sft-f16",
        "qwen3-1.7b-sft-f16",
    ]
    assert all(
        manifest.status == "complete"
        and manifest.artifact_kind == "f16"
        and manifest.is_anchor
        for manifest in manifests
    )
    stage_names = [Path(command[0]).name for command in runner.commands]
    assert "quantize" not in stage_names
    assert "build-imatrix" not in stage_names
