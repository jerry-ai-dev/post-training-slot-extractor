"""Ordered, failure-isolated Phase 05 quantization pipeline."""

import hashlib
import subprocess
from dataclasses import dataclass, replace
from pathlib import Path

from .lineage import Lineage, cache_key
from .manifest import (
    ArtifactHash,
    ManifestError,
    StageManifest,
    read_and_verify_manifest,
    sha256_file,
    write_manifest_atomic,
)
from .registry import ModelRegistry, ModelSpec
from .runner import CommandRunner, Toolchain, ToolRunnerError

STAGES = ("resolve", "merge", "convert-f16", "build-imatrix", "quantize", "verify")


class PipelineError(RuntimeError):
    """Raised when a model cannot complete the quantization pipeline."""


@dataclass(frozen=True)
class PipelinePaths:
    work_root: Path
    calibration_data: Path


@dataclass(frozen=True)
class MatrixResult:
    completed: tuple[str, ...]
    failed: tuple[str, ...]
    manifests: tuple[StageManifest, ...]


class QuantizationPipeline:
    def __init__(
        self,
        registry: ModelRegistry,
        toolchain: Toolchain,
        runner: CommandRunner,
        paths: PipelinePaths,
    ) -> None:
        self.registry = registry
        self.toolchain = toolchain
        self.runner = runner
        self.paths = paths

    def run(self, model_id: str, *, force: bool = False) -> StageManifest:
        self._validate_calibration()
        spec = self.registry.get(model_id)
        if spec.is_anchor:
            raise PipelineError("anchors must be run with run_anchors")
        lineage = self._lineage(spec)
        expected_key = cache_key(lineage, "pipeline", {"quantization": "Q4_K_M"})
        if not force and spec.manifest_path.is_file():
            cached = read_and_verify_manifest(spec.manifest_path)
            if cached.status == "complete" and cached.cache_key == expected_key:
                return cached
        manifest = StageManifest(
            spec.model_id,
            "resolve",
            "running",
            spec.artifact_kind,
            False,
            expected_key,
            lineage,
            (),
            (),
            (),
            None,
        )
        current_input: Path | None = None
        commands: list[str] = []
        try:
            for stage in STAGES:
                stage_dir = self.paths.work_root / spec.model_id / stage
                stage_dir.mkdir(parents=True, exist_ok=True)
                output = self._stage_output(spec, stage, stage_dir)
                tool_name = {
                    "convert-f16": "convert_f16",
                    "build-imatrix": "imatrix",
                    "verify": "server",
                }.get(stage, stage)
                executable = getattr(self.toolchain, tool_name)
                argv = [
                    str(executable),
                    "--model-id",
                    spec.model_id,
                    "--output",
                    str(output),
                ]
                if current_input is not None:
                    argv.extend(("--input", str(current_input)))
                if stage == "build-imatrix":
                    argv.extend(("--calibration", str(self.paths.calibration_data)))
                if stage == "quantize":
                    argv.extend(("--type", "Q4_K_M"))
                self.runner.run(argv, stage_dir, stage_dir / "stage.log")
                if not output.is_file() or output.stat().st_size == 0:
                    raise PipelineError(f"{stage} did not produce a non-empty output")
                current_input = output
                commands.extend(argv)
                manifest = replace(manifest, stage=stage, command=tuple(commands))
            if not spec.artifact_path.is_file() or spec.artifact_path.stat().st_size == 0:
                raise PipelineError("quantized artifact is missing or empty")
            complete = replace(
                manifest,
                status="complete",
                outputs=(ArtifactHash(str(spec.artifact_path), sha256_file(spec.artifact_path)),),
            )
            write_manifest_atomic(spec.manifest_path, complete)
            return complete
        except (OSError, ToolRunnerError, ManifestError, PipelineError) as exc:
            failed = replace(manifest, status="failed", error=str(exc))
            write_manifest_atomic(spec.manifest_path, failed)
            raise PipelineError(f"{spec.model_id} failed: {exc}") from exc

    def run_matrix(self, *, continue_on_error: bool = True) -> MatrixResult:
        completed: list[str] = []
        failed: list[str] = []
        manifests: list[StageManifest] = []
        for spec in self.registry.quantization_targets():
            try:
                manifests.append(self.run(spec.model_id))
                completed.append(spec.model_id)
            except PipelineError:
                failed.append(spec.model_id)
                if not continue_on_error:
                    break
        return MatrixResult(tuple(completed), tuple(failed), tuple(manifests))

    def run_anchors(self) -> tuple[StageManifest, ...]:
        return tuple(self._run_anchor(spec) for spec in self.registry.anchors())

    def _run_anchor(self, spec: ModelSpec) -> StageManifest:
        self._validate_calibration()
        lineage = self._lineage(spec)
        expected_key = cache_key(lineage, "anchor", {"format": "F16"})
        if spec.manifest_path.is_file():
            cached = read_and_verify_manifest(spec.manifest_path)
            if cached.status == "complete" and cached.cache_key == expected_key:
                return cached
        manifest = StageManifest(
            spec.model_id,
            "resolve",
            "running",
            "f16",
            True,
            expected_key,
            lineage,
            (),
            (),
            (),
            None,
        )
        current_input: Path | None = None
        commands: list[str] = []
        try:
            for stage in ("resolve", "merge", "convert-f16", "verify"):
                stage_dir = self.paths.work_root / spec.model_id / stage
                stage_dir.mkdir(parents=True, exist_ok=True)
                if stage == "convert-f16":
                    spec.artifact_path.parent.mkdir(parents=True, exist_ok=True)
                    output = spec.artifact_path
                else:
                    output = stage_dir / ("verified.ok" if stage == "verify" else "artifact.bin")
                tool_name = {"convert-f16": "convert_f16", "verify": "server"}.get(
                    stage, stage
                )
                argv = [
                    str(getattr(self.toolchain, tool_name)),
                    "--model-id",
                    spec.model_id,
                    "--output",
                    str(output),
                ]
                if current_input is not None:
                    argv.extend(("--input", str(current_input)))
                self.runner.run(argv, stage_dir, stage_dir / "stage.log")
                current_input = output
                commands.extend(argv)
                manifest = replace(manifest, stage=stage, command=tuple(commands))
            complete = replace(
                manifest,
                status="complete",
                outputs=(ArtifactHash(str(spec.artifact_path), sha256_file(spec.artifact_path)),),
            )
            write_manifest_atomic(spec.manifest_path, complete)
            return complete
        except (OSError, ToolRunnerError, ManifestError) as exc:
            write_manifest_atomic(
                spec.manifest_path, replace(manifest, status="failed", error=str(exc))
            )
            raise PipelineError(f"{spec.model_id} failed: {exc}") from exc

    def _validate_calibration(self) -> None:
        normalized = self.paths.calibration_data.as_posix().lower()
        if normalized == "data/eval/test.jsonl" or "/data/eval/" in f"/{normalized}":
            raise PipelineError("calibration data must not use frozen evaluation data")
        if not self.paths.calibration_data.is_file():
            raise PipelineError(f"calibration data does not exist: {self.paths.calibration_data}")

    def _lineage(self, spec: ModelSpec) -> Lineage:
        calibration_hash = sha256_file(self.paths.calibration_data)
        source_id = hashlib.sha256(spec.base_model.encode("utf-8")).hexdigest()
        tools = (
            self.toolchain.resolve,
            self.toolchain.merge,
            self.toolchain.convert_f16,
            self.toolchain.imatrix,
            self.toolchain.quantize,
            self.toolchain.server,
        )
        versions = tuple(
            (str(tool), self.runner.version(tool, cwd=self.paths.work_root)) for tool in tools
        )
        git = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
            ).stdout.strip()
            or "unknown"
        )
        return Lineage(
            spec.model_id,
            spec.base_model,
            spec.base_revision,
            spec.parent_model_id,
            spec.adapter_run_id,
            (("base_model", source_id), ("calibration", calibration_hash)),
            git,
            versions,
        )

    @staticmethod
    def _stage_output(spec: ModelSpec, stage: str, stage_dir: Path) -> Path:
        if stage == "quantize":
            spec.artifact_path.parent.mkdir(parents=True, exist_ok=True)
            return spec.artifact_path
        return stage_dir / ("verified.ok" if stage == "verify" else "artifact.bin")
