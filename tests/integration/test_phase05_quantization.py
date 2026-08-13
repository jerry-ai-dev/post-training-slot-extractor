import json
from pathlib import Path

from scripts.quantize import run_phase05
from slot_extractor.quantization.pipeline import MatrixResult


class FakePipeline:
    def __init__(self, registry, toolchain, runner, paths):
        self.registry = registry
        self.paths = paths

    def run_matrix(self, *, continue_on_error=True):
        ids = tuple(model.model_id for model in self.registry.quantization_targets())
        return MatrixResult(ids, (), ())

    def run(self, model_id, *, force=False):
        return type("Manifest", (), {"model_id": model_id})()


def test_cli_runs_exact_matrix_and_writes_summary(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(run_phase05, "QuantizationPipeline", FakePipeline)
    summary = tmp_path / "summary.json"

    result = run_phase05.main(
        [
            "--config",
            "configs/quantization/phase05.yaml",
            "--continue-on-error",
            "--summary",
            str(summary),
        ]
    )

    payload = json.loads(summary.read_text(encoding="utf-8"))
    assert result == 0
    assert len(payload["completed"]) == 8
    assert payload["failed"] == []


def test_cli_returns_one_when_requested_model_fails(monkeypatch, tmp_path: Path):
    class FailingPipeline(FakePipeline):
        def run(self, model_id, *, force=False):
            raise run_phase05.PipelineError("failed")

    monkeypatch.setattr(run_phase05, "QuantizationPipeline", FailingPipeline)
    summary = tmp_path / "summary.json"

    result = run_phase05.main(
        [
            "--model-id",
            "qwen3-0.6b-base-q4-k-m",
            "--summary",
            str(summary),
        ]
    )

    assert result == 1
    assert json.loads(summary.read_text(encoding="utf-8"))["failed"] == [
        "qwen3-0.6b-base-q4-k-m"
    ]
