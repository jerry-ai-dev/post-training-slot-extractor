from pathlib import Path

from fastapi.testclient import TestClient

from slot_extractor.quantization.registry import ModelRegistry
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.tool_loop.app import create_app

CONFIG = Path("configs/quantization/phase06-round009-app.yaml")


class Backend:
    def __init__(self, model: str) -> None:
        self.model = model

    def generate(self, messages, params=None):
        return GenerationResult("{}", self.model, 0, 0, 1)


def test_round009_registry_contains_baseline_and_final_profiles() -> None:
    registry = ModelRegistry.from_config(CONFIG)
    assert [(model.model_id, model.artifact_kind) for model in registry.models] == [
        ("qwen3-0.6b-base-f16", "f16"),
        ("r004-qwen3-0.6b-sft-q8-quality", "q8_0"),
        ("r004-qwen3-0.6b-sft-q4-fast", "q4_k_m"),
    ]
    assert all("--cache-prompt" in model.server_args for model in registry.models)


def test_round009_app_exposes_baseline_and_integrated_models(tmp_path: Path) -> None:
    registry = ModelRegistry.from_config(CONFIG)
    app = create_app(
        registry=registry,
        backend_factory=lambda spec: Backend(spec.model_id),
        log_path=tmp_path / "app.jsonl",
        quantization_config=CONFIG,
    )
    with TestClient(app) as client:
        models = client.get("/api/models").json()
    assert [model["model_id"] for model in models] == [
        "qwen3-0.6b-base-f16",
        "r004-qwen3-0.6b-sft-q8-quality",
        "r004-qwen3-0.6b-sft-q4-fast",
    ]
