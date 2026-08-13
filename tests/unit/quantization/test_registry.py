from pathlib import Path

import pytest

from slot_extractor.quantization.registry import (
    ModelRegistry,
    ModelSpec,
    RegistryError,
)


def test_registry_has_exactly_eight_q4_targets_and_two_sft_anchors():
    registry = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))

    assert len(registry.quantization_targets()) == 8
    assert {model.artifact_kind for model in registry.quantization_targets()} == {"q4_k_m"}
    assert {model.model_id for model in registry.anchors()} == {
        "qwen3-0.6b-sft-f16",
        "qwen3-1.7b-sft-f16",
    }


def test_registry_rejects_duplicate_model_ids():
    model = ModelSpec(
        "x",
        "qwen3",
        0.6,
        "base",
        "",
        "Qwen/Qwen3-0.6B",
        "main",
        None,
        None,
        "f16",
        Path("x.gguf"),
        Path("x.manifest.json"),
        False,
    )

    with pytest.raises(RegistryError, match="duplicate model_id"):
        ModelRegistry((model, model))


def test_registry_rejects_unknown_model_id():
    registry = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))

    with pytest.raises(RegistryError, match="unknown model_id: missing"):
        registry.get("missing")


def test_dpo_targets_retain_parent_and_adapter_lineage():
    registry = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))

    dpo = registry.get("qwen3-0.6b-dpo-b01-q4-k-m")
    assert dpo.parent_model_id == "qwen3-0.6b-sft-q4-k-m"
    assert dpo.adapter_run_id == "qwen3-0.6b-dpo-b01"
