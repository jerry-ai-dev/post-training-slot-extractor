from pathlib import Path

import pytest

from scripts.quantize.build_phase06_round006_matrix import BuildError, load_config, require_adapter


def test_phase06_round006_matrix_is_frozen() -> None:
    config = load_config(Path("configs/quantization/phase06-round006.yaml"))
    assert config["model"]["run_id"] == "r004-qwen3-0.6b-sft"
    assert config["quantizations"] == [
        "F16",
        "Q8_0",
        "Q6_K",
        "Q5_K_M",
        "Q4_K_M",
        "Q3_K_M",
        "IQ2_M",
        "IQ2_XS",
    ]


def test_external_adapter_requires_config_and_weights(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    with pytest.raises(BuildError, match="adapter_model.safetensors"):
        require_adapter(adapter)


def test_external_adapter_is_accepted(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "adapter_config.json").write_text("{}", encoding="utf-8")
    (adapter / "adapter_model.safetensors").write_bytes(b"placeholder")
    require_adapter(adapter)
