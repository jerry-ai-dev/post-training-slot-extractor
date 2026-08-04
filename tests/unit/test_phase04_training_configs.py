import json
from pathlib import Path

import yaml

CONFIG_ROOT = Path("configs/training/llamafactory")
RUNS = {
    "qwen3-0.6b-sft": ("sft", None),
    "qwen3-1.7b-sft": ("sft", None),
    "qwen3-0.6b-dpo-b01": ("dpo", 0.1),
    "qwen3-0.6b-dpo-b03": ("dpo", 0.3),
    "qwen3-1.7b-dpo-b01": ("dpo", 0.1),
    "qwen3-1.7b-dpo-b03": ("dpo", 0.3),
}
SHARED_KEYS = {
    "stage",
    "finetuning_type",
    "lora_rank",
    "learning_rate",
    "num_train_epochs",
    "template",
    "enable_thinking",
}


def load_yaml(path: str | Path) -> dict[str, object]:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def test_phase04_matrix_has_exactly_six_overrides() -> None:
    actual = {
        path.stem
        for stage in ("sft", "dpo")
        for path in (CONFIG_ROOT / stage).glob("qwen3-*.yaml")
    }
    assert actual == set(RUNS)


def test_base_contracts() -> None:
    sft = load_yaml(CONFIG_ROOT / "_base_sft.yaml")
    assert (sft["stage"], sft["finetuning_type"], sft["lora_rank"]) == (
        "sft",
        "lora",
        16,
    )
    assert (sft["learning_rate"], sft["num_train_epochs"]) == (1e-4, 3.0)
    assert sft["train_on_prompt"] is False and sft["mask_history"] is True
    assert sft["template"] == "qwen3" and sft["enable_thinking"] is False
    assert sft["dataset"] == "phase03_sft_v0_1"
    assert sft["eval_dataset"] == "phase03_sft_val_v0_1"

    dpo = load_yaml(CONFIG_ROOT / "_base_dpo.yaml")
    assert dpo["stage"] == "dpo" and dpo["pref_loss"] == "sigmoid"
    assert dpo["dataset"] == "phase03_dpo_v0_1"
    assert dpo["eval_dataset"] == "phase03_dpo_val_v0_1"


def test_run_overrides_only_contain_run_specific_values() -> None:
    for run_id, (stage, beta) in RUNS.items():
        config = load_yaml(CONFIG_ROOT / stage / f"{run_id}.yaml")
        assert config["run_id"] == run_id
        assert not SHARED_KEYS.intersection(config)
        assert config["output_dir"] == f"models/adapters/{run_id}"
        if stage == "dpo":
            model_size = run_id.split("-")[1]
            assert config["adapter_name_or_path"] == f"models/adapters/qwen3-{model_size}-sft"
            assert config["pref_beta"] == beta
        else:
            assert "pref_beta" not in config


def test_registered_dataset_paths_resolve_to_existing_files() -> None:
    dataset_dir = Path("data/processed/v0.1")
    info = json.loads((dataset_dir / "dataset_info.json").read_text(encoding="utf-8"))
    assert "phase03_sft_val_v0_1" in info
    for dataset_id in (
        "phase03_sft_v0_1",
        "phase03_sft_val_v0_1",
        "phase03_dpo_v0_1",
        "phase03_dpo_val_v0_1",
    ):
        assert (dataset_dir / info[dataset_id]["file_name"]).resolve().is_file()
