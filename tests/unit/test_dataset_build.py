from copy import deepcopy

import pytest
from test_raw_validator import _final

from slot_extractor.data.dataset_build import _split, build_dataset
from slot_extractor.data.raw_sample import RawSample, raw_sample_from_record
from slot_extractor.utils.jsonl import read_jsonl


def _samples():
    samples = []
    for index in range(5):
        record = deepcopy(_final())
        record["id"] = f"ask-{index}"
        record["input"]["user_input"] = f"明天第{index}个时段"
        record["tags"].append("相对时间")
        samples.append(raw_sample_from_record(record))
    return samples


def test_build_writes_expected_artifacts(tmp_path) -> None:
    result = build_dataset(
        _samples(), eval_records=[], output_root=tmp_path, version="v0.1", seed=42
    )
    assert result.sft_train.exists()
    assert result.sft_val.exists()
    assert result.dpo_train.exists()
    assert result.dpo_val.exists()
    assert result.dataset_info.exists()
    assert result.version_card.exists()
    assert len(list(read_jsonl(result.sft_train))) == 4
    assert len(list(read_jsonl(result.sft_val))) == 1
    assert len(list(read_jsonl(result.dpo_train))) == 4
    assert len(list(read_jsonl(result.dpo_val))) == 1


def test_dataset_registration_uses_sharegpt_contract(tmp_path) -> None:
    result = build_dataset(_samples(), [], tmp_path, "v0.1", 42)
    import json

    info = json.loads(result.dataset_info.read_text(encoding="utf-8"))
    assert info["phase03_sft_v0_1"]["formatting"] == "sharegpt"
    assert info["phase03_dpo_v0_1"]["ranking"] is True
    assert info["phase03_dpo_v0_1"]["tags"]["function_tag"] == "function_call"
    assert info["phase03_dpo_v0_1"]["file_name"] == "dpo/v0.1/train.jsonl"
    assert info["phase03_dpo_val_v0_1"]["file_name"] == "dpo/v0.1/val.jsonl"


def test_split_uses_ten_percent_validation_for_larger_categories(tmp_path) -> None:
    samples = []
    for index in range(20):
        record = deepcopy(_final())
        record["id"] = f"large-ask-{index}"
        record["input"]["user_input"] = f"明天第{index}个大样本时段"
        record["tags"].append("相对时间")
        samples.append(raw_sample_from_record(record))
    result = build_dataset(samples, [], tmp_path, "v0.1", 42)
    assert result.train_count == 18
    assert result.val_count == 2
    assert result.dpo_train_count == 18
    assert result.dpo_val_count == 2


def test_strict_build_rejects_missing_semantic_coverage(tmp_path) -> None:
    with pytest.raises(ValueError, match="semantic coverage"):
        build_dataset(_samples(), [], tmp_path, "v0.1", 42, strict_audit=True)


def test_build_samples_exact_dpo_quota_and_splits_it_nine_to_one(tmp_path) -> None:
    samples = []
    for index in range(20):
        record = deepcopy(_final())
        record["id"] = f"quota-{index}"
        record["input"]["user_input"] = f"明天第{index}个配额样本"
        record["tags"].append("相对时间")
        samples.append(raw_sample_from_record(record))
    result = build_dataset(
        samples,
        [],
        tmp_path,
        "v0.1",
        42,
        dpo_target_counts={"P7": 10},
    )
    assert result.dpo_train_count == 9
    assert result.dpo_val_count == 1


def test_split_keeps_exact_global_ten_percent_for_500_samples() -> None:
    counts = {"追问": 107, "工具调用": 107, "最终 JSON": 107, "确认": 89, "无关": 90}
    samples = [
        RawSample(
            id=f"{category}-{index}",
            output_kind="final",
            conversation_kind="single_turn",
            tags=(category,),
            input={},
            expected={},
            dpo_targets=(),
        )
        for category, count in counts.items()
        for index in range(count)
    ]
    train, validation = _split(samples, 42)
    assert len(train) == 450
    assert len(validation) == 50
    assert {sample.category for sample in validation} == set(counts)
