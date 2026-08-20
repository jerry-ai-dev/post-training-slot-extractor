from __future__ import annotations

from slot_extractor.data.isolation import input_fingerprint
from slot_extractor.data.phase06_sft import generate_targeted_samples
from slot_extractor.data.raw_validator import validate_raw_sample


def test_targeted_phase06_samples_are_valid_and_unique() -> None:
    samples = generate_targeted_samples()

    assert len(samples) == 316
    assert len({sample.id for sample in samples}) == len(samples)
    assert len({input_fingerprint(sample) for sample in samples}) == len(samples)
    for sample in samples:
        validate_raw_sample(sample)


def test_targeted_phase06_samples_cover_round_one_failures() -> None:
    samples = generate_targeted_samples()
    tags = {tag for sample in samples for tag in sample.tags}

    assert {"日期标准化", "星期换算", "最小替换", "动作边界", "预约成功"} <= tags
    assert sum(sample.output_kind == "tool_call" for sample in samples) >= 170
    assert any(
        sample.expected.get("reply_type") == "booking_authorized"
        and "预约已成功" in sample.expected.get("reply", "")
        for sample in samples
    )


def test_date_labels_cover_month_and_year_boundaries() -> None:
    samples = generate_targeted_samples()
    dates = {
        sample.expected["arguments"]["start_time"]
        for sample in samples
        if "日期标准化" in sample.tags
    }

    assert any(value.startswith("2027-") for value in dates)
    assert "2026-02-01 20:00" in dates
