from slot_extractor.data.isolation import input_fingerprint
from slot_extractor.data.phase06_round3_sft import (
    generate_large_round3_specialty,
    generate_shared_round3_samples,
    generate_small_round3_specialty,
)
from slot_extractor.data.raw_validator import validate_raw_sample


def test_round3_targeted_samples_are_valid_and_unique() -> None:
    samples = [
        *generate_shared_round3_samples(),
        *generate_small_round3_specialty(),
        *generate_large_round3_specialty(),
    ]
    assert len(samples) == 240
    assert len({sample.id for sample in samples}) == len(samples)
    assert len({input_fingerprint(sample) for sample in samples}) == len(samples)
    for sample in samples:
        validate_raw_sample(sample)


def test_round3_specialties_match_observed_regressions() -> None:
    small = generate_small_round3_specialty()
    large = generate_large_round3_specialty()
    assert all(sample.output_kind == "tool_call" for sample in small)
    assert all(sample.output_kind == "final" for sample in large)
    assert any("无关请求" in sample.tags for sample in large)
    assert any("严格JSON" in sample.tags for sample in large)
    assert any("参数完整性" in sample.tags for sample in small)
