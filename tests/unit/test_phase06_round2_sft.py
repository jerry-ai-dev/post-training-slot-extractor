from slot_extractor.data.isolation import input_fingerprint
from slot_extractor.data.phase06_round2_sft import (
    generate_large_model_specialty,
    generate_shared_samples,
    generate_small_model_specialty,
)
from slot_extractor.data.raw_validator import validate_raw_sample


def test_round2_targeted_samples_are_valid_and_unique() -> None:
    samples = [
        *generate_shared_samples(),
        *generate_small_model_specialty(),
        *generate_large_model_specialty(),
    ]
    assert len(samples) == 370
    assert len({sample.id for sample in samples}) == len(samples)
    assert len({input_fingerprint(sample) for sample in samples}) == len(samples)
    for sample in samples:
        validate_raw_sample(sample)


def test_round2_model_specialties_are_distinct() -> None:
    small = generate_small_model_specialty()
    large = generate_large_model_specialty()
    assert all(sample.output_kind == "tool_call" for sample in small)
    assert all(sample.output_kind == "final" for sample in large)
    assert any("matched映射" in sample.tags for sample in large)
    assert any("严格短JSON" in sample.tags for sample in small)
