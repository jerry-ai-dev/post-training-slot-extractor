from slot_extractor.data.isolation import input_fingerprint
from slot_extractor.data.phase06_round4_sft import (
    generate_large_round4_specialty,
    generate_round4_holdout,
    generate_small_round4_specialty,
)
from slot_extractor.data.raw_validator import validate_raw_sample
from slot_extractor.schemas.sample import load_samples


def test_round4_samples_are_valid_and_unique() -> None:
    training = [*generate_small_round4_specialty(), *generate_large_round4_specialty()]
    holdout = generate_round4_holdout()
    assert len(training) == 160
    assert len(holdout) == 24
    assert not (
        {input_fingerprint(item) for item in training}
        & {input_fingerprint(item) for item in holdout}
    )
    for sample in [*training, *holdout]:
        validate_raw_sample(sample)


def test_round4_model_specialties_are_distinct() -> None:
    small = generate_small_round4_specialty()
    large = generate_large_round4_specialty()
    assert any(sample.output_kind == "tool_call" for sample in small)
    assert all(sample.output_kind == "final" for sample in large)
    assert any("确认动作" in sample.tags for sample in large)


def test_round4_built_holdout_uses_evaluation_schema(tmp_path) -> None:
    from slot_extractor.data.phase06_round4_sft import _holdout_eval_record
    from slot_extractor.utils.jsonl import write_jsonl

    path = tmp_path / "holdout.jsonl"
    write_jsonl(path, [_holdout_eval_record(sample) for sample in generate_round4_holdout()])
    loaded = load_samples(path)
    assert len(loaded) == 24
