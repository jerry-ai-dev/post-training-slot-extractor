# tests/unit/test_sample_loader.py
from pathlib import Path

import pytest

from slot_extractor.schemas.sample import Sample, load_samples


def test_load_samples_parses_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"case-1","layer":"final","input":{"user_input":"约明天两点","current_time":"2026-06-08 10:00"},"expected":{"action":"final"},"assertions":["no_field_outside_schema"],"gold_facts":{"technicians_in_db":["王芳"]},"tags":["smoke"]}\n',
        encoding="utf-8",
    )

    samples = load_samples(path)

    assert samples == [
        Sample(
            id="case-1",
            layer="final",
            input={"user_input": "约明天两点", "current_time": "2026-06-08 10:00"},
            expected={"action": "final"},
            assertions=["no_field_outside_schema"],
            gold_facts={"technicians_in_db": ["王芳"]},
            tags=["smoke"],
        )
    ]


def test_load_samples_rejects_missing_required_field(tmp_path: Path) -> None:
    path = tmp_path / "cases.jsonl"
    path.write_text(
        '{"id":"case-1","layer":"final","input":{},"assertions":[],"tags":[]}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="case-1.*expected"):
        load_samples(path)
