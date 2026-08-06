from pathlib import Path


def test_training_versions_are_exactly_pinned() -> None:
    lines = Path("requirements-train.txt").read_text(encoding="utf-8").splitlines()
    expected = {
        "llamafactory==0.9.5",
        "transformers==5.6.0",
        "peft==0.18.1",
        "trl==0.24.0",
        "torch==2.6.0",
        "torchvision==0.21.0",
        "torchaudio==2.6.0",
    }
    assert set(lines) == expected
    assert Path("configs/training/llamafactory/VERSION").read_text() == "v0.9.5\n"
