# tests/integration/test_pipeline_phase02.py
from pathlib import Path

from slot_extractor.evaluation.runner import run_evaluation
from slot_extractor.inference.factory import build_backend_from_config
from slot_extractor.schemas.sample import load_samples


def test_phase02_pipeline_simplified_dimensions() -> None:
    all_samples = load_samples(Path("data/eval/test.jsonl"))
    ids = {"ask-0001", "unrelated-0001", "specific-available-01-call"}
    samples = [s for s in all_samples if s.id in ids]
    backend = build_backend_from_config(Path("configs/inference/mock.yaml"))

    card = run_evaluation(samples, backend)

    assert card.dimensions["protocol"].score is not None
    assert card.dimensions["task_correctness"].score is not None
    assert card.dimensions["resource"].score is None
