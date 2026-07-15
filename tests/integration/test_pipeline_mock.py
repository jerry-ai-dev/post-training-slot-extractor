# tests/integration/test_pipeline_mock.py
from pathlib import Path

from slot_extractor.evaluation.runner import run_evaluation
from slot_extractor.inference.factory import build_backend_from_config
from slot_extractor.schemas.sample import load_samples


def test_pipeline_runs_mock_backend_end_to_end() -> None:
    samples = load_samples(Path("tests/fixtures/phase01_eval.jsonl"))
    backend = build_backend_from_config(Path("configs/inference/mock.yaml"))

    scorecard = run_evaluation(samples, backend)

    assert scorecard.model == "mock-phase01"
    assert scorecard.n == 3
    assert scorecard.dimensions["instruction"].score == 1.0
    assert scorecard.dimensions["speed"].score == 1.0
    assert scorecard.dimensions["hallucination"].score is None
