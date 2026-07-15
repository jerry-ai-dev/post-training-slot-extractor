# tests/integration/test_pipeline_mock.py
from pathlib import Path

from slot_extractor.evaluation.runner import run_evaluation
from slot_extractor.inference.factory import build_backend_from_config
from slot_extractor.schemas.sample import load_samples


def test_pipeline_runs_mock_backend_end_to_end() -> None:
    samples = load_samples(Path("tests/fixtures/phase01_eval.jsonl"))
    backend = build_backend_from_config(Path("configs/inference/mock.yaml"))

    scorecard = run_evaluation(samples, backend)

    assert scorecard.model == "mock-phase02-v08"
    assert scorecard.n == 3
    assert scorecard.dimensions["protocol"].score == 1.0
    # 速度不再是打分维度：改为原始时延统计，随分数卡一并产出。
    assert "speed" not in scorecard.dimensions
    assert scorecard.timing is not None
    assert scorecard.timing.count == 3
    assert scorecard.timing.total_ms_mean is not None
    assert scorecard.dimensions["task_correctness"].score == 1.0
    # resource 是阶段二唯一保留的占位维度
    assert scorecard.dimensions["resource"].score is None
