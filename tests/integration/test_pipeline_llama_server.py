# tests/integration/test_pipeline_llama_server.py
import pytest

from slot_extractor.evaluation.runner import run_evaluation
from slot_extractor.inference.factory import build_backend_from_config
from slot_extractor.schemas.sample import load_samples


@pytest.mark.local_backend
def test_pipeline_llama_server_end_to_end() -> None:
    samples = load_samples("tests/fixtures/phase01_eval.jsonl")
    backend = build_backend_from_config("configs/inference/llama_server.yaml")

    scorecard = run_evaluation(samples[:1], backend)

    assert scorecard.n == 1
    # 速度改为原始时延统计，随分数卡产出（不再是 0/1 打分维度）。
    assert scorecard.timing is not None
    assert scorecard.timing.total_ms_mean is not None
