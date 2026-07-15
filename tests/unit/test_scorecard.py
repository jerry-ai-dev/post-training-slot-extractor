# tests/unit/test_scorecard.py
from slot_extractor.evaluation.scorecard import aggregate_scorecard, render_scorecard
from slot_extractor.schemas.results import CaseResult, DimensionScore


def test_scorecard_aggregates_dimensions_and_renders_na() -> None:
    card = aggregate_scorecard(
        "mock",
        [
            CaseResult(
                sample_id="case-1",
                layer="final",
                model_output="{}",
                dimensions={
                    "protocol": DimensionScore("protocol", 1.0, True, "ok"),
                    "task_correctness": DimensionScore(
                        "task_correctness", 0.8, False, "partial"
                    ),
                },
            )
        ],
    )

    text = render_scorecard(card)

    assert card.dimensions["protocol"].score == 1.0
    assert card.dimensions["task_correctness"].score == 0.8
    assert card.dimensions["resource"].score is None
    assert "Appointment-Agent 评估分数卡" in text
    assert "任务正确性" in text
    assert "n/a" in text
