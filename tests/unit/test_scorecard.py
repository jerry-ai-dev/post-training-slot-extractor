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
                    "instruction": DimensionScore("instruction", 1.0, True, "ok"),
                    "speed": DimensionScore("speed", 0.5, False, "slow"),
                },
            )
        ],
    )

    text = render_scorecard(card)

    assert card.dimensions["instruction"].score == 1.0
    assert card.dimensions["hallucination"].score is None
    assert "Slot-Extractor 评估分数卡" in text
    assert "n/a" in text
