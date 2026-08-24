from slot_extractor.evaluation.scorers.reply_faithfulness import (
    ReplyFaithfulnessScorer,
    _mentioned_technicians,
)
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.schemas.sample import ReplyExpectations, Sample


def test_generic_gender_phrase_is_not_treated_as_technician_name() -> None:
    assert _mentioned_technicians("暂时没有符合条件的女技师。") == set()
    assert _mentioned_technicians("没有符合条件的泰式女技师。") == set()
    assert (
        _mentioned_technicians(
            "明天晚上8点没有符合条件的泰式女技师，您愿意调整时间、按摩类型或技师性别吗？"
        )
        == set()
    )


def test_booking_success_is_allowed_when_not_forbidden() -> None:
    sample = Sample(
        id="confirmed",
        output_kind="final",
        conversation_kind="multi_turn",
        tags=[],
        assertions=[],
        input={"current_time": "2026-06-08 10:00"},
        expected={
            "action": "final",
            "technician_name": "王芳",
            "technician_status": "available",
            "reply_type": "booking_authorized",
        },
        reply_expectations=ReplyExpectations(
            required_acts=("claim_booking_success",),
            forbidden_acts=(),
            required_fields=(),
            references=("好的，预约成功。",),
        ),
    )
    result = GenerationResult(
        text='{"action":"final","reply":"好的，王芳技师的预约已成功。"}',
        model="test",
        prefill_ms=None,
        first_token_ms=None,
        total_ms=1,
    )

    score = ReplyFaithfulnessScorer().score(sample, result)

    assert score.passed is True
