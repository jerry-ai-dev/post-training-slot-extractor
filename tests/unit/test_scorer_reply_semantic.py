import json

from slot_extractor.evaluation.scorers.reply_semantic import ReplySemanticScorer
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.schemas.sample import ReplyExpectations, Sample


def _sample() -> Sample:
    return Sample(
        id="reply-semantic",
        layer="final",
        input={},
        expected={"action": "final", "reply_type": "confirm_available"},
        assertions=[],
        tags=[],
        reply_expectations=ReplyExpectations(
            required_acts=("inform_technician_available", "request_confirmation"),
            forbidden_acts=("claim_booking_success",),
            required_fields=(),
            references=("王芳技师有空，您确认吗？",),
        ),
    )


def _result(reply: str) -> GenerationResult:
    return GenerationResult(
        text=json.dumps({"reply_type": "confirm_available", "reply": reply}, ensure_ascii=False),
        model="mock",
        prefill_ms=None,
        first_token_ms=None,
        total_ms=1,
    )


def test_reply_semantic_scorer_accepts_paraphrase() -> None:
    score = ReplySemanticScorer().score(_sample(), _result("王芳技师有空，这个安排您看可以吗？"))
    assert score.passed is True


def test_reply_semantic_scorer_rejects_premature_success() -> None:
    score = ReplySemanticScorer().score(_sample(), _result("王芳技师有空，已经预约成功。"))
    assert score.passed is False
    assert "claim_booking_success" in score.detail


def test_reply_semantic_scorer_accepts_available_provide_wording() -> None:
    score = ReplySemanticScorer().score(
        _sample(), _result("王芳技师明天下午可以提供60分钟服务，请问是否确认？")
    )
    assert score.passed is True
