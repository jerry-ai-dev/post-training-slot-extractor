# tests/unit/test_scorers.py
from slot_extractor.evaluation.scorers.instruction import InstructionScorer
from slot_extractor.evaluation.scorers.not_available import NotAvailableScorer
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.schemas.sample import Sample


def sample(layer: str, expected: dict) -> Sample:
    return Sample(
        id="case-1",
        layer=layer,
        input={},
        expected=expected,
        assertions=[],
        tags=[],
    )


def result(text: str) -> GenerationResult:
    return GenerationResult(
        text=text,
        model="mock",
        prefill_ms=5,
        first_token_ms=10,
        total_ms=50,
        output_tokens=25,
        tokens_per_s=500,
    )


def test_instruction_scorer_accepts_valid_final_json() -> None:
    score = InstructionScorer().score(
        sample("final", {"action": "final"}),
        result(
            '{"action":"final","gender":"female","start_time":"2026-06-09 14:00","duration_minutes":60,"preferences":["精油"],"technician_name":"王芳","technician_status":"available","confirmation":false,"info_complete":true,"unrelated":false,"missing_info":[],"reply_type":"confirm_available","reply":"王芳技师明天下午2点有空，可以安排60分钟精油按摩，您确认吗？"}'
        ),
    )

    assert score.score == 1.0
    assert score.passed is True


def test_not_available_scorer_returns_none_score() -> None:
    score = NotAvailableScorer("hallucination").score(
        sample("final", {"action": "final"}),
        result('{"action":"final"}'),
    )

    assert score.score is None
    assert score.passed is None
