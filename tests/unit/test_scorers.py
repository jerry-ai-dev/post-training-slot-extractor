# tests/unit/test_scorers.py
from slot_extractor.evaluation.scorers.field_extraction import FieldExtractionScorer
from slot_extractor.evaluation.scorers.instruction import InstructionScorer
from slot_extractor.evaluation.scorers.not_available import NotAvailableScorer
from slot_extractor.evaluation.scorers.speed import SpeedScorer
from slot_extractor.evaluation.scorers.tool_call import ToolCallScorer
from slot_extractor.schemas.results import GenerationResult
from slot_extractor.schemas.sample import Sample


def sample(layer: str, expected: dict) -> Sample:
    return Sample(
        id="case-1",
        layer=layer,
        input={},
        expected=expected,
        assertions=[],
        gold_facts={},
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
            '{"action":"final","gender":"female","start_time":"2026-06-09 14:00","duration":"60分钟","project":"精油按摩","preference":"无","technician_name":"王芳","confirmation":false,"info_complete":true,"unrelated":false,"missing_info":[]}'
        ),
    )

    assert score.score == 1.0
    assert score.passed is True


def test_field_extraction_scorer_computes_ratio() -> None:
    score = FieldExtractionScorer().score(
        sample("final", {"action": "final", "gender": "female", "duration": "90分钟"}),
        result(
            '{"action":"final","gender":"female","start_time":"未知","duration":"60分钟","project":"未知","preference":"无","technician_name":"未知","confirmation":false,"info_complete":false,"unrelated":false,"missing_info":[]}'
        ),
    )

    assert score.score == 0.5
    assert score.passed is False


def test_tool_call_scorer_checks_tool_and_arguments() -> None:
    score = ToolCallScorer().score(
        sample(
            "tool_call",
            {
                "action": "tool_call",
                "tool_name": "find_technicians",
                "arguments": {"technician_name": "小王"},
            },
        ),
        result(
            '{"action":"tool_call","tool_name":"find_technicians","arguments":{"technician_name":"小王"}}'
        ),
    )

    assert score.score == 1.0
    assert score.passed is True


def test_speed_scorer_reads_generation_timing() -> None:
    score = SpeedScorer(first_token_good_ms=800, total_good_ms=1500).score(
        sample("final", {"action": "final"}),
        result('{"action":"final"}'),
    )

    assert score.score == 1.0
    assert "first_token_ms=10" in score.detail


def test_not_available_scorer_returns_none_score() -> None:
    score = NotAvailableScorer("hallucination").score(
        sample("final", {"action": "final"}),
        result('{"action":"final"}'),
    )

    assert score.score is None
    assert score.passed is None
