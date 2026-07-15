# tests/unit/test_output_schema.py
import pytest

from slot_extractor.schemas.output import (
    OutputValidationError,
    parse_model_json,
    validate_final_output,
    validate_tool_call_output,
)


def test_parse_model_json_rejects_markdown_fence() -> None:
    with pytest.raises(OutputValidationError, match="must be raw JSON"):
        parse_model_json('```json\n{"action":"final"}\n```')


def test_validate_final_output_accepts_exact_schema() -> None:
    data = parse_model_json(
        '{"action":"final","gender":"female","start_time":"2026-06-09 14:00","duration":"60分钟","project":"精油按摩","preference":"无","technician_name":"王芳","confirmation":false,"info_complete":true,"unrelated":false,"missing_info":[]}'
    )

    validate_final_output(data)


def test_validate_final_output_rejects_extra_field() -> None:
    data = parse_model_json(
        '{"action":"final","gender":"female","start_time":"2026-06-09 14:00","duration":"60分钟","project":"精油按摩","preference":"无","technician_name":"王芳","confirmation":false,"info_complete":true,"unrelated":false,"missing_info":[],"note":"hello"}'
    )

    with pytest.raises(OutputValidationError, match="schema fields"):
        validate_final_output(data)


def test_validate_tool_call_output_accepts_known_shape() -> None:
    data = parse_model_json(
        '{"action":"tool_call","tool_name":"find_technicians","arguments":{"technician_name":"小王","start_time":"2026-06-09 14:00","duration":"未知","gender":"未知","preference":"无"}}'
    )

    validate_tool_call_output(data)
