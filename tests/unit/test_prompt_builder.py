# tests/unit/test_prompt_builder.py
from slot_extractor.prompts.template import PromptBuilder
from slot_extractor.schemas.sample import Sample


def test_prompt_builder_includes_rules_and_case_input() -> None:
    sample = Sample(
        id="case-1",
        layer="final",
        input={
            "history": "用户：想约按摩",
            "user_input": "明天下午两点，60分钟",
            "current_time": "2026-06-08 10:00",
        },
        expected={"action": "final"},
        assertions=[],
        gold_facts={},
        tags=[],
    )

    prompt = PromptBuilder().build(sample)

    assert "只输出一个 JSON 对象" in prompt
    assert '"current_time": "2026-06-08 10:00"' in prompt
    assert "不要输出 Markdown 代码块" in prompt
    assert "Sample ID: case-1" in prompt
