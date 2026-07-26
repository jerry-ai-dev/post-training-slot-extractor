import json
from dataclasses import dataclass

import pytest
from test_raw_validator import _final

from slot_extractor.data.generator import (
    GenerationError,
    GenerationRequest,
    RawGenerator,
    build_generation_messages,
)


@dataclass
class Result:
    text: str


class StubBackend:
    model = "stub"

    def __init__(self, text: str):
        self.text = text
        self.calls = []
        self.params = []

    def generate(self, messages, params=None):
        self.calls.append(messages)
        self.params.append(params)
        return Result(self.text)


class SequenceBackend(StubBackend):
    def __init__(self, texts: list[str]):
        super().__init__("")
        self.texts = iter(texts)

    def generate(self, messages, params=None):
        self.calls.append(messages)
        self.params.append(params)
        return Result(next(self.texts))


def test_generate_one_validates_backend_json() -> None:
    record = _final()
    backend = StubBackend(json.dumps(record, ensure_ascii=False))
    sample = RawGenerator(backend).generate_one(GenerationRequest("追问", 1))
    assert sample.id == "x"
    assert backend.calls[0][-1]["role"] == "user"


def test_generate_one_rejects_markdown_fence() -> None:
    with pytest.raises(GenerationError, match="raw JSON"):
        RawGenerator(StubBackend("```json\n{}\n```")).generate_one(GenerationRequest("追问", 1))


def test_generate_one_retries_with_validation_feedback() -> None:
    valid = _final()
    invalid = json.loads(json.dumps(valid))
    invalid["expected"]["start_time"] = "明天下午"
    backend = SequenceBackend(
        [json.dumps(invalid, ensure_ascii=False), json.dumps(valid, ensure_ascii=False)]
    )
    sample = RawGenerator(backend).generate_one(GenerationRequest("追问", 1))
    assert sample.id == "x"
    assert len(backend.calls) == 2
    assert "start_time" in str(backend.calls[1][-1]["content"])


def test_generate_one_sends_strict_raw_schema_on_every_attempt() -> None:
    valid = _final()
    invalid = json.loads(json.dumps(valid))
    invalid["expected"]["start_time"] = "明天下午"
    backend = SequenceBackend(
        [json.dumps(invalid, ensure_ascii=False), json.dumps(valid, ensure_ascii=False)]
    )
    RawGenerator(backend).generate_one(GenerationRequest("追问", 1))
    assert len(backend.params) == 2
    assert all(params.response_schema_name == "phase03_raw" for params in backend.params)
    assert all(params.response_schema["additionalProperties"] is False for params in backend.params)


def test_generate_one_overrides_dpo_targets_from_scenario() -> None:
    valid = _final()
    valid["id"] = "phase03-confirm-002"
    valid["tags"] = ["确认", "多义短词"]
    valid["expected"].update(
        duration_minutes=60,
        missing_info=[],
        info_complete=True,
        confirmation=False,
        reply_type="appointment_paused",
        reply="好的，暂不预约。",
    )
    valid["dpo_targets"] = ["P5"]
    backend = StubBackend(json.dumps(valid, ensure_ascii=False))
    sample = RawGenerator(backend).generate_one(GenerationRequest("确认", 2, "confirm_reject"))
    assert sample.dpo_targets == ()


def test_generation_prompt_contains_full_contract() -> None:
    text = "\n".join(
        str(message["content"])
        for message in build_generation_messages(GenerationRequest("工具调用", 2))
    )
    for required in (
        "dpo_targets",
        "current_state",
        "expected.action",
        "P6",
        "P2P3",
        "只输出",
        "七个顶层字段",
        "gender_preference",
        "technician_gender",
        "technician_status",
        "reply_type",
        "tool_name",
        "duration_minutes",
        "自然消息只能有 role/content",
        "assistant 工具消息只能有 role/content/tool_calls",
        "工具结果只能有 role/tool_call_id/content",
        "工具调用类按场景使用 single_turn 或 multi_turn",
        'tags 必须严格为 ["工具调用","易混边界"]',
    ):
        assert required in text
