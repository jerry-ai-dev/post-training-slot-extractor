# tests/unit/test_inference_backend.py
from pathlib import Path

import httpx

from slot_extractor.inference.factory import build_backend_from_config


def test_build_mock_backend_from_config(tmp_path: Path) -> None:
    config = tmp_path / "mock.yaml"
    config.write_text(
        """
backend: mock
model: mock-test
responses:
  case-1:
    text: '{"action":"final"}'
    prefill_ms: 1
    first_token_ms: 2
    total_ms: 3
    output_tokens: 4
""",
        encoding="utf-8",
    )

    backend = build_backend_from_config(config)
    result = backend.generate([{"role": "system", "content": "Sample ID: case-1"}])

    assert result.text == '{"action":"final"}'
    assert result.model == "mock-test"
    assert result.total_ms == 3
    assert result.tokens_per_s == 4000 / 3


def test_llama_backend_uses_configured_generation_params_and_hides_metadata(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "llama.yaml"
    config.write_text(
        """
backend: llama_server
model: local-test
base_url: http://127.0.0.1:8080/v1
temperature: 0.25
max_tokens: 512
timeout_s: 30
""",
        encoding="utf-8",
    )
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "choices": [{"message": {"content": '{"action":"final"}'}}],
                "usage": {"completion_tokens": 4},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    backend = build_backend_from_config(config)
    backend.generate(
        [
            {
                "role": "system",
                "content": "只输出 JSON",
                "_sample_id": "case-1",
            },
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {
                            "name": "find_technicians",
                            "arguments": '{"technician_name":"王芳"}',
                        },
                    }
                ],
            },
            {
                "role": "tool",
                "tool_call_id": "call-1",
                "content": '{"status":"available"}',
            },
        ]
    )

    assert captured["json"]["temperature"] == 0.25
    assert captured["json"]["max_tokens"] == 512
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "只输出 JSON"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call-1",
                    "type": "function",
                    "function": {
                        "name": "find_technicians",
                        "arguments": '{"technician_name":"王芳"}',
                    },
                }
            ],
        },
        {"role": "tool", "tool_call_id": "call-1", "content": '{"status":"available"}'},
    ]


def test_responses_backend_maps_tool_history_and_reads_output_text(
    tmp_path: Path, monkeypatch
) -> None:
    config = tmp_path / "responses.yaml"
    config.write_text(
        """
backend: openai_responses
model: gpt-test
base_url_env: TEST_OPENAI_BASE_URL
api_key_env: TEST_OPENAI_API_KEY
max_tokens: 256
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_OPENAI_BASE_URL", "http://example.test/v1")
    monkeypatch.setenv("TEST_OPENAI_API_KEY", "secret")
    captured: dict = {}

    def fake_post(url, *, headers, json, timeout):
        captured.update({"url": url, "headers": headers, "json": json, "timeout": timeout})
        return httpx.Response(
            200,
            request=httpx.Request("POST", url),
            json={
                "output": [
                    {
                        "type": "message",
                        "content": [
                            {"type": "output_text", "text": '{"action":"final"}'}
                        ],
                    }
                ],
                "usage": {"output_tokens": 4},
            },
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    backend = build_backend_from_config(config)
    result = backend.generate(
        [
            {"role": "system", "content": "只输出 JSON", "_sample_id": "case-1"},
            {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
                        "function": {"name": "find_technicians", "arguments": "{}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call-1", "content": '{"status":"ok"}'},
        ]
    )

    assert result.text == '{"action":"final"}'
    assert captured["url"] == "http://example.test/v1/responses"
    assert captured["json"]["input"] == [
        {"role": "system", "content": "只输出 JSON"},
        {"type": "function_call", "call_id": "call-1", "name": "find_technicians", "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call-1", "output": '{"status":"ok"}'},
    ]
