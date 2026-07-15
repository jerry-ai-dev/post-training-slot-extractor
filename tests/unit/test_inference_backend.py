# tests/unit/test_inference_backend.py
from pathlib import Path

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
    result = backend.generate("Sample ID: case-1\nOutput JSON:")

    assert result.text == '{"action":"final"}'
    assert result.model == "mock-test"
    assert result.total_ms == 3
    assert result.tokens_per_s == 4000 / 3
