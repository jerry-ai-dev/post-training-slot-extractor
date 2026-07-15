# src/slot_extractor/inference/mock.py
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from slot_extractor.inference.base import GenerationParams
from slot_extractor.schemas.results import GenerationResult


@dataclass(frozen=True)
class MockResponse:
    text: str
    prefill_ms: float
    first_token_ms: float
    total_ms: float
    output_tokens: int


class MockBackend:
    def __init__(self, model: str, responses: dict[str, MockResponse]) -> None:
        self.model = model
        self._responses = responses

    def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult:
        match = re.search(r"Sample ID:\s*([^\n]+)", prompt)
        sample_id = match.group(1).strip() if match else ""
        if sample_id not in self._responses:
            raise ValueError(f"mock response not configured for sample id: {sample_id}")
        response = self._responses[sample_id]
        tokens_per_s = response.output_tokens * 1000 / response.total_ms
        return GenerationResult(
            text=response.text,
            model=self.model,
            prefill_ms=response.prefill_ms,
            first_token_ms=response.first_token_ms,
            total_ms=response.total_ms,
            output_tokens=response.output_tokens,
            tokens_per_s=tokens_per_s,
            raw={"backend": "mock"},
        )


def mock_response_from_config(config: dict[str, Any]) -> MockResponse:
    return MockResponse(
        text=str(config["text"]),
        prefill_ms=float(config["prefill_ms"]),
        first_token_ms=float(config["first_token_ms"]),
        total_ms=float(config["total_ms"]),
        output_tokens=int(config["output_tokens"]),
    )
