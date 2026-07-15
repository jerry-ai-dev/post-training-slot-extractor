# src/slot_extractor/inference/llama_server.py
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from slot_extractor.inference.base import GenerationParams
from slot_extractor.schemas.results import GenerationResult


@dataclass(frozen=True)
class LlamaServerConfig:
    model: str
    base_url: str
    api_key: str = "local-no-key"
    timeout_s: float = 120.0
    temperature: float = 0.0
    max_tokens: int = 256


class LlamaServerBackend:
    def __init__(self, config: LlamaServerConfig) -> None:
        self.model = config.model
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key
        self._timeout_s = config.timeout_s
        self._default_params = GenerationParams(
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )

    def generate(
        self, messages: list[dict[str, Any]], params: GenerationParams | None = None
    ) -> GenerationResult:
        generation_params = params or self._default_params
        api_messages = [
            {key: value for key, value in message.items() if not key.startswith("_")}
            for message in messages
        ]
        started = time.perf_counter()
        response = httpx.post(
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json={
                "model": self.model,
                "messages": api_messages,
                "temperature": generation_params.temperature,
                "max_tokens": generation_params.max_tokens,
                "chat_template_kwargs": {"enable_thinking": False},
            },
            timeout=self._timeout_s,
        )
        total_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        text = payload["choices"][0]["message"]["content"]
        usage = payload.get("usage", {})
        output_tokens = usage.get("completion_tokens")
        tokens_per_s = output_tokens * 1000 / total_ms if output_tokens else None
        timings = payload.get("timings", {})
        return GenerationResult(
            text=text,
            model=self.model,
            prefill_ms=timings.get("prompt_ms"),
            first_token_ms=timings.get("predicted_ms"),
            total_ms=total_ms,
            output_tokens=output_tokens,
            tokens_per_s=tokens_per_s,
            raw=payload,
        )
