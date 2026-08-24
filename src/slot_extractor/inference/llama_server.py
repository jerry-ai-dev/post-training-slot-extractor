# src/slot_extractor/inference/llama_server.py
from __future__ import annotations

import json
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
        request = {
            "model": self.model,
            "messages": api_messages,
            "temperature": generation_params.temperature,
            "max_tokens": generation_params.max_tokens,
            "chat_template_kwargs": {"enable_thinking": False},
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        started = time.perf_counter()
        first_token_ms: float | None = None
        chunks: list[str] = []
        usage: dict[str, Any] = {}
        timings: dict[str, Any] = {}
        raw_events: list[dict[str, Any]] = []
        with httpx.stream(
            "POST",
            f"{self._base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=request,
            timeout=self._timeout_s,
        ) as response:
            response.raise_for_status()
            for line in response.iter_lines():
                if not line.startswith("data:"):
                    continue
                data = line.removeprefix("data:").strip()
                if not data or data == "[DONE]":
                    continue
                event = json.loads(data)
                raw_events.append(event)
                usage.update(event.get("usage") or {})
                timings.update(event.get("timings") or {})
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = choices[0].get("delta", {}).get("content")
                if content:
                    if first_token_ms is None:
                        first_token_ms = (time.perf_counter() - started) * 1000
                    chunks.append(content)
        total_ms = (time.perf_counter() - started) * 1000
        text = "".join(chunks)
        output_tokens = usage.get("completion_tokens")
        decode_ms = timings.get("predicted_ms")
        decode_tokens_per_s = timings.get("predicted_per_second")
        tokens_per_s = decode_tokens_per_s
        if tokens_per_s is None and output_tokens and decode_ms:
            tokens_per_s = output_tokens * 1000 / decode_ms
        return GenerationResult(
            text=text,
            model=self.model,
            prefill_ms=timings.get("prompt_ms"),
            first_token_ms=first_token_ms,
            total_ms=total_ms,
            output_tokens=output_tokens,
            tokens_per_s=tokens_per_s,
            input_tokens=usage.get("prompt_tokens"),
            decode_ms=decode_ms,
            prefill_tokens_per_s=timings.get("prompt_per_second"),
            decode_tokens_per_s=decode_tokens_per_s,
            raw={"events": raw_events, "usage": usage, "timings": timings},
        )
