from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import httpx

from slot_extractor.inference.base import GenerationParams
from slot_extractor.schemas.results import GenerationResult


@dataclass(frozen=True)
class OpenAIResponsesConfig:
    model: str
    base_url: str
    api_key: str
    timeout_s: float = 180.0
    temperature: float | None = None
    max_tokens: int = 512


def _responses_input(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for message in messages:
        role = message.get("role")
        content = message.get("content")
        if role == "assistant" and isinstance(message.get("tool_calls"), list):
            if isinstance(content, str) and content:
                items.append({"role": "assistant", "content": content})
            for tool_call in message["tool_calls"]:
                function = tool_call.get("function", {})
                items.append(
                    {
                        "type": "function_call",
                        "call_id": tool_call.get("id"),
                        "name": function.get("name"),
                        "arguments": function.get("arguments", "{}"),
                    }
                )
            continue
        if role == "tool":
            items.append(
                {
                    "type": "function_call_output",
                    "call_id": message.get("tool_call_id"),
                    "output": content,
                }
            )
            continue
        if role in {"system", "user", "assistant"} and isinstance(content, str):
            items.append({"role": role, "content": content})
    return items


def _response_text(payload: dict[str, Any]) -> str:
    output_text = payload.get("output_text")
    if isinstance(output_text, str):
        return output_text
    texts: list[str] = []
    for item in payload.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            text = content.get("text")
            if content.get("type") == "output_text" and isinstance(text, str):
                texts.append(text)
    if not texts:
        raise ValueError("Responses API payload contains no output text")
    return "".join(texts)


class OpenAIResponsesBackend:
    def __init__(self, config: OpenAIResponsesConfig) -> None:
        self.model = config.model
        self._base_url = config.base_url.rstrip("/")
        self._api_key = config.api_key
        self._timeout_s = config.timeout_s
        self._temperature = config.temperature
        self._default_params = GenerationParams(max_tokens=config.max_tokens)

    def generate(
        self, messages: list[dict[str, Any]], params: GenerationParams | None = None
    ) -> GenerationResult:
        generation_params = params or self._default_params
        request_payload: dict[str, Any] = {
            "model": self.model,
            "input": _responses_input(messages),
            "max_output_tokens": generation_params.max_tokens,
        }
        if self._temperature is not None:
            request_payload["temperature"] = self._temperature

        started = time.perf_counter()
        response = httpx.post(
            f"{self._base_url}/responses",
            headers={"Authorization": f"Bearer {self._api_key}"},
            json=request_payload,
            timeout=self._timeout_s,
        )
        total_ms = (time.perf_counter() - started) * 1000
        response.raise_for_status()
        payload = response.json()
        usage = payload.get("usage", {})
        output_tokens = usage.get("output_tokens")
        tokens_per_s = output_tokens * 1000 / total_ms if output_tokens else None
        return GenerationResult(
            text=_response_text(payload),
            model=self.model,
            prefill_ms=None,
            first_token_ms=None,
            total_ms=total_ms,
            output_tokens=output_tokens,
            tokens_per_s=tokens_per_s,
            raw=payload,
        )
