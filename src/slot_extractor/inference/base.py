# src/slot_extractor/inference/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from slot_extractor.schemas.results import GenerationResult


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.0
    max_tokens: int = 256
    response_schema: dict[str, Any] | None = None
    response_schema_name: str = "response"


class Backend(Protocol):
    model: str

    def generate(
        self, messages: list[dict[str, Any]], params: GenerationParams | None = None
    ) -> GenerationResult:
        raise NotImplementedError
