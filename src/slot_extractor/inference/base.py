# src/slot_extractor/inference/base.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

from slot_extractor.schemas.results import GenerationResult


@dataclass(frozen=True)
class GenerationParams:
    temperature: float = 0.0
    max_tokens: int = 256


class Backend(Protocol):
    model: str

    def generate(self, prompt: str, params: GenerationParams | None = None) -> GenerationResult:
        raise NotImplementedError
