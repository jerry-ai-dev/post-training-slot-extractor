# src/slot_extractor/inference/factory.py
from __future__ import annotations

import os
from pathlib import Path

import yaml

from slot_extractor.inference.base import Backend
from slot_extractor.inference.llama_server import LlamaServerBackend, LlamaServerConfig
from slot_extractor.inference.mock import MockBackend, mock_response_from_config
from slot_extractor.inference.openai_responses import (
    OpenAIResponsesBackend,
    OpenAIResponsesConfig,
)


def build_backend_from_config(path: str | Path) -> Backend:
    with Path(path).open("r", encoding="utf-8") as file:
        config = yaml.safe_load(file)

    backend = config["backend"]
    if backend == "mock":
        responses = {
            sample_id: mock_response_from_config(value)
            for sample_id, value in config["responses"].items()
        }
        return MockBackend(model=config["model"], responses=responses)

    if backend == "llama_server":
        return LlamaServerBackend(
            LlamaServerConfig(
                model=config["model"],
                base_url=config["base_url"],
                api_key=config.get("api_key", "local-no-key"),
                timeout_s=float(config.get("timeout_s", 120)),
                temperature=float(config.get("temperature", 0.0)),
                max_tokens=int(config.get("max_tokens", 256)),
            )
        )

    if backend == "openai_responses":
        base_url = config.get("base_url") or os.environ[config["base_url_env"]]
        api_key = config.get("api_key") or os.environ[config["api_key_env"]]
        temperature = config.get("temperature")
        return OpenAIResponsesBackend(
            OpenAIResponsesConfig(
                model=config["model"],
                base_url=base_url,
                api_key=api_key,
                timeout_s=float(config.get("timeout_s", 180)),
                temperature=None if temperature is None else float(temperature),
                max_tokens=int(config.get("max_tokens", 512)),
            )
        )

    raise ValueError(f"unsupported backend: {backend}")
