# src/slot_extractor/inference/factory.py
from __future__ import annotations

from pathlib import Path

import yaml

from slot_extractor.inference.base import Backend
from slot_extractor.inference.llama_server import LlamaServerBackend, LlamaServerConfig
from slot_extractor.inference.mock import MockBackend, mock_response_from_config


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
            )
        )

    raise ValueError(f"unsupported backend: {backend}")
