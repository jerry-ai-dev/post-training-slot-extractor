"""Deterministic quantization lineage and cache keys."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Lineage:
    model_id: str
    base_model: str
    base_revision: str
    parent_model_id: str | None
    adapter_run_id: str | None
    source_sha256: tuple[tuple[str, str], ...]
    git_revision: str
    tool_versions: tuple[tuple[str, str], ...]


def cache_key(lineage: Lineage, stage: str, parameters: Mapping[str, str]) -> str:
    payload = {
        "lineage": asdict(lineage),
        "stage": stage,
        "parameters": dict(sorted(parameters.items())),
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
