"""Canonical Phase 05 model registry."""

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import yaml


class RegistryError(ValueError):
    """Raised when the canonical model registry is invalid."""


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    family: str
    size_b: float
    stage: Literal["base", "sft", "dpo"]
    variant: str
    base_model: str
    base_revision: str
    adapter_run_id: str | None
    parent_model_id: str | None
    artifact_kind: Literal["f16", "q4_k_m"]
    artifact_path: Path
    manifest_path: Path
    is_anchor: bool


@dataclass(frozen=True)
class ModelRegistry:
    models: tuple[ModelSpec, ...]

    def __post_init__(self) -> None:
        ids = [model.model_id for model in self.models]
        if len(ids) != len(set(ids)):
            raise RegistryError("duplicate model_id")
        if len(self.quantization_targets()) != 8:
            raise RegistryError("Phase 05 requires exactly 8 Q4_K_M targets")
        anchors = self.anchors()
        if len(anchors) != 2 or any(model.stage != "sft" for model in anchors):
            raise RegistryError("Phase 05 requires exactly 2 SFT F16 anchors")

    @classmethod
    def from_config(cls, path: Path) -> "ModelRegistry":
        try:
            payload = yaml.safe_load(path.read_text(encoding="utf-8"))
            records = payload["models"]
            models = tuple(
                ModelSpec(
                    model_id=record["model_id"],
                    family=record["family"],
                    size_b=float(record["size_b"]),
                    stage=record["stage"],
                    variant=record["variant"],
                    base_model=record["base_model"],
                    base_revision=record["base_revision"],
                    adapter_run_id=record.get("adapter_run_id"),
                    parent_model_id=record.get("parent_model_id"),
                    artifact_kind=record["artifact_kind"],
                    artifact_path=Path(record["artifact_path"]),
                    manifest_path=Path(record["manifest_path"]),
                    is_anchor=bool(record.get("is_anchor", False)),
                )
                for record in records
            )
        except (OSError, KeyError, TypeError, yaml.YAMLError) as exc:
            raise RegistryError(f"invalid registry config: {path}") from exc
        return cls(models)

    def get(self, model_id: str) -> ModelSpec:
        try:
            return next(model for model in self.models if model.model_id == model_id)
        except StopIteration as exc:
            raise RegistryError(f"unknown model_id: {model_id}") from exc

    def quantization_targets(self) -> tuple[ModelSpec, ...]:
        return tuple(
            model
            for model in self.models
            if model.artifact_kind == "q4_k_m" and not model.is_anchor
        )

    def anchors(self) -> tuple[ModelSpec, ...]:
        return tuple(model for model in self.models if model.is_anchor)
