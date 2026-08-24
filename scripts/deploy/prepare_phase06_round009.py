"""Create hash-verifiable external-model manifests for the Round 009 app."""

from __future__ import annotations

import argparse
from pathlib import Path

from slot_extractor.quantization.lineage import Lineage, cache_key
from slot_extractor.quantization.manifest import (
    ArtifactHash,
    StageManifest,
    sha256_file,
    write_manifest_atomic,
)
from slot_extractor.quantization.registry import ModelRegistry


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/quantization/phase06-round009-app.yaml"),
    )
    args = parser.parse_args(argv)
    registry = ModelRegistry.from_config(args.config)
    for spec in registry.models:
        if not spec.artifact_path.is_file():
            raise FileNotFoundError(f"model missing: {spec.artifact_path}")
        digest = sha256_file(spec.artifact_path)
        lineage = Lineage(
            spec.model_id,
            spec.base_model,
            spec.base_revision,
            spec.parent_model_id,
            spec.adapter_run_id,
            (),
            "phase06-round009",
            (("llama.cpp", "c198af4dc"),),
        )
        manifest = StageManifest(
            spec.model_id,
            "app-integration",
            "complete",
            spec.artifact_kind,
            False,
            cache_key(lineage, "round009-app", {"sha256": digest}),
            lineage,
            (),
            (ArtifactHash(str(spec.artifact_path), digest),),
            ("external-model-registration",),
            None,
        )
        write_manifest_atomic(spec.manifest_path, manifest)
        print(f"ready: {spec.model_id}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
