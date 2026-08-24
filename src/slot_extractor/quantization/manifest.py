"""Atomic, hash-verifiable manifests for quantization stages."""

import hashlib
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from .lineage import Lineage


class ManifestError(RuntimeError):
    """Raised when a stage manifest is invalid or its artifacts changed."""


@dataclass(frozen=True)
class ArtifactHash:
    path: str
    sha256: str


@dataclass(frozen=True)
class StageManifest:
    model_id: str
    stage: str
    status: Literal["running", "complete", "failed"]
    artifact_kind: str
    is_anchor: bool
    cache_key: str
    lineage: Lineage
    inputs: tuple[ArtifactHash, ...]
    outputs: tuple[ArtifactHash, ...]
    command: tuple[str, ...]
    error: str | None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_manifest_atomic(path: Path, manifest: StageManifest) -> None:
    if path.is_file():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ManifestError(f"cannot read existing manifest: {path}") from exc
        if existing.get("status") == "complete" and manifest.status != "complete":
            raise ManifestError("cannot overwrite completed manifest with partial state")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    payload = json.dumps(asdict(manifest), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
    try:
        with temporary.open("w", encoding="utf-8", newline="\n") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())
        temporary.replace(path)
    except OSError as exc:
        raise ManifestError(f"cannot write manifest: {path}") from exc


def read_and_verify_manifest(path: Path) -> StageManifest:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
        lineage_payload = payload["lineage"]
        lineage = Lineage(
            model_id=lineage_payload["model_id"],
            base_model=lineage_payload["base_model"],
            base_revision=lineage_payload["base_revision"],
            parent_model_id=lineage_payload.get("parent_model_id"),
            adapter_run_id=lineage_payload.get("adapter_run_id"),
            source_sha256=tuple(tuple(item) for item in lineage_payload["source_sha256"]),
            git_revision=lineage_payload["git_revision"],
            tool_versions=tuple(tuple(item) for item in lineage_payload["tool_versions"]),
        )
        manifest = StageManifest(
            model_id=payload["model_id"],
            stage=payload["stage"],
            status=payload["status"],
            artifact_kind=payload["artifact_kind"],
            is_anchor=payload["is_anchor"],
            cache_key=payload["cache_key"],
            lineage=lineage,
            inputs=tuple(ArtifactHash(**item) for item in payload["inputs"]),
            outputs=tuple(ArtifactHash(**item) for item in payload["outputs"]),
            command=tuple(payload["command"]),
            error=payload.get("error"),
        )
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ManifestError(f"invalid manifest: {path}") from exc
    for artifact in (*manifest.inputs, *manifest.outputs):
        artifact_path = Path(artifact.path)
        if not artifact_path.is_file():
            raise ManifestError(f"missing artifact: {artifact.path}")
        if sha256_file(artifact_path) != artifact.sha256:
            raise ManifestError(f"sha256 mismatch: {artifact.path}")
    return manifest
