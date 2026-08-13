from dataclasses import replace
from pathlib import Path

import pytest

from slot_extractor.quantization.lineage import Lineage
from slot_extractor.quantization.manifest import (
    ArtifactHash,
    ManifestError,
    StageManifest,
    read_and_verify_manifest,
    sha256_file,
    write_manifest_atomic,
)


def complete_manifest_for(output: Path) -> StageManifest:
    return StageManifest(
        model_id="model",
        stage="verify",
        status="complete",
        artifact_kind="q4_k_m",
        is_anchor=False,
        cache_key="key",
        lineage=Lineage(
            "model", "base", "main", None, None, (), "deadbeef", ()
        ),
        inputs=(),
        outputs=(ArtifactHash(str(output), sha256_file(output)),),
        command=("verify",),
        error=None,
    )


def test_manifest_round_trip_and_verification(tmp_path: Path):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"ok")
    manifest_path = tmp_path / "manifest.json"
    expected = complete_manifest_for(output)

    write_manifest_atomic(manifest_path, expected)

    assert read_and_verify_manifest(manifest_path) == expected
    assert not manifest_path.with_suffix(".json.tmp").exists()


def test_manifest_verification_rejects_changed_output(tmp_path: Path):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"ok")
    manifest_path = tmp_path / "manifest.json"
    write_manifest_atomic(manifest_path, complete_manifest_for(output))

    output.write_bytes(b"tampered")

    with pytest.raises(ManifestError, match="sha256 mismatch"):
        read_and_verify_manifest(manifest_path)


def test_complete_manifest_cannot_be_overwritten_by_partial_state(tmp_path: Path):
    output = tmp_path / "model.gguf"
    output.write_bytes(b"ok")
    manifest_path = tmp_path / "manifest.json"
    complete = complete_manifest_for(output)
    write_manifest_atomic(manifest_path, complete)

    with pytest.raises(ManifestError, match="completed manifest"):
        write_manifest_atomic(manifest_path, replace(complete, status="failed"))
