from pathlib import Path

from scripts.eval.import_phase06_results import _sha256


def test_sha256(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_bytes(b"phase06\n")
    assert _sha256(artifact) == "f839512aad59a2983c975dacbd25d7810107122a43520e5c61417c50f25a3ae3"
