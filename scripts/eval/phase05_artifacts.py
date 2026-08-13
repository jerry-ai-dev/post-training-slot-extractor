"""Atomic Phase 05 evaluation artifact writers."""

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

REQUIRED_MODEL_FILES = (
    "result.json", "workloads.json", "quality.json", "manifest.json", "complete.marker"
)
FORBIDDEN_KEYS = {"winner", "threshold", "pass", "fail"}


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(f"{path}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary.replace(path)
    return path


def write_model_result(root: Path, model_id: str, payload: Mapping[str, Any]) -> Path:
    if payload.get("status") != "complete":
        raise ValueError("model result status must be complete")
    directory = root / model_id
    result = _write_json(directory / "result.json", payload)
    _write_json(directory / "workloads.json", payload.get("workloads", {}))
    _write_json(directory / "quality.json", payload.get("quality", {}))
    _write_json(directory / "manifest.json", payload.get("manifest", {}))
    (directory / "complete.marker").write_text("complete\n", encoding="utf-8", newline="\n")
    return result


def write_failure(root: Path, model_id: str, error: BaseException) -> Path:
    return _write_json(
        root / model_id / "failure.json",
        {"status": "failed", "error_type": type(error).__name__, "error": str(error)},
    )


def is_complete(root: Path, model_id: str) -> bool:
    directory = root / model_id
    return all((directory / name).is_file() for name in REQUIRED_MODEL_FILES)


def models_to_run(root: Path, model_ids: Sequence[str], *, skip_complete: bool) -> list[str]:
    if not skip_complete:
        return list(model_ids)
    return [model_id for model_id in model_ids if not is_complete(root, model_id)]


def _check_forbidden(payload: Mapping[str, Any]) -> None:
    for key, value in payload.items():
        if key.lower() in FORBIDDEN_KEYS:
            raise ValueError(f"forbidden selection/gate field: {key}")
        if isinstance(value, Mapping):
            _check_forbidden(value)


def write_matrix_summary(root: Path, payload: Mapping[str, Any]) -> Path:
    _check_forbidden(payload)
    return _write_json(root / "matrix-summary.json", payload)


def write_local_marker(root: Path, marker: Mapping[str, Any]) -> Path:
    return _write_json(root / "local-run.json", marker)
