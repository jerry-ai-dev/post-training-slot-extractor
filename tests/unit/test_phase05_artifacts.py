import json
from pathlib import Path

from scripts.eval.phase05_artifacts import (
    is_complete,
    models_to_run,
    write_failure,
    write_local_marker,
    write_matrix_summary,
    write_model_result,
)


def test_failure_isolated_and_complete_requires_all_artifacts(tmp_path: Path):
    failed_id = "qwen3-0.6b-base-q4-k-m"
    complete_id = "qwen3-0.6b-sft-q4-k-m"
    write_failure(tmp_path, failed_id, RuntimeError("server exited"))
    assert json.loads((tmp_path / failed_id / "failure.json").read_text())["status"] == "failed"
    assert not is_complete(tmp_path, failed_id)
    write_model_result(
        tmp_path,
        complete_id,
        {"status": "complete", "quality": {}, "workloads": {}, "manifest": {}},
    )
    write_local_marker(tmp_path, {"marker": "phase05-local", "models": [complete_id]})
    assert is_complete(tmp_path, complete_id)


def test_skip_complete_does_not_skip_partial_or_failed_models(tmp_path: Path):
    ids = ["failed", "partial", "complete"]
    write_failure(tmp_path, "failed", RuntimeError("x"))
    (tmp_path / "partial").mkdir()
    (tmp_path / "partial" / "result.json").write_text("{}")
    write_model_result(
        tmp_path, "complete", {"status": "complete", "quality": {}, "workloads": {}, "manifest": {}}
    )
    assert models_to_run(tmp_path, ids, skip_complete=True) == ["failed", "partial"]


def test_matrix_summary_rejects_selection_or_gate_fields(tmp_path: Path):
    try:
        write_matrix_summary(tmp_path, {"winner": "x"})
    except ValueError as error:
        assert "forbidden" in str(error)
    else:
        raise AssertionError("selection field was accepted")
