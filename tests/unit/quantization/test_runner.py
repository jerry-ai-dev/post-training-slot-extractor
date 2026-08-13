import subprocess
from pathlib import Path

import pytest

from slot_extractor.quantization.runner import CommandRunner, ToolRunnerError


def test_runner_sets_cpu_environment_and_writes_log(monkeypatch, tmp_path: Path):
    captured = {}

    def fake_run(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return subprocess.CompletedProcess(argv, 0, "tool output", "tool warning")

    monkeypatch.setattr(subprocess, "run", fake_run)
    log_path = tmp_path / "run.log"

    CommandRunner(threads=8).run(["fake-tool", "--version"], tmp_path, log_path)

    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""
    assert captured["env"]["OMP_NUM_THREADS"] == "8"
    assert captured["text"] is True
    assert captured["encoding"] == "utf-8"
    assert captured["check"] is False
    assert "tool output" in log_path.read_text(encoding="utf-8")
    assert "tool warning" in log_path.read_text(encoding="utf-8")


def test_runner_includes_command_and_exit_code_in_error(monkeypatch, tmp_path: Path):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 7, "", "bad input")

    monkeypatch.setattr(subprocess, "run", fake_run)

    with pytest.raises(ToolRunnerError, match="fake-tool bad.*exit code 7"):
        CommandRunner().run(["fake-tool", "bad"], tmp_path, tmp_path / "run.log")


def test_version_returns_first_nonempty_output_line(monkeypatch, tmp_path: Path):
    def fake_run(argv, **kwargs):
        return subprocess.CompletedProcess(argv, 0, "fake-tool 1.2\nmore", "")

    monkeypatch.setattr(subprocess, "run", fake_run)

    assert CommandRunner().version(Path("fake-tool"), cwd=tmp_path) == "fake-tool 1.2"
