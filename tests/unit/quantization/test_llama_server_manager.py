from pathlib import Path
from unittest.mock import Mock

import pytest

from slot_extractor.inference import llama_server_manager as module
from slot_extractor.inference.llama_server_manager import LlamaServerManager, ServerError
from slot_extractor.quantization.registry import ModelRegistry


def test_manager_uses_registry_path_and_cpu_flags(monkeypatch, tmp_path: Path):
    registry = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))
    captured = {}
    process = Mock()

    def fake_popen(argv, **kwargs):
        captured.update(argv=argv, **kwargs)
        return process

    monkeypatch.setattr(module, "read_and_verify_manifest", lambda path: Mock(status="complete"))
    monkeypatch.setattr(module.subprocess, "Popen", fake_popen)
    manager = LlamaServerManager(registry, Path("llama-server.exe"))

    assert manager.start("qwen3-0.6b-base-q4-k-m", tmp_path / "server.log") is process
    assert captured["argv"] == [
        "llama-server.exe", "-m",
        str(registry.get("qwen3-0.6b-base-q4-k-m").artifact_path),
        "--host", "127.0.0.1", "--port", "8080", "--threads", "8",
    ]
    assert captured["env"]["CUDA_VISIBLE_DEVICES"] == ""


def test_manager_stops_process_after_timeout(monkeypatch):
    registry = ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))
    process = Mock()
    process.poll.return_value = None
    monkeypatch.setattr(module, "urlopen", Mock(side_effect=OSError("not ready")))
    manager = LlamaServerManager(registry, Path("llama-server.exe"))

    with pytest.raises(ServerError, match="not ready"):
        manager.wait_ready(process, timeout_s=0)
    manager.stop(process)
    process.terminate.assert_called_once()
