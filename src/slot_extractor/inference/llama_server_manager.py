"""Registry-driven llama-server lifecycle management."""

import argparse
import os
import subprocess
import time
from pathlib import Path
from urllib.request import urlopen

from slot_extractor.quantization.manifest import read_and_verify_manifest
from slot_extractor.quantization.registry import ModelRegistry


class ServerError(RuntimeError):
    """Raised when llama-server cannot start or become healthy."""


class LlamaServerManager:
    def __init__(
        self,
        registry: ModelRegistry,
        server: Path,
        host: str = "127.0.0.1",
        port: int = 8080,
        threads: int = 8,
    ) -> None:
        self.registry = registry
        self.server = server
        self.host = host
        self.port = port
        self.threads = threads
        self._logs: dict[int, object] = {}

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}/v1"

    def start(self, model_id: str, log_path: Path) -> subprocess.Popen[str]:
        spec = self.registry.get(model_id)
        manifest = read_and_verify_manifest(spec.manifest_path)
        if manifest.status != "complete":
            raise ServerError(f"model manifest is not complete: {model_id}")
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        log_path.parent.mkdir(parents=True, exist_ok=True)
        log = log_path.open("a", encoding="utf-8")
        argv = [
            str(self.server),
            "-m",
            str(spec.artifact_path),
            "--host",
            self.host,
            "--port",
            str(self.port),
            "--threads",
            str(self.threads),
        ]
        try:
            process = subprocess.Popen(
                argv,
                stdout=log,
                stderr=subprocess.STDOUT,
                env=environment,
                text=True,
            )
        except OSError as exc:
            log.close()
            raise ServerError(f"cannot start llama-server: {exc}") from exc
        self._logs[id(process)] = log
        return process

    def wait_ready(self, process: subprocess.Popen[str], timeout_s: float) -> None:
        deadline = time.monotonic() + timeout_s
        last_error = "not ready"
        while True:
            if process.poll() is not None:
                raise ServerError(f"llama-server exited with code {process.returncode}")
            try:
                with urlopen(f"{self.base_url}/models", timeout=1) as response:
                    if response.status == 200:
                        return
            except OSError as exc:
                last_error = str(exc)
            if time.monotonic() >= deadline:
                raise ServerError(f"llama-server not ready: {last_error}")
            time.sleep(0.1)

    def stop(self, process: subprocess.Popen[str]) -> None:
        if process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=5)
        log = self._logs.pop(id(process), None)
        if log is not None:
            log.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Start a registry model with llama-server.")
    parser.add_argument("--config", type=Path, default=Path("configs/quantization/phase05.yaml"))
    parser.add_argument("--server", type=Path, required=True)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args(argv)
    manager = LlamaServerManager(
        ModelRegistry.from_config(args.config), args.server, port=args.port, threads=args.threads
    )
    process = manager.start(args.model_id, Path("models/quantization/server.log"))
    try:
        manager.wait_ready(process, 60)
        return process.wait()
    finally:
        manager.stop(process)


if __name__ == "__main__":
    raise SystemExit(main())
