"""Windows CPU subprocess adapter for Phase 05 tools."""

import os
import shutil
import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path


class ToolRunnerError(RuntimeError):
    """Raised when an external quantization tool fails."""


@dataclass(frozen=True)
class Toolchain:
    resolve: Path
    merge: Path
    convert_f16: Path
    imatrix: Path
    quantize: Path
    server: Path


class CommandRunner:
    def __init__(self, *, threads: int = 8) -> None:
        if threads < 1:
            raise ValueError("threads must be positive")
        self.threads = threads

    def run(
        self, argv: Sequence[str], cwd: Path, log_path: Path
    ) -> subprocess.CompletedProcess[str]:
        if not argv:
            raise ToolRunnerError("command cannot be empty")
        command = [self._resolve_executable(argv[0]), *argv[1:]]
        environment = os.environ.copy()
        environment["CUDA_VISIBLE_DEVICES"] = ""
        environment["OMP_NUM_THREADS"] = str(self.threads)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            completed = subprocess.run(
                command,
                cwd=cwd,
                env=environment,
                text=True,
                encoding="utf-8",
                errors="replace",
                capture_output=True,
                check=False,
            )
        except OSError as exc:
            raise ToolRunnerError(f"cannot start {' '.join(command)}: {exc}") from exc
        with log_path.open("a", encoding="utf-8", newline="\n") as log:
            log.write(f"$ {' '.join(command)}\n")
            if completed.stdout:
                log.write(completed.stdout.rstrip("\n") + "\n")
            if completed.stderr:
                log.write(completed.stderr.rstrip("\n") + "\n")
        if completed.returncode != 0:
            raise ToolRunnerError(
                f"command {' '.join(command)} failed with exit code {completed.returncode}"
            )
        return completed

    def version(self, executable: Path, *, cwd: Path | None = None) -> str:
        completed = self.run(
            [str(executable), "--version"],
            cwd or Path.cwd(),
            (cwd or Path.cwd()) / ".tool-version.log",
        )
        output = completed.stdout or completed.stderr
        return next((line.strip() for line in output.splitlines() if line.strip()), "unknown")

    @staticmethod
    def _resolve_executable(executable: str) -> str:
        path = Path(executable)
        if path.parent != Path(".") or path.is_absolute():
            return str(path)
        return shutil.which(executable) or executable
