"""Local rotating JSONL diagnostics for the Phase 05 comparison app."""

import json
import logging
from datetime import UTC, datetime
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Lock
from typing import Any


class DiagnosticLog:
    def __init__(
        self, path: Path, *, max_bytes: int = 10 * 1024 * 1024, backup_count: int = 5
    ) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._lock = Lock()
        self._logger = logging.Logger(f"phase05.{path.resolve()}", level=logging.INFO)
        handler = RotatingFileHandler(
            path, maxBytes=max_bytes, backupCount=backup_count, encoding="utf-8"
        )
        handler.setFormatter(logging.Formatter("%(message)s"))
        self._logger.addHandler(handler)
        self._logger.propagate = False

    def write(self, event: str, **fields: Any) -> None:
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "event": event,
            **fields,
        }
        try:
            payload = json.dumps(record, ensure_ascii=False, default=str)
            with self._lock:
                self._logger.info(payload)
        except Exception:
            # Diagnostics must never interrupt model loading or inference.
            return

    def close(self) -> None:
        with self._lock:
            for handler in tuple(self._logger.handlers):
                handler.close()
                self._logger.removeHandler(handler)
