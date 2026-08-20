from collections.abc import Callable
from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from functools import cache
from pathlib import Path
from threading import Lock, RLock
from time import perf_counter
from traceback import format_exc
from uuid import uuid4

import yaml
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from slot_extractor.inference.base import Backend
from slot_extractor.inference.llama_server import LlamaServerBackend, LlamaServerConfig
from slot_extractor.inference.llama_server_manager import LlamaServerManager
from slot_extractor.quantization.manifest import read_and_verify_manifest
from slot_extractor.quantization.registry import ModelRegistry, ModelSpec

from .diagnostics import DiagnosticLog
from .find_technicians import FindTechniciansExecutor
from .fixture_store import FixtureStore
from .models import CompareEvent
from .ndjson import encode_event, encode_side_status
from .orchestrator import ConversationOrchestrator

STATIC = Path(__file__).parent / "static"
SHANGHAI_TIMEZONE = timezone(timedelta(hours=8))


def _shanghai_now() -> datetime:
    return datetime.now(SHANGHAI_TIMEZONE)


class TimedBackend:
    def __init__(self, backend: Backend) -> None:
        self.backend = backend
        self.model = backend.model
        self.inference_duration_ms = 0.0

    def generate(self, messages, params=None):
        started = perf_counter()
        try:
            return self.backend.generate(messages, params)
        finally:
            self.inference_duration_ms += (perf_counter() - started) * 1000


class CompareRequest(BaseModel):
    left_model_id: str
    right_model_id: str
    mode: str = "sequential"
    user_input: str
    left_history: list[dict] = []
    right_history: list[dict] = []


class LoadModelRequest(BaseModel):
    model_id: str


class ClientLogRequest(BaseModel):
    level: str
    message: str
    context: dict = {}


class ResidentModelSlots:
    def __init__(self, registry, backend_factory=None, diagnostics=None) -> None:
        self.registry = registry
        self.backend_factory = backend_factory
        self.diagnostics = diagnostics
        self._slots = {}
        self._lock = RLock()

    def load(self, side: str, model_id: str) -> Backend:
        if side not in {"left", "right"}:
            raise ValueError("side must be left or right")
        with self._lock:
            current = self._slots.get(side)
            if current and current["model_id"] == model_id:
                self.diagnostics.write("model_reused", side=side, model_id=model_id)
                return current["backend"]
            self.unload(side)
            spec = self.registry.get(model_id)
            self.diagnostics.write("model_load_started", side=side, model_id=model_id)
            try:
                if self.backend_factory is not None:
                    backend = self.backend_factory(spec)
                    manager = process = None
                else:
                    config = yaml.safe_load(
                        Path("configs/quantization/phase05.yaml").read_text(encoding="utf-8")
                    )
                    port = 18080 if side == "left" else 18081
                    manager = LlamaServerManager(
                        self.registry, Path(config["toolchain"]["server"]), port=port
                    )
                    process = manager.start(
                        model_id,
                        Path("reports/phase05/app") / model_id / f"{side}-server.log",
                    )
                    manager.wait_ready(process, 60)
                    backend = LlamaServerBackend(
                        LlamaServerConfig(model=model_id, base_url=manager.base_url)
                    )
            except Exception:
                if "manager" in locals() and manager is not None and process is not None:
                    manager.stop(process)
                self.diagnostics.write(
                    "model_load_failed",
                    side=side,
                    model_id=model_id,
                    traceback=format_exc(),
                )
                raise
            self._slots[side] = {
                "model_id": model_id,
                "backend": backend,
                "manager": manager,
                "process": process,
            }
            self.diagnostics.write("model_ready", side=side, model_id=model_id)
            return backend

    def get(self, side: str, model_id: str) -> Backend:
        with self._lock:
            current = self._slots.get(side)
            if not current or current["model_id"] != model_id:
                return self.load(side, model_id)
            return current["backend"]

    def unload(self, side: str) -> None:
        with self._lock:
            current = self._slots.pop(side, None)
            if current and current["manager"] is not None:
                current["manager"].stop(current["process"])
            if current:
                self.diagnostics.write(
                    "model_unloaded", side=side, model_id=current["model_id"]
                )

    def close(self) -> None:
        for side in ("left", "right"):
            self.unload(side)


def create_app(
    store: FixtureStore | None = None,
    registry: ModelRegistry | None = None,
    backend_factory: Callable[[ModelSpec], Backend] | None = None,
    log_path: Path = Path("reports/phase05/app/app.jsonl"),
    now_provider: Callable[[], datetime] | None = None,
) -> FastAPI:
    now_provider = now_provider or _shanghai_now
    store = store or FixtureStore.from_yaml(
        Path("data/fixtures/technicians/phase05-v1.yaml"),
        target_date=now_provider().date(),
    )
    registry = registry or ModelRegistry.from_config(Path("configs/quantization/phase05.yaml"))
    executor = FindTechniciansExecutor(store)
    comparison_lock = Lock()
    diagnostics = DiagnosticLog(log_path)
    model_slots = ResidentModelSlots(registry, backend_factory, diagnostics)

    @asynccontextmanager
    async def lifespan(_app):
        diagnostics.write("app_started")
        yield
        model_slots.close()
        diagnostics.write("app_stopped")
        diagnostics.close()

    app = FastAPI(title="Phase 05 双模型工具循环对比", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    @cache
    def availability(model_id: str) -> tuple[bool, str | None]:
        spec = registry.get(model_id)
        if backend_factory is not None:
            return True, None
        try:
            read_and_verify_manifest(spec.manifest_path)
            return True, None
        except Exception as error:
            return False, str(error)

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/models")
    def models():
        result = []
        for spec in registry.models:
            is_available, reason = availability(spec.model_id)
            result.append({
                "model_id": spec.model_id,
                "stage": spec.stage,
                "artifact_kind": spec.artifact_kind,
                "available": is_available,
                "unavailable_reason": reason,
            })
        return result

    @app.post("/api/model-slots/{side}/load")
    def load_model(side: str, request: LoadModelRequest):
        if side not in {"left", "right"}:
            raise HTTPException(404, "unknown model slot")
        ok, reason = availability(request.model_id)
        if not ok:
            raise HTTPException(409, reason or "backend unavailable")
        try:
            with comparison_lock:
                model_slots.load(side, request.model_id)
        except Exception as error:
            raise HTTPException(500, f"model load failed: {error}") from error
        return {"side": side, "model_id": request.model_id, "status": "ready"}

    @app.post("/api/client-logs", status_code=204)
    def client_log(request: ClientLogRequest):
        event = "client_error" if request.level == "error" else "client_log"
        diagnostics.write(
            event,
            level=request.level,
            message=request.message,
            context=request.context,
        )

    @app.get("/api/technicians")
    def technicians():
        return {
            "version": store.version,
            "date": store.date,
            "fixture_hash": store.fixture_hash,
            "technicians": [
                {
                    "name": tech.name,
                    "gender": tech.gender,
                    "specialties": tech.specialties,
                    "availability": [
                        {"start": window.start.isoformat(" "), "end": window.end.isoformat(" ")}
                        for window in tech.availability
                    ],
                }
                for tech in store.technicians()
            ],
        }

    @app.get("/api/technicians/{name}")
    def technician(name: str):
        data = technicians()
        try:
            return next(item for item in data["technicians"] if item["name"] == name)
        except StopIteration as error:
            raise HTTPException(404, "technician not found") from error

    @app.post("/api/compare")
    def compare(request: CompareRequest):
        if request.mode not in {"sequential", "parallel"}:
            raise HTTPException(422, "unsupported comparison mode")
        comparable = request.mode == "sequential"
        request_id = uuid4().hex
        diagnostics.write(
            "comparison_started",
            request_id=request_id,
            mode=request.mode,
            left_model_id=request.left_model_id,
            right_model_id=request.right_model_id,
            user_input=request.user_input,
            left_history=request.left_history,
            right_history=request.right_history,
        )
        sides = (
            ("left", request.left_model_id, request.left_history),
            ("right", request.right_model_id, request.right_history),
        )
        for _, model_id, _ in sides:
            ok, reason = availability(model_id)
            if not ok:
                raise HTTPException(409, reason or "backend unavailable")

        def stream():
            with comparison_lock:
                for side, model_id, history in sides:
                    try:
                        diagnostics.write(
                            "side_started",
                            request_id=request_id,
                            side=side,
                            model_id=model_id,
                            history_turns=len(history),
                        )
                        backend = model_slots.get(side, model_id)
                        timed_backend = TimedBackend(backend)
                        yield encode_side_status(side, "inferencing", comparable)
                        result = ConversationOrchestrator(
                            timed_backend, executor, now_provider=now_provider
                        ).run(request.user_input, history)
                        for event in result.events:
                            yield encode_event(CompareEvent(side, event, comparable)) + "\n"
                        has_error = any(e.kind == "error" for e in result.events)
                        status = "error" if has_error else "complete"
                        diagnostics.write(
                            "side_completed",
                            request_id=request_id,
                            side=side,
                            model_id=model_id,
                            status=status,
                            inference_duration_ms=round(
                                timed_backend.inference_duration_ms, 3
                            ),
                            events=[
                                {"kind": event.kind, "payload": event.payload}
                                for event in result.events
                            ],
                        )
                        yield encode_side_status(
                            side,
                            status,
                            comparable,
                            inference_duration_ms=round(
                                timed_backend.inference_duration_ms, 3
                            ),
                        )
                    except Exception:
                        diagnostics.write(
                            "side_failed",
                            request_id=request_id,
                            side=side,
                            model_id=model_id,
                            traceback=format_exc(),
                        )
                        yield encode_side_status(
                            side, "error", comparable, inference_duration_ms=0.0
                        )

        return StreamingResponse(stream(), media_type="application/x-ndjson")

    return app
