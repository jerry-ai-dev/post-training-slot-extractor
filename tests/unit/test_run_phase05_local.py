from pathlib import Path

from scripts.eval import run_phase05_local as module
from scripts.eval.run_phase05_local import (
    evaluate_quality,
    load_phase05_config,
    manifest_payload,
    measure_workloads,
    run_matrix,
)
from slot_extractor.quantization.lineage import Lineage
from slot_extractor.quantization.manifest import StageManifest
from slot_extractor.quantization.registry import ModelRegistry
from slot_extractor.schemas.results import GenerationResult


def test_phase05_config_has_eight_q4_and_two_f16_anchors():
    config = load_phase05_config(Path("configs/evaluation/phase05.yaml"))
    registry = ModelRegistry.from_config(config.registry)
    assert tuple(model.model_id for model in registry.quantization_targets()) == config.q4_model_ids
    assert tuple(model.model_id for model in registry.anchors()) == config.f16_anchor_ids
    assert config.workloads == ("short", "medium", "2k", "4k")
    assert config.include_8k is False
    assert config.execution == "windows_cpu_sequential"


def test_evaluate_quality_runs_frozen_cases_and_returns_phase02_style_details(
    monkeypatch, tmp_path
):
    cases = tmp_path / "cases.jsonl"
    cases.write_text(
        Path("tests/fixtures/phase01_eval.jsonl").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    config = load_phase05_config(Path("configs/evaluation/phase05.yaml"))
    config = module.replace(config, quality_cases=cases)

    class Backend:
        model = "model-a"

        def generate(self, messages, params=None):
            return GenerationResult(
                '{"action":"final","gender_preference":null,"technician_gender":null,'
                '"start_time":null,"duration_minutes":null,"preferences":[],'
                '"technician_name":null,"technician_status":"not_checked",'
                '"confirmation":false,"info_complete":false,"unrelated":true,'
                '"missing_info":[],"reply_type":"handoff","reply":null}',
                self.model,
                1,
                2,
                3,
                4,
                5,
            )

    monkeypatch.setattr(module, "build_quality_backend", lambda *args: Backend())
    quality = evaluate_quality("http://local/v1", config, "model-a")
    assert quality["n"] == 3
    assert set(quality["aggregate_dimensions"]) >= {"protocol", "task_correctness"}
    assert len(quality["records"]) == 3
    assert quality["records"][0]["messages_sent"]
    assert quality["scenario_slices"]


def test_manifest_payload_serializes_nested_lineage_dataclass():
    lineage = Lineage("m", "base", "main", None, None, (), "git", ())
    manifest = StageManifest(
        "m", "verify", "complete", "q4_k_m", False, "key", lineage, (), (), (), None
    )
    payload = manifest_payload(manifest)
    assert payload["lineage"]["base_model"] == "base"


def test_run_matrix_is_sequential_and_isolates_failure(monkeypatch, tmp_path):
    config = load_phase05_config(Path("configs/evaluation/phase05.yaml"))
    failed = config.q4_model_ids[2]

    class Manager:
        base_url = "http://local/v1"

        def __init__(self):
            self.started = []
            self.stopped = []

        def start(self, model_id, log_path):
            self.started.append(model_id)
            if model_id == failed:
                raise RuntimeError("injected")
            return model_id

        def wait_ready(self, process, timeout_s):
            pass

        def stop(self, process):
            self.stopped.append(process)

    manager = Manager()
    monkeypatch.setattr(module, "read_and_verify_manifest", lambda path: {})
    monkeypatch.setattr(module, "evaluate_quality", lambda *args: {})
    monkeypatch.setattr(module, "measure_workloads", lambda *args, **kwargs: [])
    summary = run_matrix(
        Path("configs/evaluation/phase05.yaml"),
        reports_root=tmp_path,
        server_manager=manager,
    )
    assert summary.failed_model_ids == (failed,)
    assert manager.stopped == [item for item in manager.started if item != failed]


def test_default_workloads_exclude_8k_and_capture_cold_hot_fields(monkeypatch):
    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {
                "usage": {"completion_tokens": 7},
                "timings": {
                    "prompt_ms": 2,
                    "predicted_ms": 3,
                    "ttft_ms": 1,
                },
            }

    monkeypatch.setattr(module.httpx, "post", lambda *args, **kwargs: Response())
    rows = measure_workloads(
        "http://local/v1",
        None,
        ("short", "medium", "2k", "4k"),
        warmup_requests=1,
        repetitions=2,
        file_size_bytes=123,
    )
    assert {row.workload for row in rows} == {"short", "medium", "2k", "4k"}
    assert {row.phase for row in rows} == {"cold", "hot"}
    assert all(
        set(row.to_dict())
        >= {
            "load_ms",
            "prefill_ms",
            "ttft_ms",
            "decode_ms",
            "total_ms",
            "tokens",
            "peak_rss_mb",
            "file_size_bytes",
        }
        for row in rows
    )
